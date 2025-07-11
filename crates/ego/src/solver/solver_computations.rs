use crate::errors::Result;
use crate::gpmix::mixint::to_discrete_space;
use crate::{types::*, utils};

use crate::utils::{compute_cstr_scales, logpofs, logpofs_grad, pofs, pofs_grad};
use crate::{solver::coego, EgorSolver};

use argmin::core::{CostFunction, Problem};

use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use egobox_moe::MixtureGpSurrogate;

use log::{debug, info, warn};
use ndarray::{s, stack, Array, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2, Zip};
use ndarray_rand::rand::seq::SliceRandom;

use ndarray_rand::rand::Rng;
use ndarray_stats::QuantileExt;
use rand_xoshiro::Xoshiro256Plus;
use serde::de::DeserializeOwned;

const CSTR_DOUBT: f64 = 3.;

/// LocalMultiStarter is a multistart strategy that samples points in the xlimits.
#[allow(dead_code)]
pub(crate) struct LhsMultiStarter<'a, R: Rng + Clone> {
    xlimits: &'a Array2<f64>,
    rng: R,
}

impl<R: Rng + Clone> super::solver_infill_optim::MultiStarter for LhsMultiStarter<'_, R> {
    fn multistart(&mut self, n_start: usize, active: &[usize]) -> Array2<f64> {
        let xlimits = coego::get_active_x(Axis(0), self.xlimits, active);
        let sampling = Lhs::new(&xlimits)
            .with_rng(&mut self.rng)
            .kind(LhsKind::Maximin);
        sampling.sample(n_start)
    }
}

impl<'a, R: Rng + Clone> LhsMultiStarter<'a, R> {
    #[allow(dead_code)]
    pub fn new(xlimits: &'a Array2<f64>, rng: R) -> Self {
        LhsMultiStarter { xlimits, rng }
    }
}

/// MiddlePickerMultiStarter is a multistart strategy where starting points
/// are picked in the area in between the training data points where
/// infill criterion is expected to be high
pub(crate) struct MiddlePickerMultiStarter<'a, 'b, R: Rng + Clone> {
    xlimits: &'a Array2<f64>,
    xtrain: &'b Array2<f64>,
    rng: R,
}

impl<R: Rng + Clone> super::solver_infill_optim::MultiStarter
    for MiddlePickerMultiStarter<'_, '_, R>
{
    fn multistart(&mut self, n_start: usize, active: &[usize]) -> Array2<f64> {
        let xlimits = coego::get_active_x(Axis(0), self.xlimits, active);
        let n = (n_start - 2) / 2;
        if self.xtrain.nrows() > n_start {
            let xt = self.xtrain;
            let mut indices: Vec<usize> = (0..xt.nrows()).collect();
            indices.shuffle(&mut self.rng);
            let selected: Vec<_> = indices
                .iter()
                .take(n)
                .map(|&i| xt.slice(s![i, ..]).to_owned())
                .collect();
            let vxt = selected.iter().map(|p| p.view()).collect::<Vec<_>>();
            let xt = stack(Axis(0), &vxt).unwrap();

            let xt = coego::get_active_x(Axis(1), &xt, active);
            utils::start_points(
                &xt,
                &xlimits.column(0).to_owned(),
                &xlimits.column(1).to_owned(),
            )
        } else {
            // fallback on LHS
            let sampling = Lhs::new(&xlimits)
                .with_rng(&mut self.rng)
                .kind(LhsKind::Maximin);
            sampling.sample(n_start)
        }
    }
}

impl<'a, 'b, R: Rng + Clone> MiddlePickerMultiStarter<'a, 'b, R> {
    pub fn new(xlimits: &'a Array2<f64>, xtrain: &'b Array2<f64>, rng: R) -> Self {
        MiddlePickerMultiStarter {
            xlimits,
            xtrain,
            rng,
        }
    }
}

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + DeserializeOwned,
    C: CstrFn,
{
    pub(crate) fn compute_scaling(
        &self,
        sampling: &Lhs<f64, Xoshiro256Plus>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        cstr_tols: &Array1<f64>,
        fcstrs: &[impl CstrFn],
        fmin: f64,
    ) -> (f64, Array1<f64>, Array1<f64>, f64) {
        let npts = (100 * self.xlimits.nrows()).min(1000);
        debug!("Use {npts} points to evaluate scalings");
        let scaling_points = sampling.sample(npts);

        let scale_ic = if self.config.infill_criterion.name() == "WB2S" {
            let scale_ic =
                self.config
                    .infill_criterion
                    .scaling(&scaling_points.view(), obj_model, fmin);
            info!("WB2S scaling factor = {scale_ic}");
            scale_ic
        } else {
            1.
        };

        let scale_infill_obj = self.compute_infill_obj_scale(
            &scaling_points.view(),
            obj_model,
            cstr_models,
            cstr_tols,
            fmin,
            scale_ic,
        );
        info!(
            "Infill criterion {} scaling is updated to {}",
            self.config.infill_criterion.name(),
            scale_infill_obj
        );
        let scale_cstr = if cstr_models.is_empty() {
            Array1::zeros((0,))
        } else {
            let scale_cstr = compute_cstr_scales(&scaling_points.view(), cstr_models);
            info!("Surrogated constraints scaling is updated to {scale_cstr}");
            scale_cstr
        };

        let scale_fcstr = if fcstrs.is_empty() {
            Array1::zeros((0,))
        } else {
            let fcstr_values = self.eval_fcstrs(fcstrs, &scaling_points).map(|v| v.abs());
            let mut scale_fcstr = Array1::zeros(fcstr_values.ncols());
            Zip::from(&mut scale_fcstr)
                .and(fcstr_values.columns())
                .for_each(|sc, vals| *sc = *vals.max().unwrap());
            info!("Fonctional constraints scaling is updated to {scale_fcstr}");
            scale_fcstr
        };
        (scale_infill_obj, scale_cstr, scale_fcstr, scale_ic)
    }

    pub fn mean_cstr(
        cstr_model: &dyn MixtureGpSurrogate,
        x: &[f64],
        gradient: Option<&mut [f64]>,
        scale_cstr: f64,
        active: &[usize],
    ) -> f64 {
        let x = Array::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
        if let Some(grad) = gradient {
            let grd = cstr_model
                .predict_gradients(&x.view())
                .unwrap()
                .row(0)
                .mapv(|v| v / scale_cstr)
                .to_vec();
            let grd_coop = grd
                .iter()
                .enumerate()
                .filter(|(i, _)| active.contains(i))
                .map(|(_, &g)| g)
                .collect::<Vec<_>>();
            grad[..].copy_from_slice(&grd_coop);
        }
        cstr_model.predict(&x.view()).unwrap()[0] / scale_cstr
    }

    pub fn upper_trust_bound_cstr(
        cstr_model: &dyn MixtureGpSurrogate,
        x: &[f64],
        gradient: Option<&mut [f64]>,
        scale_cstr: f64,
        active: &[usize],
    ) -> f64 {
        let x = Array::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
        let sigma = cstr_model.predict_var(&x.view()).unwrap()[[0, 0]].sqrt();
        let cstr_val = cstr_model.predict(&x.view()).unwrap()[0];

        if let Some(grad) = gradient {
            let sigma_prime = if sigma < f64::EPSILON {
                0.
            } else {
                cstr_model.predict_var_gradients(&x.view()).unwrap()[[0, 0]] / (2. * sigma)
            };
            let grd = cstr_model
                .predict_gradients(&x.view())
                .unwrap()
                .row(0)
                .mapv(|v| (v + CSTR_DOUBT * sigma_prime) / scale_cstr)
                .to_vec();
            let grd_coop = grd
                .iter()
                .enumerate()
                .filter(|(i, _)| active.contains(i))
                .map(|(_, &g)| g)
                .collect::<Vec<_>>();
            grad[..].copy_from_slice(&grd_coop);
        }
        (cstr_val + CSTR_DOUBT * sigma) / scale_cstr
    }

    /// Return the virtual points regarding the qei strategy
    /// The default is to return GP prediction
    pub(crate) fn compute_virtual_point(
        &self,
        xk: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
    ) -> Result<Vec<f64>> {
        let mut res: Vec<f64> = vec![];
        if self.config.q_ei == QEiStrategy::ConstantLiarMinimum {
            let index_min = y_data.slice(s![.., 0_usize]).argmin().unwrap();
            res.push(y_data[[index_min, 0]]);
            for ic in 1..=self.config.n_cstr {
                res.push(y_data[[index_min, ic]]);
            }
            Ok(res)
        } else {
            let x = &xk.view().insert_axis(Axis(0));
            let pred = obj_model.predict(x)?[0];
            let var = obj_model.predict_var(x)?[[0, 0]];
            let conf = match self.config.q_ei {
                QEiStrategy::KrigingBeliever => 0.,
                QEiStrategy::KrigingBelieverLowerBound => -3.,
                QEiStrategy::KrigingBelieverUpperBound => 3.,
                _ => -1., // never used
            };
            res.push(pred + conf * f64::sqrt(var));
            for cstr_model in cstr_models {
                res.push(cstr_model.predict(x)?[0]);
            }
            Ok(res)
        }
    }

    /// The infill criterion scaling is computed using x (n points of nx dim)
    /// given the objective function surrogate
    pub(crate) fn compute_infill_obj_scale(
        &self,
        x: &ArrayView2<f64>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        cstr_tols: &Array1<f64>,
        fmin: f64,
        scale_ic: f64,
    ) -> f64 {
        let mut crit_vals = Array1::zeros(x.nrows());
        let (mut nan_count, mut inf_count) = (0, 0);
        Zip::from(&mut crit_vals).and(x.rows()).for_each(|c, x| {
            let val = self.eval_infill_obj(&x.to_vec(), obj_model, fmin, 1.0, scale_ic);
            *c = if val.is_nan() {
                nan_count += 1;
                1.0
            } else if val.is_infinite() {
                inf_count += 1;
                1.0
            } else {
                val
            };
        });
        if self.config.cstr_infill {
            Zip::from(&mut crit_vals).and(x.rows()).for_each(|c, x| {
                if self.config.infill_criterion.name() == "LogEI" {
                    *c -= logpofs(&x.to_vec(), cstr_models, &cstr_tols.to_vec());
                } else {
                    *c *= pofs(&x.to_vec(), cstr_models, &cstr_tols.to_vec());
                }
            });
        }

        if inf_count > 0 || nan_count > 0 {
            warn!(
                "Criterion scale computation warning: ({nan_count} NaN + {inf_count} Inf) / {} points",
                x.nrows()
            );
        }
        let scale = *crit_vals.mapv(|v| v.abs()).max().unwrap_or(&1.0);
        if scale < 100.0 * f64::EPSILON {
            1.0
        } else {
            scale
        }
    }

    /// Compute infill criterion objective expected to be minimized
    /// meaning infill criterion objective is negative infill criterion
    /// the latter is expected to be maximized
    pub(crate) fn eval_infill_obj(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        scale: f64,
        scale_ic: f64,
    ) -> f64 {
        let x_f = x.to_vec();
        let obj = -(self
            .config
            .infill_criterion
            .value(&x_f, obj_model, fmin, Some(scale_ic)));
        obj / scale
    }

    pub fn eval_grad_infill_obj(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        scale: f64,
        scale_ic: f64,
    ) -> Vec<f64> {
        let x_f = x.to_vec();
        let grad = -(self
            .config
            .infill_criterion
            .grad(&x_f, obj_model, fmin, Some(scale_ic)));
        (grad / scale).to_vec()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn eval_infill_obj_with_cstrs(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        cstr_tols: &Array1<f64>,
        fmin: f64,
        scale: f64,
        scale_ic: f64,
    ) -> f64 {
        let infill_obj = self.eval_infill_obj(x, obj_model, fmin, scale, scale_ic);
        if self.config.infill_criterion.name() == "LogEI" {
            infill_obj - logpofs(x, cstr_models, &cstr_tols.to_vec())
        } else {
            infill_obj * pofs(x, cstr_models, &cstr_tols.to_vec())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn eval_grad_infill_obj_with_cstrs(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        cstr_tols: &Array1<f64>,
        fmin: f64,
        scale: f64,
        scale_ic: f64,
    ) -> Vec<f64> {
        if cstr_models.is_empty() {
            self.eval_grad_infill_obj(x, obj_model, fmin, scale, scale_ic)
        } else {
            let infill_grad =
                Array1::from_vec(self.eval_grad_infill_obj(x, obj_model, fmin, scale, scale_ic));

            if self.config.infill_criterion.name() == "LogEI" {
                let logcei_grad = infill_grad - logpofs_grad(x, cstr_models, &cstr_tols.to_vec());
                logcei_grad.to_vec()
            } else {
                let infill = self.eval_infill_obj(x, obj_model, fmin, scale, scale_ic);
                let infill_grad = Array1::from_vec(
                    self.eval_grad_infill_obj(x, obj_model, fmin, scale, scale_ic),
                );

                let pofs = pofs(x, cstr_models, &cstr_tols.to_vec());
                let pofs_grad = pofs_grad(x, cstr_models, &cstr_tols.to_vec());

                let cei_grad = infill_grad * pofs + pofs_grad * infill;
                cei_grad.to_vec()
            }
        }
    }

    pub fn eval_obj<O: CostFunction<Param = Array2<f64>, Output = Array2<f64>>>(
        &self,
        pb: &mut Problem<O>,
        x: &Array2<f64>,
    ) -> Array2<f64> {
        let x = if self.config.discrete() {
            // We have to cast x to folded space as EgorSolver
            // works internally in the continuous space while
            // the objective function expects discrete variable in folded space
            to_discrete_space(&self.config.xtypes, x)
        } else {
            x.to_owned()
        };
        pb.problem("cost_count", |problem| problem.cost(&x))
            .expect("Objective evaluation")
    }

    pub fn eval_fcstrs(
        &self,
        fcstrs: &[impl CstrFn],
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Array2<f64> {
        let mut unused = InfillObjData::default();

        let mut res = Array2::zeros((x.nrows(), fcstrs.len()));
        Zip::from(res.rows_mut())
            .and(x.rows())
            .for_each(|mut r, xi| {
                let cstr_vals = &fcstrs
                    .iter()
                    .map(|cstr| {
                        let xuser = if self.config.discrete() {
                            let xary = xi.to_owned().insert_axis(Axis(0));
                            // We have to cast x to folded space as EgorSolver
                            // works internally in the continuous space while
                            // the constraint function expects discrete variable in folded space
                            to_discrete_space(&self.config.xtypes, &xary)
                                .row(0)
                                .into_owned();
                            xary.into_iter().collect::<Vec<_>>()
                        } else {
                            xi.to_vec()
                        };
                        cstr(&xuser, None, &mut unused)
                    })
                    .collect::<Array1<_>>();
                r.assign(cstr_vals)
            });

        res
    }

    pub fn eval_problem_fcstrs<O: DomainConstraints<C>>(
        &self,
        pb: &mut Problem<O>,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Array2<f64> {
        let problem = pb.take_problem().unwrap();
        let fcstrs = problem.fn_constraints();

        let res = self.eval_fcstrs(fcstrs, x);

        pb.problem = Some(problem);
        res
    }
}
