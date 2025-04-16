use crate::errors::Result;
use crate::gpmix::mixint::to_discrete_space;
use crate::optimizers::*;
use crate::types::*;

use crate::utils::{compute_cstr_scales, pofs, pofs_grad};
use crate::EgorSolver;

#[cfg(not(feature = "nlopt"))]
use crate::types::ObjFn;
#[cfg(feature = "nlopt")]
use nlopt::ObjFn;

use argmin::core::{CostFunction, Problem};

use egobox_doe::{Lhs, SamplingMethod};
// use finitediff::FiniteDiff;

use egobox_moe::MixtureGpSurrogate;
use log::{debug, info, warn};
use ndarray::{s, Array, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2, Zip};

use ndarray_stats::QuantileExt;
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use serde::de::DeserializeOwned;

const CSTR_DOUBT: f64 = 3.;

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
            info!("WB2S scaling factor = {}", scale_ic);
            scale_ic
        } else {
            1.
        };

        let scale_infill_obj =
            self.compute_infill_obj_scale(&scaling_points.view(), obj_model, fmin, scale_ic);
        info!(
            "Infill criterion {} scaling is updated to {}",
            self.config.infill_criterion.name(),
            scale_infill_obj
        );
        let scale_cstr = if cstr_models.is_empty() {
            Array1::zeros((0,))
        } else {
            let scale_cstr = compute_cstr_scales(&scaling_points.view(), cstr_models);
            info!(
                "Surrogated constraints scaling is updated to {}",
                scale_cstr
            );
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
            info!(
                "Fonctional constraints scaling is updated to {}",
                scale_fcstr
            );
            scale_fcstr
        };
        (scale_infill_obj, scale_cstr, scale_fcstr, scale_ic)
    }

    /// Find best x promising points by optimizing the chosen infill criterion
    /// The optimized value of the criterion is returned together with the
    /// optimum location
    /// Returns (infill_obj, x_opt)
    #[allow(clippy::too_many_arguments)]
    //#[allow(clippy::type_complexity)]
    pub(crate) fn compute_best_point(
        &self,
        sampling: &Lhs<f64, Xoshiro256Plus>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        cstr_funcs: &[impl CstrFn],
        cstr_tols: &Array1<f64>,
        lhs_optim_seed: Option<u64>,
        infill_data: &InfillObjData<f64>,
        current_best: (f64, Array1<f64>, Array1<f64>, Array1<f64>),
        actives: &Array2<usize>,
    ) -> (f64, Array1<f64>) {
        let algorithm = match self.config.infill_optimizer {
            InfillOptimizer::Slsqp => crate::optimizers::Algorithm::Slsqp,
            InfillOptimizer::Cobyla => crate::optimizers::Algorithm::Cobyla,
        };

        let mut infill_data = infill_data.clone();

        let mut best_point = (current_best.0, current_best.1.to_owned());
        let mut current_best_point = current_best.to_owned();

        for (i, active) in actives.outer_iter().enumerate() {
            let mut success = false;
            let mut n_optim = 1;
            let n_max_optim = 3;

            let active = active.to_vec();
            let obj = |x: &[f64],
                       gradient: Option<&mut [f64]>,
                       params: &mut InfillObjData<f64>|
             -> f64 {
                let InfillObjData {
                    scale_infill_obj,
                    scale_wb2,
                    xbest: xcoop,
                    fmin,
                    ..
                } = params;
                let mut xcoop = xcoop.clone();
                Self::setx(&mut xcoop, &active, x);

                // Defensive programming NlOpt::Cobyla may pass NaNs
                if xcoop.iter().any(|x| x.is_nan()) {
                    return f64::INFINITY;
                }

                if let Some(grad) = gradient {
                    // Use finite differences
                    // let f = |x: &Vec<f64>| -> f64 {
                    //     self.eval_infill_obj(x, obj_model, fmin, *scale_infill_obj, *scale_wb2)
                    // };
                    // grad[..].copy_from_slice(&x.to_vec().central_diff(&f));

                    let g_infill_obj = if self.config.cstr_infill {
                        self.eval_grad_infill_obj_with_cstrs(
                            &xcoop,
                            obj_model,
                            cstr_models,
                            cstr_tols,
                            *fmin,
                            *scale_infill_obj,
                            *scale_wb2,
                        )
                    } else {
                        self.eval_grad_infill_obj(
                            &xcoop,
                            obj_model,
                            *fmin,
                            *scale_infill_obj,
                            *scale_wb2,
                        )
                    };
                    let g_infill_obj = g_infill_obj
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| active.contains(i))
                        .map(|(_, &g)| g)
                        .collect::<Vec<_>>();
                    grad[..].copy_from_slice(&g_infill_obj);
                }
                if self.config.cstr_infill {
                    self.eval_infill_obj(&xcoop, obj_model, *fmin, *scale_infill_obj, *scale_wb2)
                        * pofs(&xcoop, cstr_models, &cstr_tols.to_vec())
                } else {
                    self.eval_infill_obj(&xcoop, obj_model, *fmin, *scale_infill_obj, *scale_wb2)
                }
            };

            let cstrs: Vec<_> = if self.config.cstr_infill {
                vec![]
            } else {
                (0..self.config.n_cstr)
                    .map(|i| {
                        let active = active.to_vec();
                        let cstr = move |x: &[f64],
                                         gradient: Option<&mut [f64]>,
                                         params: &mut InfillObjData<f64>|
                              -> f64 {
                            let InfillObjData { xbest: xcoop, .. } = params;
                            let mut xcoop = xcoop.clone();
                            Self::setx(&mut xcoop, &active, x);

                            let scale_cstr =
                                params.scale_cstr.as_ref().expect("constraint scaling")[i];
                            if self.config.cstr_strategy == ConstraintStrategy::MeanConstraint {
                                Self::mean_cstr(
                                    &*cstr_models[i],
                                    &xcoop,
                                    gradient,
                                    scale_cstr,
                                    &active,
                                )
                            } else {
                                Self::upper_trust_bound_cstr(
                                    &*cstr_models[i],
                                    &xcoop,
                                    gradient,
                                    scale_cstr,
                                    &active,
                                )
                            }
                        };
                        #[cfg(feature = "nlopt")]
                        {
                            Box::new(cstr) as Box<dyn nlopt::ObjFn<InfillObjData<f64>> + Sync>
                        }
                        #[cfg(not(feature = "nlopt"))]
                        {
                            Box::new(cstr)
                                as Box<dyn crate::types::ObjFn<InfillObjData<f64>> + Sync>
                        }
                    })
                    .collect()
            };

            // We merge metamodelized constraints and function constraints
            let mut cstr_refs: Vec<_> = cstrs.iter().map(|c| c.as_ref()).collect();
            let cstr_funcs = cstr_funcs
                .iter()
                .map(|cstr| cstr as &(dyn ObjFn<InfillObjData<f64>> + Sync))
                .collect::<Vec<_>>();
            cstr_refs.extend(cstr_funcs.clone());

            // Limits
            log::info!("self.xlimits={}", self.xlimits);
            log::info!("active={:?}", &active);
            let xlimits = Self::getx(&self.xlimits, Axis(0), &active);

            if i == 0 {
                info!("Optimize infill criterion...");
            }
            while !success && n_optim <= n_max_optim {
                let x_start = sampling.sample(self.config.n_start);
                let x_start_coop = Self::getx(&x_start, Axis(1), &active);

                if let Some(seed) = lhs_optim_seed {
                    let (y_opt, x_opt) =
                        Optimizer::new(Algorithm::Lhs, &obj, &cstr_refs, &infill_data, &xlimits)
                            .cstr_tol(cstr_tols.to_owned())
                            .seed(seed)
                            .minimize();

                    info!("LHS optimization best_x {}", x_opt);
                    best_point = (y_opt, x_opt);
                    success = true;
                } else {
                    let res = (0..self.config.n_start)
                        .into_par_iter()
                        .map(|i| {
                            debug!("Begin optim {}", i);
                            let optim_res =
                                Optimizer::new(algorithm, &obj, &cstr_refs, &infill_data, &xlimits)
                                    .xinit(&x_start_coop.row(i))
                                    .max_eval((10 * x_start_coop.len()).min(10 * MAX_EVAL_DEFAULT))
                                    .ftol_rel(1e-4)
                                    .ftol_abs(1e-4)
                                    .minimize();
                            debug!("End optim {}", i);
                            optim_res
                        })
                        .reduce(
                            || (f64::INFINITY, Array::ones((xlimits.nrows(),))),
                            |a, b| if b.0 < a.0 { b } else { a },
                        );

                    if res.0.is_nan() || res.0.is_infinite() {
                        success = false;
                    } else {
                        let mut xopt_coop = current_best_point.1.to_vec();
                        Self::setx(&mut xopt_coop, &active, &res.1.to_vec());
                        infill_data.xbest = xopt_coop.clone();
                        let xopt_coop = Array1::from(xopt_coop);

                        if crate::solver::coego::COEGO_IMPROVEMENT_CHECK {
                            let (is_better, best) = self.is_objective_improved(
                                &current_best_point,
                                &xopt_coop,
                                obj_model,
                                cstr_models,
                                cstr_tols,
                                &cstr_funcs,
                            );
                            if is_better || i == 0 {
                                if i > 0 {
                                    info!(
                                        "Partial infill criterion optim c={} has better result={} at x={}",
                                        i, best.0, xopt_coop
                                    );
                                }
                                best_point = (res.0, xopt_coop);
                                current_best_point = best;
                            }
                        } else {
                            best_point = (res.0, xopt_coop.to_owned());
                            current_best_point =
                                (res.0, xopt_coop, current_best_point.2, current_best_point.3);
                        }
                        success = true;
                    }
                }

                if n_optim == n_max_optim && !success {
                    warn!("All optimizations fail => Trigger LHS optimization");
                    let (y_opt, x_opt) =
                        Optimizer::new(Algorithm::Lhs, &obj, &cstr_refs, &infill_data, &xlimits)
                            .minimize();

                    info!("LHS optimization best_x {}", x_opt);
                    best_point = (y_opt, x_opt);
                    success = true;
                }
                n_optim += 1;
            }
        }
        best_point
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
                val.abs()
            };
        });
        if inf_count > 0 || nan_count > 0 {
            warn!(
                "Criterion scale computation warning: ({nan_count} NaN + {inf_count} Inf) / {} points",
                x.nrows()
            );
        }
        let scale = *crit_vals.max().unwrap_or(&1.0);
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
            let infill = self.eval_infill_obj(x, obj_model, fmin, scale, scale_ic);
            let pofs = pofs(x, cstr_models, &cstr_tols.to_vec());

            let infill_grad =
                Array1::from_vec(self.eval_grad_infill_obj(x, obj_model, fmin, scale, scale_ic))
                    .mapv(|v| v * pofs);
            let pofs_grad = pofs_grad(x, cstr_models, &cstr_tols.to_vec());

            let cei_grad = infill_grad.mapv(|v| v * pofs) + pofs_grad.mapv(|v| v * infill);
            cei_grad.to_vec()
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
