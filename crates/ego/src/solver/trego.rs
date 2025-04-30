use crate::optimizers::*;
use crate::types::DomainConstraints;
use crate::utils::find_best_result_index_from;
use crate::utils::pofs;
use crate::utils::update_data;
use crate::ConstraintStrategy;
use crate::CstrFn;
use crate::EgorSolver;
use crate::EgorState;
use crate::InfillObjData;
use crate::InfillOptimizer;
use crate::SurrogateBuilder;

use argmin::core::CostFunction;
use argmin::core::Problem;

use egobox_doe::Lhs;
use egobox_doe::SamplingMethod;
use egobox_moe::MixtureGpSurrogate;

// use finitediff::FiniteDiff;
use log::debug;
use log::info;
use ndarray::aview1;
use ndarray::Zip;
use ndarray::{Array, Array1, Array2, ArrayView1, Axis};

use rayon::prelude::*;
use serde::de::DeserializeOwned;

#[cfg(not(feature = "nlopt"))]
use crate::types::ObjFn;
#[cfg(feature = "nlopt")]
use nlopt::ObjFn;

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + DeserializeOwned,
    C: CstrFn,
{
    /// Local step where infill criterion is optimized within trust region
    pub fn trego_step<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>> + DomainConstraints<C>,
    >(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
        models: Vec<Box<dyn MixtureGpSurrogate>>,
        infill_data: &InfillObjData<f64>,
    ) -> EgorState<f64> {
        let mut new_state = state.clone();
        let (mut x_data, mut y_data, mut c_data) = new_state.take_data().expect("DOE data");

        let best_index = new_state.best_index.unwrap();
        let y_old = y_data[[best_index, 0]];
        let rho = |sigma| sigma * sigma;
        let (obj_model, cstr_models) = models.split_first().unwrap();
        let cstr_tols = new_state.cstr_tol.clone();

        let fmin = y_data[[best_index, 0]];
        let ybest = y_data.row(best_index).to_owned();
        let xbest = x_data.row(best_index).to_owned();
        let cbest = c_data.row(best_index).to_owned();

        let pb = problem.take_problem().unwrap();
        let fcstrs = pb.fn_constraints();
        // Optimize infill criterion
        let activity = new_state.activity.clone();
        let actives = activity.unwrap_or(self.full_activity()).to_owned();
        let (infill_obj, x_opt) = self.compute_local_best_point(
            obj_model.as_ref(),
            cstr_models,
            fcstrs,
            &cstr_tols,
            None,
            infill_data,
            (fmin, xbest.to_owned(), ybest, cbest),
            &actives,
            &xbest.view(),
            (
                new_state.sigma * self.config.trego.d.0,
                new_state.sigma * self.config.trego.d.1,
            ),
        );

        problem.problem = Some(pb);

        let mut new_state = new_state.infill_value(-infill_obj);
        info!(
            "{} criterion {} max found = {}",
            if self.config.cstr_infill {
                "Constrained infill"
            } else {
                "Infill"
            },
            self.config.infill_criterion.name(),
            state.get_infill_value()
        );

        let x_new = x_opt.insert_axis(Axis(0));
        debug!(
            "x_old={} x_new={}",
            x_data.row(new_state.best_index.unwrap()),
            x_data.row(best_index)
        );
        let y_new = self.eval_obj(problem, &x_new);
        debug!(
            "y_old-y_new={}, rho={}",
            y_old - y_new[[0, 0]],
            rho(new_state.sigma)
        );

        let c_new = self.eval_problem_fcstrs(problem, &x_new);

        // Update DOE and best point
        let added = update_data(
            &mut x_data,
            &mut y_data,
            &mut c_data,
            &x_new,
            &y_new,
            &c_new,
        );
        new_state.prev_added = new_state.added;
        new_state.added += added.len();
        info!(
            "+{} point(s), total: {} points",
            added.len(),
            new_state.added
        );

        let new_best_index = find_best_result_index_from(
            best_index,
            y_data.nrows() - 1,
            &y_data,
            &c_data,
            &new_state.cstr_tol,
        );
        new_state = new_state.data((x_data, y_data, c_data));
        new_state.prev_best_index = new_state.best_index;
        new_state.best_index = Some(new_best_index);
        new_state
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_local_best_point(
        &self,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        cstr_funcs: &[impl CstrFn],
        cstr_tols: &Array1<f64>,
        lhs_optim_seed: Option<u64>,
        infill_data: &InfillObjData<f64>,
        current_best: (f64, Array1<f64>, Array1<f64>, Array1<f64>),
        actives: &Array2<usize>,
        xbest: &ArrayView1<f64>,
        local_bounds: (f64, f64),
    ) -> (f64, Array1<f64>) {
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
            let xlimits = Self::getx(&self.xlimits, Axis(0), &active);
            let xbest = Self::getx(&xbest.to_owned(), Axis(0), &active.to_vec());

            let algorithm = match self.config.infill_optimizer {
                InfillOptimizer::Slsqp => crate::optimizers::Algorithm::Slsqp,
                InfillOptimizer::Cobyla => crate::optimizers::Algorithm::Cobyla,
            };
            if i == 0 {
                info!("Optimize infill criterion...");
            }
            while !success && n_optim <= n_max_optim {
                let x_start = self.local_sampling(local_bounds, &xlimits, xbest.clone());
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
                                    .max_eval((10 * x_start_coop.len()).min(MAX_EVAL_DEFAULT))
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
                    log::warn!("All optimizations fail => Trigger LHS optimization");
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

    fn local_sampling(
        &self,
        local_bounds: (f64, f64),
        xlimits: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>,
        xbest: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>,
    ) -> Array2<f64> {
        // Draw n_start initial points (multistart optim) in the local_area
        // local_area = intersection(trust_region, xlimits)
        let mut local_area = Array2::zeros((xlimits.nrows(), xlimits.ncols()));
        Zip::from(local_area.rows_mut())
            .and(&xbest)
            .and(xlimits.rows())
            .for_each(|mut row, xb, xlims| {
                let (lo, up) = (
                    xlims[0].max(xb - local_bounds.0),
                    xlims[1].min(xb + local_bounds.1),
                );
                row.assign(&aview1(&[lo, up]))
            });
        let rng = self.rng.clone();
        let lhs = Lhs::new(&local_area)
            .kind(egobox_doe::LhsKind::Maximin)
            .with_rng(rng);
        lhs.sample(self.config.n_start)
    }
}
