use crate::optimizers::*;
use crate::types::*;

use crate::utils::pofs;
use crate::EgorSolver;

#[cfg(not(feature = "nlopt"))]
use crate::types::ObjFn;
#[cfg(feature = "nlopt")]
use nlopt::ObjFn;

use egobox_moe::MixtureGpSurrogate;
use log::info;
use ndarray::{Array, Array1, Array2, Axis};

use rayon::prelude::*;
use serde::de::DeserializeOwned;

use super::coego;

/// A trait for multi start initial points computation
pub(crate) trait MultiStarter {
    /// Return initial points for optimization multistart
    /// taking into account active components given as a set of indices
    fn multistart(&mut self, n_start: usize, active: &[usize]) -> Array2<f64>;
}

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + DeserializeOwned,
    C: CstrFn,
{
    /// Find best x promising points by optimizing the chosen infill criterion
    /// The optimized value of the criterion is returned together with the
    /// optimum location
    /// Returns (infill_obj, x_opt)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn optimize_infill_criterion<MS>(
        &self,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        cstr_funcs: &[impl CstrFn],
        cstr_tols: &Array1<f64>,
        infill_data: &InfillObjData<f64>,
        current_best: (f64, Array1<f64>, Array1<f64>, Array1<f64>),
        actives: &Array2<usize>,
        mut multistarter: MS,
    ) -> (f64, Array1<f64>)
    where
        MS: MultiStarter,
    {
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
                coego::set_active_x(&mut xcoop, &active, x);

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
                            coego::set_active_x(&mut xcoop, &active, x);

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

            // Limits of activated components
            let xlimits_active = coego::get_active_x(Axis(0), &self.xlimits, &active);

            let algorithm = match self.config.infill_optimizer {
                InfillOptimizer::Slsqp => crate::optimizers::Algorithm::Slsqp,
                InfillOptimizer::Cobyla => crate::optimizers::Algorithm::Cobyla,
                InfillOptimizer::Gbnm => crate::optimizers::Algorithm::Gbnm,
            };

            if i == 0 {
                info!("Optimize infill criterion...");
            }
            while !success && n_optim <= n_max_optim {
                let x_start = multistarter.multistart(self.config.n_start, &active);
                let res = (0..x_start.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let optim_res = Optimizer::new(
                            algorithm,
                            &obj,
                            &cstr_refs,
                            &infill_data,
                            &xlimits_active,
                        )
                        .xinit(&x_start.row(i))
                        .max_eval((10 * x_start.len()).min(INFILL_MAX_EVAL_DEFAULT))
                        .ftol_rel(1e-4)
                        .ftol_abs(1e-4)
                        .minimize();
                        optim_res
                    })
                    .reduce(
                        || (f64::INFINITY, Array::ones((xlimits_active.nrows(),))),
                        |a, b| if b.0 < a.0 { b } else { a },
                    );

                if res.0.is_nan() || res.0.is_infinite() {
                    success = false;
                } else {
                    let mut xopt_coop = current_best_point.1.to_vec();
                    coego::set_active_x(&mut xopt_coop, &active, &res.1.to_vec());
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
                n_optim += 1;
            }
        }
        best_point
    }
}
