use crate::optimizers::*;
use crate::types::*;
use crate::utils::pofs;
use crate::EgorSolver;

#[cfg(not(feature = "nlopt"))]
use crate::types::ObjFn;
use ndarray::Axis;
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::SeedableRng;
#[cfg(feature = "nlopt")]
use nlopt::ObjFn;

use egobox_doe::{Lhs, SamplingMethod};

use egobox_moe::MixtureGpSurrogate;
use log::{debug, info};
use ndarray::{Array, Array1, Array2};

use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use serde::de::DeserializeOwned;

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
    //#[allow(clippy::type_complexity)]
    pub(crate) fn compute_best_point_coop(
        &self,
        sampling: &Lhs<f64, Xoshiro256Plus>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        cstr_tols: &Array1<f64>,
        lhs_optim_seed: Option<u64>,
        infill_data: &InfillObjData<f64>,
        cstr_funcs: &[&(dyn ObjFn<InfillObjData<f64>> + Sync)],
        current_best: (f64, Array1<f64>),
    ) -> (f64, Array1<f64>) {
        let fmin = infill_data.fmin;

        let mut success = false;
        let mut n_optim = 1;
        let n_max_optim = 3;
        let mut best_point = current_best.to_owned();

        let rng = &mut Xoshiro256Plus::seed_from_u64(42);
        let g_size = current_best.1.len() / self.config.coego.n_coop.max(1);
        let mut indices: Vec<usize> = (0..current_best.1.len()).collect();
        indices.shuffle(rng);
        let actives = Array2::from_shape_vec(
            (self.config.coego.n_coop, g_size),
            indices[..(self.config.coego.n_coop * g_size)].to_vec(),
        )
        .unwrap();

        // TODO: manage remaining indices, atm suppose an empty remainder
        let _remainder = indices[(self.config.coego.n_coop * g_size)..].to_vec();

        let algorithm = match self.config.infill_optimizer {
            InfillOptimizer::Slsqp => crate::optimizers::Algorithm::Slsqp,
            InfillOptimizer::Cobyla => crate::optimizers::Algorithm::Cobyla,
        };

        for active in actives.outer_iter() {
            let obj =
                |x: &[f64], gradient: Option<&mut [f64]>, params: &mut InfillObjData<f64>| -> f64 {
                    let InfillObjData {
                        scale_infill_obj,
                        scale_wb2,
                        xbest: xcoop,
                        ..
                    } = params;
                    let mut xcoop = xcoop.clone();
                    setx(&mut xcoop, &active.to_vec(), x);

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
                                fmin,
                                *scale_infill_obj,
                                *scale_wb2,
                            )
                        } else {
                            self.eval_grad_infill_obj(
                                &xcoop,
                                obj_model,
                                fmin,
                                *scale_infill_obj,
                                *scale_wb2,
                            )
                        };
                        grad[..].copy_from_slice(&g_infill_obj);
                    }
                    if self.config.cstr_infill {
                        self.eval_infill_obj(&xcoop, obj_model, fmin, *scale_infill_obj, *scale_wb2)
                    } else {
                        self.eval_infill_obj(&xcoop, obj_model, fmin, *scale_infill_obj, *scale_wb2)
                            * pofs(x, cstr_models, &cstr_tols.to_vec())
                    }
                };

            let cstrs: Vec<_> = (0..self.config.n_cstr)
                .map(|i| {
                    let cstr = move |x: &[f64],
                                     gradient: Option<&mut [f64]>,
                                     params: &mut InfillObjData<f64>|
                          -> f64 {
                        let InfillObjData { xbest: xcoop, .. } = params;
                        let mut xcoop = xcoop.clone();
                        setx(&mut xcoop, &active.to_vec(), x);

                        let scale_cstr = params.scale_cstr.as_ref().expect("constraint scaling")[i];
                        if self.config.cstr_strategy == ConstraintStrategy::MeanConstraint {
                            Self::mean_cstr(&*cstr_models[i], &xcoop, gradient, scale_cstr)
                        } else {
                            Self::upper_trust_bound_cstr(
                                &*cstr_models[i],
                                &xcoop,
                                gradient,
                                scale_cstr,
                            )
                        }
                    };
                    #[cfg(feature = "nlopt")]
                    {
                        Box::new(cstr) as Box<dyn nlopt::ObjFn<InfillObjData<f64>> + Sync>
                    }
                    #[cfg(not(feature = "nlopt"))]
                    {
                        Box::new(cstr) as Box<dyn crate::types::ObjFn<InfillObjData<f64>> + Sync>
                    }
                })
                .collect();

            // We merge metamodelized constraints and function constraints
            let mut cstr_refs: Vec<_> = cstrs.iter().map(|c| c.as_ref()).collect();
            cstr_refs.extend(cstr_funcs);

            info!("Optimize infill criterion...");
            while !success && n_optim <= n_max_optim {
                let x_start = sampling.sample(self.config.n_start);
                let x_start_coop = x_start.select(Axis(1), &active.to_vec());

                if let Some(seed) = lhs_optim_seed {
                    let (y_opt, x_opt) = Optimizer::new(
                        Algorithm::Lhs,
                        &obj,
                        &cstr_refs,
                        infill_data,
                        &self.xlimits,
                    )
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
                            let optim_res = Optimizer::new(
                                algorithm,
                                &obj,
                                &cstr_refs,
                                infill_data,
                                &self.xlimits,
                            )
                            .xinit(&x_start_coop.row(i))
                            .max_eval(200)
                            .ftol_rel(1e-4)
                            .ftol_abs(1e-4)
                            .minimize();
                            debug!("End optim {}", i);
                            optim_res
                        })
                        .reduce(
                            || (f64::INFINITY, Array::ones((self.xlimits.nrows(),))),
                            |a, b| if b.0 < a.0 { b } else { a },
                        );

                    if res.0.is_nan() || res.0.is_infinite() {
                        success = false;
                    } else {
                        let mut xopt_coop = best_point.1.to_vec();
                        setx(&mut xopt_coop, &active.to_vec(), &res.1.to_vec());
                        best_point = (res.0, Array1::from(xopt_coop));
                        success = true;
                    }
                }

                if n_optim == n_max_optim && !success {
                    info!("All optimizations fail => Trigger LHS optimization");
                    let (y_opt, x_opt) = Optimizer::new(
                        Algorithm::Lhs,
                        &obj,
                        &cstr_refs,
                        infill_data,
                        &self.xlimits,
                    )
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
}

/// Set active components to xcoop using xopt values
/// active and values must have the same size
fn setx(xcoop: &mut [f64], active: &[usize], values: &[f64]) {
    std::iter::zip(active, values).for_each(|(&i, &xi)| xcoop[i] = xi)
}
