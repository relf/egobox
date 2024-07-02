use crate::optimizers::*;
use crate::utils::find_best_result_index_from;
use crate::utils::update_data;
use crate::EgorSolver;
use crate::EgorState;
use crate::InfillObjData;
use crate::SurrogateBuilder;

use argmin::core::CostFunction;
use argmin::core::Problem;

use egobox_doe::Lhs;
use egobox_doe::SamplingMethod;
use egobox_moe::MixtureGpSurrogate;

use finitediff::FiniteDiff;
use linfa_linalg::norm::*;
use log::{debug, info};
use ndarray::{s, Array, Array1, Array2, ArrayView1, Axis};

use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;

impl<SB: SurrogateBuilder> EgorSolver<SB> {
    #[allow(clippy::too_many_arguments)]
    pub fn trego_step<O: CostFunction<Param = Array2<f64>, Output = Array2<f64>>>(
        &mut self,
        fobj: &mut Problem<O>,
        models: Vec<Box<dyn MixtureGpSurrogate>>,
        sampling: Lhs<f64, rand_xoshiro::Xoshiro256Plus>,
        best_index: usize,
        x_data: &mut ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>,
        y_data: &mut ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>,
        state: &EgorState<f64>,
        new_state: &mut EgorState<f64>,
        infill_data: &InfillObjData<f64>,
    ) -> usize {
        let y_new = y_data[[best_index, 0]];
        let y_old = y_data[[state.best_index.unwrap(), 0]];
        let rho = |sigma| sigma * sigma;
        if y_new < y_old - rho(new_state.sigma) {
            info!("Ego global step successful!");
            best_index
        } else {
            debug!("Trego local step");
            let mut new_best_index = best_index;
            (0..self.config.trego.n_local_steps).for_each(|_| {
                let (obj_model, cstr_models) = models.split_first().unwrap();
                let xbest = x_data.row(best_index);
                let x_opt = self.local_step(
                    &sampling,
                    obj_model.as_ref(),
                    cstr_models,
                    &xbest,
                    (
                        new_state.sigma * self.config.trego.d.0,
                        new_state.sigma * self.config.trego.d.1,
                    ),
                    infill_data,
                );
                let x_new = x_opt.insert_axis(Axis(0));
                debug!(
                    "x_old={} x_new={}",
                    x_data.row(state.best_index.unwrap()),
                    x_data.row(best_index)
                );
                let y_new = self.eval_obj(fobj, &x_new);
                debug!(
                    "y_old-y_new={}, rho={}",
                    y_old - y_new[[0, 0]],
                    rho(new_state.sigma)
                );
                if y_new[[0, 0]] < y_old - rho(new_state.sigma) {
                    let new_index = update_data(x_data, y_data, &x_new, &y_new);
                    if new_index.len() == 1 {
                        let new_index = find_best_result_index_from(
                            best_index,
                            y_data.nrows() - 1,
                            &*y_data,
                            &state.cstr_tol,
                        );
                        if new_index == y_data.nrows() - 1 {
                            // trego local step successful
                            new_best_index = new_index;
                        }
                    }
                }
                if new_best_index == best_index {
                    let old = new_state.sigma;
                    new_state.sigma *= self.config.trego.beta;
                    debug!(
                        "Trego Local step not successful: sigma {} -> {}",
                        old, new_state.sigma
                    );
                } else {
                    let old = new_state.sigma;
                    new_state.sigma *= self.config.trego.gamma;
                    info!(
                        "Trego local step successful: sigma {} -> {}",
                        old, new_state.sigma
                    );
                }
            });
            new_best_index
        }
    }

    fn local_step(
        &self,
        sampling: &Lhs<f64, Xoshiro256Plus>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        xbest: &ArrayView1<f64>,
        local_bounds: (f64, f64),
        infill_data: &InfillObjData<f64>,
    ) -> Array1<f64> {
        let InfillObjData {
            fmin, scale_cstr, ..
        } = infill_data;
        let scale_cstr = scale_cstr
            .clone()
            .unwrap_or(Array1::from_elem(cstr_models.len(), 1.));
        let obj =
            |x: &[f64], gradient: Option<&mut [f64]>, params: &mut InfillObjData<f64>| -> f64 {
                // Defensive programming NlOpt::Cobyla may pass NaNs
                if x.iter().any(|x| x.is_nan()) {
                    return f64::INFINITY;
                }
                let InfillObjData {
                    scale_infill_obj,
                    scale_wb2,
                    ..
                } = params;
                if let Some(grad) = gradient {
                    let f = |x: &Vec<f64>| -> f64 {
                        self.eval_infill_obj(x, obj_model, *fmin, *scale_infill_obj, *scale_wb2)
                    };
                    grad[..].copy_from_slice(&x.to_vec().central_diff(&f));
                }
                self.eval_infill_obj(x, obj_model, *fmin, *scale_infill_obj, *scale_wb2)
            };

        let cstrs: Vec<_> = (0..self.config.n_cstr)
            .map(|i| {
                let index = i;
                let cstr = move |x: &[f64],
                                 gradient: Option<&mut [f64]>,
                                 params: &mut InfillObjData<f64>|
                      -> f64 {
                    if let Some(grad) = gradient {
                        let scale_cstr = params.scale_cstr.as_ref().expect("constraint scaling")[i];
                        let grd = cstr_models[i]
                            .predict_gradients(
                                &Array::from_shape_vec((1, x.len()), x.to_vec())
                                    .unwrap()
                                    .view(),
                            )
                            .unwrap()
                            .row(0)
                            .mapv(|v| v / scale_cstr)
                            .to_vec();
                        grad[..].copy_from_slice(&grd);
                    }
                    let scale_cstr = params.scale_cstr.as_ref().expect("constraint scaling")[index];
                    cstr_models[index]
                        .predict(
                            &Array::from_shape_vec((1, x.len()), x.to_vec())
                                .unwrap()
                                .view(),
                        )
                        .unwrap()[[0, 0]]
                        / scale_cstr
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
        let cstr_refs: Vec<_> = cstrs.iter().map(|c| c.as_ref()).collect();

        let cstr_up = Box::new(
            |x: &[f64], gradient: Option<&mut [f64]>, _params: &mut InfillObjData<f64>| -> f64 {
                let f = |x: &Vec<f64>| -> f64 {
                    let x = Array1::from_shape_vec((x.len(),), x.to_vec()).unwrap();
                    let d = (&x - xbest).norm_l2();
                    d - local_bounds.1
                };
                if let Some(grad) = gradient {
                    grad[..].copy_from_slice(&x.to_vec().central_diff(&f));
                }
                f(&x.to_vec())
            },
        ) as Box<dyn crate::types::ObjFn<InfillObjData<f64>> + Sync>;

        let cstr_lo = Box::new(
            |x: &[f64], gradient: Option<&mut [f64]>, _params: &mut InfillObjData<f64>| -> f64 {
                let f = |x: &Vec<f64>| -> f64 {
                    let x = Array1::from_shape_vec((x.len(),), x.to_vec()).unwrap();
                    let d = (&x - xbest).norm_l2();
                    local_bounds.0 - d
                };
                if let Some(grad) = gradient {
                    grad[..].copy_from_slice(&x.to_vec().central_diff(&f));
                }
                f(&x.to_vec())
            },
        ) as Box<dyn crate::types::ObjFn<InfillObjData<f64>> + Sync>;

        let mut cons = cstr_refs.to_vec();
        cons.push(&cstr_lo);
        cons.push(&cstr_up);

        let mut scale_cstr_ext = Array1::zeros(scale_cstr.len() + 2);
        scale_cstr_ext
            .slice_mut(s![..scale_cstr.len()])
            .assign(&scale_cstr);
        let infill_data = InfillObjData {
            scale_cstr: Some(scale_cstr_ext),
            ..*infill_data
        };

        let x_start = sampling.sample(self.config.n_start);

        let algorithm = crate::optimizers::Algorithm::Slsqp;
        let (_, x_opt) = (0..self.config.n_start)
            .into_par_iter()
            .map(|i| {
                Optimizer::new(algorithm, &obj, &cstr_refs, &infill_data, &self.xlimits)
                    .xinit(&x_start.row(i))
                    .max_eval(200)
                    .ftol_rel(1e-4)
                    .ftol_abs(1e-4)
                    .minimize()
            })
            .reduce(
                || (f64::INFINITY, Array::ones((xbest.len(),))),
                |a, b| if b.0 < a.0 { b } else { a },
            );

        x_opt
    }
}
