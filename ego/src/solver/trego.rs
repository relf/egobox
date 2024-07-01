use crate::optimizers::*;
use crate::EgorSolver;
use crate::InfillObjData;
use crate::InfillOptimizer;
use crate::SurrogateBuilder;

use egobox_doe::Lhs;
use egobox_moe::MixtureGpSurrogate;
use finitediff::FiniteDiff;
use linfa_linalg::norm::*;
use ndarray::s;
use ndarray::{Array, Array1, ArrayView1, ArrayView2};
use ndarray_stats::QuantileExt;
use rand_xoshiro::Xoshiro256Plus;

pub trait TregoStep {
    fn trego_step(
        &self,
        y_data: &ArrayView2<f64>,
        sampling: &Lhs<f64, Xoshiro256Plus>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        xbest: &ArrayView1<f64>,
        local_bounds: (f64, f64),
    ) -> Array1<f64>;
}

impl<SB: SurrogateBuilder> TregoStep for EgorSolver<SB> {
    fn trego_step(
        &self,
        y_data: &ArrayView2<f64>,
        sampling: &Lhs<f64, Xoshiro256Plus>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        xbest: &ArrayView1<f64>,
        local_bounds: (f64, f64),
    ) -> Array1<f64> {
        let f_min = y_data.min().unwrap();

        let (scale_infill_obj, scale_cstr, scale_wb2) =
            self.compute_scaling(sampling, obj_model, cstr_models, *f_min);

        let algorithm = match self.config.infill_optimizer {
            InfillOptimizer::Slsqp => crate::optimizers::Algorithm::Slsqp,
            InfillOptimizer::Cobyla => crate::optimizers::Algorithm::Cobyla,
        };

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
                        self.eval_infill_obj(x, obj_model, *f_min, *scale_infill_obj, *scale_wb2)
                    };
                    grad[..].copy_from_slice(&x.to_vec().central_diff(&f));
                }
                self.eval_infill_obj(x, obj_model, *f_min, *scale_infill_obj, *scale_wb2)
            };

        let cstrs: Vec<_> = (0..self.config.n_cstr)
            .map(|i| {
                let index = i;
                let cstr = move |x: &[f64],
                                 gradient: Option<&mut [f64]>,
                                 params: &mut InfillObjData<f64>|
                      -> f64 {
                    if let Some(grad) = gradient {
                        let grd = cstr_models[i]
                            .predict_gradients(
                                &Array::from_shape_vec((1, x.len()), x.to_vec())
                                    .unwrap()
                                    .view(),
                            )
                            .unwrap()
                            .row(0)
                            .mapv(|v| v / params.scale_cstr[index])
                            .to_vec();
                        grad[..].copy_from_slice(&grd);
                    }
                    cstr_models[index]
                        .predict(
                            &Array::from_shape_vec((1, x.len()), x.to_vec())
                                .unwrap()
                                .view(),
                        )
                        .unwrap()[[0, 0]]
                        / params.scale_cstr[index]
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
        let obj_data = InfillObjData {
            scale_infill_obj,
            scale_cstr: scale_cstr_ext,
            scale_wb2,
        };

        let (_, x_opt) = Optimizer::new(algorithm, &obj, &cons, &obj_data, &self.xlimits)
            .xinit(&xbest.view())
            .max_eval(200)
            .ftol_rel(1e-4)
            .ftol_abs(1e-4)
            .minimize();
        x_opt
    }
}
