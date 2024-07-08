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
use log::debug;
use log::info;
use ndarray::aview1;
use ndarray::Zip;
use ndarray::{s, Array, Array1, Array2, ArrayView1, Axis};

use rayon::prelude::*;

impl<SB: SurrogateBuilder> EgorSolver<SB> {
    /// Local step where infill criterion is optimized within trust region
    pub fn trego_step<O: CostFunction<Param = Array2<f64>, Output = Array2<f64>>>(
        &mut self,
        fobj: &mut Problem<O>,
        state: EgorState<f64>,
        models: Vec<Box<dyn MixtureGpSurrogate>>,
        infill_data: &InfillObjData<f64>,
    ) -> EgorState<f64> {
        let mut new_state = state.clone();
        let (mut x_data, mut y_data) = new_state.take_data().expect("DOE data");

        let best_index = new_state.best_index.unwrap();
        let y_old = y_data[[best_index, 0]];
        let rho = |sigma| sigma * sigma;
        let (obj_model, cstr_models) = models.split_first().unwrap();
        let xbest = x_data.row(best_index).to_owned();

        // Optimize infill criterion
        let (infill_obj, x_opt) = self.local_step(
            obj_model.as_ref(),
            cstr_models,
            &xbest.view(),
            (
                new_state.sigma * self.config.trego.d.0,
                new_state.sigma * self.config.trego.d.1,
            ),
            infill_data,
        );
        let mut new_state = new_state.infill_value(-infill_obj);
        info!(
            "Infill criterion {} max found = {}",
            self.config.infill_criterion.name(),
            state.get_infill_value()
        );

        let x_new = x_opt.insert_axis(Axis(0));
        debug!(
            "x_old={} x_new={}",
            x_data.row(new_state.best_index.unwrap()),
            x_data.row(best_index)
        );
        let y_new = self.eval_obj(fobj, &x_new);
        debug!(
            "y_old-y_new={}, rho={}",
            y_old - y_new[[0, 0]],
            rho(new_state.sigma)
        );

        // Update DOE and best point
        let added = update_data(&mut x_data, &mut y_data, &x_new, &y_new);
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
            &new_state.cstr_tol,
        );
        new_state = new_state.data((x_data, y_data));
        new_state.best_index = Some(new_best_index);
        new_state
    }

    fn local_step(
        &self,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        xbest: &ArrayView1<f64>,
        local_bounds: (f64, f64),
        infill_data: &InfillObjData<f64>,
    ) -> (f64, Array1<f64>) {
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

        // Other way to constrain in trust-region with global l
        // let cstr_up = Box::new(
        //     |x: &[f64], gradient: Option<&mut [f64]>, _params: &mut InfillObjData<f64>| -> f64 {
        //         let f = |x: &Vec<f64>| -> f64 {
        //             let x = Array1::from_shape_vec((x.len(),), x.to_vec()).unwrap();
        //             let d = (&x - xbest).norm_l1();
        //             d - local_bounds.1
        //         };
        //         if let Some(grad) = gradient {
        //             grad[..].copy_from_slice(&x.to_vec().central_diff(&f));
        //         }
        //         f(&x.to_vec())
        //     },
        // ) as Box<dyn crate::types::ObjFn<InfillObjData<f64>> + Sync>;

        // let cstr_lo = Box::new(
        //     |x: &[f64], gradient: Option<&mut [f64]>, _params: &mut InfillObjData<f64>| -> f64 {
        //         let f = |x: &Vec<f64>| -> f64 {
        //             let x = Array1::from_shape_vec((x.len(),), x.to_vec()).unwrap();
        //             let d = (&x - xbest).norm_l1();
        //             local_bounds.0 - d
        //         };
        //         if let Some(grad) = gradient {
        //             grad[..].copy_from_slice(&x.to_vec().central_diff(&f));
        //         }
        //         f(&x.to_vec())
        //     },
        // ) as Box<dyn crate::types::ObjFn<InfillObjData<f64>> + Sync>;

        // let mut cons = cstr_refs.to_vec();
        // cons.push(&cstr_lo);
        // cons.push(&cstr_up);

        let mut scale_cstr_ext = Array1::zeros(scale_cstr.len() + 2);
        scale_cstr_ext
            .slice_mut(s![..scale_cstr.len()])
            .assign(&scale_cstr);
        let infill_data = InfillObjData {
            scale_cstr: Some(scale_cstr_ext),
            ..*infill_data
        };

        // local_area = intersection(trust_region, xlimits)
        let mut local_area = Array2::zeros((self.xlimits.nrows(), self.xlimits.ncols()));
        Zip::from(local_area.rows_mut())
            .and(xbest)
            .and(self.xlimits.rows())
            .for_each(|mut row, xb, xlims| {
                let (lo, up) = (
                    xlims[0].max(xb - local_bounds.0),
                    xlims[1].min(xb + local_bounds.1),
                );
                row.assign(&aview1(&[lo, up]))
            });

        // Draw n_start initial points (multistart optim)
        let rng = self.rng.clone();
        let lhs = Lhs::new(&local_area)
            .kind(egobox_doe::LhsKind::Maximin)
            .with_rng(rng);
        let x_start = lhs.sample(self.config.n_start);

        // Find best x promising points by optimizing the chosen infill criterion
        // The optimized value of the criterion is returned together with the
        // optimum location
        // Returns (infill_obj, x_opt)
        let algorithm = crate::optimizers::Algorithm::Slsqp;
        info!("Optimize infill criterion...");
        let (infill_obj, x_opt) = (0..self.config.n_start)
            .into_par_iter()
            .map(|i| {
                Optimizer::new(algorithm, &obj, &cstr_refs, &infill_data, &local_area)
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

        (infill_obj, x_opt)
    }
}
