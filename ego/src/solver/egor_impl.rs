use crate::errors::{EgoError, Result};
use crate::gpmix::mixint::{as_continuous_limits, to_discrete_space};
use crate::types::*;
use crate::utils::compute_cstr_scales;
use crate::{optimizers::*, EgorConfig};
use crate::{EgorSolver, DEFAULT_CSTR_TOL};

use argmin::core::{CostFunction, Problem};
use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use egobox_gp::ThetaTuning;
use env_logger::{Builder, Env};
use finitediff::FiniteDiff;

use egobox_moe::{Clustering, MixtureGpSurrogate};
use log::{debug, info, warn};
use ndarray::{
    concatenate, s, Array, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2, Zip,
};
use ndarray_stats::QuantileExt;
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;

impl<SB: SurrogateBuilder> EgorSolver<SB> {
    /// Constructor of the optimization of the function `f` with specified random generator
    /// to get reproducibility.
    ///
    /// The function `f` should return an objective but also constraint values if any.
    /// Design space is specified by a list of types for input variables `x` of `f` (see [`XType`]).
    pub fn new(config: EgorConfig, rng: Xoshiro256Plus) -> Self {
        let env = Env::new().filter_or("EGOBOX_LOG", "info");
        let mut builder = Builder::from_env(env);
        let builder = builder.target(env_logger::Target::Stdout);
        builder.try_init().ok();
        let xtypes = config.xtypes.clone();
        EgorSolver {
            config,
            xlimits: as_continuous_limits(&xtypes),
            surrogate_builder: SB::new_with_xtypes(&xtypes),
            rng,
        }
    }

    /// Given an evaluated doe (x, y) data, return the next promising x point
    /// where optimum may occurs regarding the infill criterium.
    /// This function inverse the control of the optimization and can used
    /// ask-and-tell interface to the EGO optimizer.
    pub fn suggest(
        &self,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Array2<f64> {
        let rng = self.rng.clone();
        let sampling = Lhs::new(&self.xlimits).with_rng(rng).kind(LhsKind::Maximin);
        let mut clusterings = vec![None; 1 + self.config.n_cstr];
        let mut theta_tunings = vec![None; 1 + self.config.n_cstr];
        let cstr_tol = self
            .config
            .cstr_tol
            .clone()
            .unwrap_or(Array1::from_elem(self.config.n_cstr, DEFAULT_CSTR_TOL));
        let (x_dat, _, _, _) = self.next_points(
            true,
            0,
            false, // done anyway
            &mut clusterings,
            &mut theta_tunings,
            x_data,
            y_data,
            &cstr_tol,
            &sampling,
            None,
        );
        x_dat
    }
}

impl<SB> EgorSolver<SB>
where
    SB: SurrogateBuilder,
{
    pub fn have_to_recluster(&self, added: usize, prev_added: usize) -> bool {
        self.config.n_clusters == 0 && (added != 0 && added % 10 == 0 && added - prev_added > 0)
    }

    /// Build surrogate given training data and surrogate builder
    /// Reclustering is triggered when recluster boolean is true otherwise
    /// previous clu=stering is used. theta_init allows to reuse
    /// previous theta without fully retraining the surrogates
    /// (faster execution at the cost of surrogate quality)
    #[allow(clippy::too_many_arguments)]
    fn make_clustered_surrogate(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        init: bool,
        iter: u64,
        recluster: bool,
        clustering: Option<&Clustering>,
        theta_inits: Option<&Array2<f64>>,
        model_name: &str,
    ) -> Box<dyn MixtureGpSurrogate> {
        let mut builder = self.surrogate_builder.clone();
        builder.set_kpls_dim(self.config.kpls_dim);
        builder.set_regression_spec(self.config.regression_spec);
        builder.set_correlation_spec(self.config.correlation_spec);
        builder.set_n_clusters(self.config.n_clusters);

        if init || recluster {
            if recluster {
                info!("{} reclustering and training...", model_name);
            } else {
                info!("{} initial clustering and training...", model_name);
            }
            let model = builder
                .train(&xt.view(), &yt.view())
                .expect("GP training failure");
            info!(
                "... {} trained ({} / {})",
                model_name,
                model.n_clusters(),
                model.recombination()
            );
            model
        } else {
            let clustering = clustering.unwrap();

            let theta_tunings = if iter % (self.config.n_optmod as u64) == 0 {
                // set hyperparameters optimization
                let inits = theta_inits
                    .unwrap()
                    .outer_iter()
                    .map(|init| ThetaTuning::Optimized {
                        init: init.to_vec(),
                        bounds: ThetaTuning::default().bounds().unwrap().to_vec(),
                    })
                    .collect::<Vec<_>>();
                if model_name == "Objective" {
                    info!("Objective model hyperparameters optim init >>> {inits:?}");
                }
                inits
            } else {
                // just use previous hyperparameters
                let inits = theta_inits
                    .unwrap()
                    .outer_iter()
                    .map(|init| ThetaTuning::Fixed(init.to_vec()))
                    .collect::<Vec<_>>();
                if model_name == "Objective" {
                    info!("Objective model hyperparameters reused >>> {inits:?}");
                }
                inits
            };
            builder.set_theta_tunings(&theta_tunings);

            let model = builder
                .train_on_clusters(&xt.view(), &yt.view(), clustering)
                .expect("GP training failure");
            model
        }
    }

    /// Returns next promising x points together with virtual (predicted) y values
    /// from surrogate models (taking into account qei strategy if q_parallel)
    /// infill criterion value is also returned
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn next_points(
        &self,
        init: bool,
        iter: u64,
        recluster: bool,
        clusterings: &mut [Option<Clustering>],
        theta_inits: &mut [Option<Array2<f64>>],
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        cstr_tol: &Array1<f64>,
        sampling: &Lhs<f64, Xoshiro256Plus>,
        lhs_optim: Option<u64>,
    ) -> (
        Array2<f64>,
        Array2<f64>,
        f64,
        Vec<std::boxed::Box<dyn egobox_moe::MixtureGpSurrogate>>,
    ) {
        debug!("Make surrogate with {}", x_data);
        let mut x_dat = Array2::zeros((0, x_data.ncols()));
        let mut y_dat = Array2::zeros((0, y_data.ncols()));
        let mut infill_val = f64::INFINITY;
        let mut models: Vec<std::boxed::Box<dyn egobox_moe::MixtureGpSurrogate>> = vec![];
        for i in 0..self.config.q_points {
            let (xt, yt) = if i == 0 {
                (x_data.to_owned(), y_data.to_owned())
            } else {
                (
                    concatenate![Axis(0), x_data.to_owned(), x_dat.to_owned()],
                    concatenate![Axis(0), y_data.to_owned(), y_dat.to_owned()],
                )
            };

            info!("Train surrogates with {} points...", xt.nrows());
            models = (0..=self.config.n_cstr)
                .into_par_iter()
                .map(|k| {
                    let name = if k == 0 {
                        "Objective".to_string()
                    } else {
                        format!("Constraint[{k}]")
                    };
                    self.make_clustered_surrogate(
                        &xt,
                        &yt.slice(s![.., k..k + 1]).to_owned(),
                        init && i == 0,
                        iter,
                        recluster,
                        clusterings[k].as_ref(),
                        theta_inits[k].as_ref(),
                        &name,
                    )
                })
                .collect();
            (0..=self.config.n_cstr).for_each(|k| {
                clusterings[k] = Some(models[k].to_clustering());
                let mut thetas_k = Array2::zeros((
                    models[k].experts().len(),
                    models[k].experts()[0].theta().len(),
                ));
                for (i, expert) in models[k].experts().iter().enumerate() {
                    thetas_k.row_mut(i).assign(expert.theta());
                }
                theta_inits[k] = Some(thetas_k);
            });

            let (obj_model, cstr_models) = models.split_first().unwrap();
            debug!("... surrogates trained");

            match self.find_best_point(
                x_data,
                y_data,
                sampling,
                obj_model.as_ref(),
                cstr_models,
                cstr_tol,
                lhs_optim,
            ) {
                Ok((infill_obj, xk)) => {
                    match self.get_virtual_point(&xk, y_data, obj_model.as_ref(), cstr_models) {
                        Ok(yk) => {
                            y_dat = concatenate![
                                Axis(0),
                                y_dat,
                                Array2::from_shape_vec((1, 1 + self.config.n_cstr), yk).unwrap()
                            ];
                            x_dat = concatenate![Axis(0), x_dat, xk.insert_axis(Axis(0))];
                            // infill objective was minimized while infill criterion itself
                            // is expected to be maximized hence the negative sign here
                            infill_val = -infill_obj;
                        }
                        Err(err) => {
                            // Error while predict at best point: ignore
                            info!("Error while getting virtual point: {}", err);
                            break;
                        }
                    }
                }
                Err(err) => {
                    // Cannot find best point: ignore
                    debug!("Find best point error: {}", err);
                    break;
                }
            }
        }
        (x_dat, y_dat, infill_val, models)
    }

    pub(crate) fn compute_scaling(
        &self,
        sampling: &Lhs<f64, Xoshiro256Plus>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        f_min: f64,
    ) -> (f64, Array1<f64>, f64) {
        let npts = (100 * self.xlimits.nrows()).min(1000);
        debug!("Use {npts} points to evaluate scalings");
        let scaling_points = sampling.sample(npts);
        let scale_infill_obj =
            self.compute_infill_obj_scale(&scaling_points.view(), obj_model, f_min);
        info!(
            "Infill criterion scaling is updated to {}",
            scale_infill_obj
        );
        let scale_cstr = if cstr_models.is_empty() {
            Array1::zeros((0,))
        } else {
            let scale_cstr = compute_cstr_scales(&scaling_points.view(), cstr_models);
            info!("Constraints scalings is updated to {}", scale_cstr);
            scale_cstr
        };
        let scale_ic =
            self.config
                .infill_criterion
                .scaling(&scaling_points.view(), obj_model, f_min);
        (scale_infill_obj, scale_cstr, scale_ic)
    }

    /// Find best x promising points by optimizing the chosen infill criterion
    /// The optimized value of the criterion is returned together with the
    /// optimum location
    #[allow(clippy::too_many_arguments)]
    fn find_best_point(
        &self,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        sampling: &Lhs<f64, Xoshiro256Plus>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        cstr_tol: &Array1<f64>,
        lhs_optim_seed: Option<u64>,
    ) -> Result<(f64, Array1<f64>)> {
        let f_min = y_data.min().unwrap();

        let mut success = false;
        let mut n_optim = 1;
        let n_max_optim = 3;
        let mut best_x = None;

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
        info!("Optimize infill criterion...");
        let obj_data = InfillObjData {
            scale_infill_obj,
            scale_cstr: scale_cstr.to_owned(),
            scale_wb2,
        };
        while !success && n_optim <= n_max_optim {
            let x_start = sampling.sample(self.config.n_start);

            if let Some(seed) = lhs_optim_seed {
                let (y_opt, x_opt) =
                    Optimizer::new(Algorithm::Lhs, &obj, &cstr_refs, &obj_data, &self.xlimits)
                        .cstr_tol(cstr_tol.to_owned())
                        .seed(seed)
                        .minimize();

                info!("LHS optimization best_x {}", x_opt);
                best_x = Some((y_opt, x_opt));
                success = true;
            } else {
                let dim = x_data.ncols();
                let res = (0..self.config.n_start)
                    .into_par_iter()
                    .map(|i| {
                        Optimizer::new(algorithm, &obj, &cstr_refs, &obj_data, &self.xlimits)
                            .xinit(&x_start.row(i))
                            .max_eval(200)
                            .ftol_rel(1e-4)
                            .ftol_abs(1e-4)
                            .minimize()
                    })
                    .reduce(
                        || (f64::INFINITY, Array::ones((dim,))),
                        |a, b| if b.0 < a.0 { b } else { a },
                    );

                if res.0.is_nan() || res.0.is_infinite() {
                    success = false;
                } else {
                    best_x = Some((res.0, Array::from(res.1.clone())));
                    success = true;
                }
            }

            if n_optim == n_max_optim && best_x.is_none() {
                info!("All optimizations fail => Trigger LHS optimization");
                let (y_opt, x_opt) =
                    Optimizer::new(Algorithm::Lhs, &obj, &cstr_refs, &obj_data, &self.xlimits)
                        .minimize();

                info!("LHS optimization best_x {}", x_opt);
                best_x = Some((y_opt, x_opt));
                success = true;
            }
            n_optim += 1;
        }
        if best_x.is_some() {
            debug!("... infill criterion optimum found");
        }
        best_x.ok_or_else(|| EgoError::EgoError(String::from("Can not find best point")))
    }

    /// Return the virtual points regarding the qei strategy
    /// The default is to return GP prediction
    fn get_virtual_point(
        &self,
        xk: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
    ) -> Result<Vec<f64>> {
        let mut res: Vec<f64> = Vec::with_capacity(3);
        if self.config.q_ei == QEiStrategy::ConstantLiarMinimum {
            let index_min = y_data.slice(s![.., 0_usize]).argmin().unwrap();
            res.push(y_data[[index_min, 0]]);
            for ic in 1..=self.config.n_cstr {
                res.push(y_data[[index_min, ic]]);
            }
            Ok(res)
        } else {
            let x = &xk.view().insert_axis(Axis(0));
            let pred = obj_model.predict(x)?[[0, 0]];
            let var = obj_model.predict_var(x)?[[0, 0]];
            let conf = match self.config.q_ei {
                QEiStrategy::KrigingBeliever => 0.,
                QEiStrategy::KrigingBelieverLowerBound => -3.,
                QEiStrategy::KrigingBelieverUpperBound => 3.,
                _ => -1., // never used
            };
            res.push(pred + conf * f64::sqrt(var));
            for cstr_model in cstr_models {
                res.push(cstr_model.predict(x)?[[0, 0]]);
            }
            Ok(res)
        }
    }

    /// The infill criterion scaling is computed using x (n points of nx dim)
    /// given the objective function surrogate
    fn compute_infill_obj_scale(
        &self,
        x: &ArrayView2<f64>,
        obj_model: &dyn MixtureGpSurrogate,
        f_min: f64,
    ) -> f64 {
        let mut crit_vals = Array1::zeros(x.nrows());
        let (mut nan_count, mut inf_count) = (0, 0);
        Zip::from(&mut crit_vals).and(x.rows()).for_each(|c, x| {
            let val = self.eval_infill_obj(&x.to_vec(), obj_model, f_min, 1.0, 1.0);
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
        f_min: f64,
        scale: f64,
        scale_ic: f64,
    ) -> f64 {
        let x_f = x.to_vec();
        let obj = -(self
            .config
            .infill_criterion
            .value(&x_f, obj_model, f_min, Some(scale_ic)));
        obj / scale
    }

    pub fn eval_grad_infill_obj(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        f_min: f64,
        scale: f64,
        scale_ic: f64,
    ) -> Vec<f64> {
        let x_f = x.to_vec();
        let grad = -(self
            .config
            .infill_criterion
            .grad(&x_f, obj_model, f_min, Some(scale_ic)));
        (grad / scale).to_vec()
    }

    pub fn eval_obj<O: CostFunction<Param = Array2<f64>, Output = Array2<f64>>>(
        &self,
        pb: &mut Problem<O>,
        x: &Array2<f64>,
    ) -> Array2<f64> {
        let params = if self.config.discrete() {
            // We have to cast x to folded space as EgorSolver
            // works internally in the continuous space while
            // the objective function expects discrete variable in folded space
            to_discrete_space(&self.config.xtypes, x)
        } else {
            x.to_owned()
        };
        pb.problem("cost_count", |problem| problem.cost(&params))
            .expect("Objective evaluation")
    }
}
