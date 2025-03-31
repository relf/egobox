use std::marker::PhantomData;

use crate::errors::{EgoError, Result};
use crate::gpmix::mixint::{as_continuous_limits, to_discrete_space};
use crate::utils::{find_best_result_index_from, update_data};
use crate::{find_best_result_index, EgorConfig};
use crate::{types::*, EgorState};
use crate::{EgorSolver, DEFAULT_CSTR_TOL, MAX_POINT_ADDITION_RETRY};

#[cfg(not(feature = "nlopt"))]
use crate::types::ObjFn;
#[cfg(feature = "nlopt")]
use nlopt::ObjFn;

use argmin::argmin_error_closure;
use argmin::core::{CostFunction, Problem, State};

use egobox_doe::{Lhs, LhsKind};
use egobox_gp::ThetaTuning;
use env_logger::{Builder, Env};

use egobox_moe::{Clustering, MixtureGpSurrogate, NbClusters};
use log::{debug, info, warn};
use ndarray::{concatenate, s, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip};

use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use serde::de::DeserializeOwned;

impl<SB: SurrogateBuilder + DeserializeOwned, C: CstrFn> EgorSolver<SB, C> {
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
            phantom: PhantomData,
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

        // TODO: Manage fonction constraints
        let fcstrs = Vec::<Cstr>::new();
        // TODO: c_data has to be passed as argument or better computed using fcstrs(x_data)
        let c_data = Array2::zeros((x_data.nrows(), 0));
        // TODO: Coego not implemented
        let activity = None;

        let (x_dat, _, _, _, _) = self.select_next_points(
            true,
            0,
            false, // done anyway
            &mut clusterings,
            &mut theta_tunings,
            activity,
            x_data,
            y_data,
            &c_data,
            &cstr_tol,
            &sampling,
            None,
            find_best_result_index(y_data, &c_data, &cstr_tol),
            &fcstrs,
        );
        x_dat
    }
}

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + DeserializeOwned,
    C: CstrFn,
{
    pub fn have_to_recluster(&self, added: usize, prev_added: usize) -> bool {
        self.config.n_clusters.is_auto()
            && (added != 0 && added % 10 == 0 && added - prev_added > 0)
    }

    /// Build surrogate given training data and surrogate builder
    /// Reclustering is triggered when recluster boolean is true otherwise
    /// previous clu=stering is used. theta_init allows to reuse
    /// previous theta without fully retraining the surrogates
    /// (faster execution at the cost of surrogate quality)
    #[allow(clippy::too_many_arguments)]
    fn make_clustered_surrogate(
        &self,
        model_name: &str,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        make_clustering: bool,
        optimize_theta: bool,
        clustering: Option<&Clustering>,
        theta_inits: Option<&Array2<f64>>,
        actives: &Array2<usize>,
    ) -> Box<dyn MixtureGpSurrogate> {
        let mut builder = self.surrogate_builder.clone();
        builder.set_kpls_dim(self.config.kpls_dim);
        builder.set_regression_spec(self.config.regression_spec);
        builder.set_correlation_spec(self.config.correlation_spec);
        builder.set_n_clusters(self.config.n_clusters.clone());

        let mut model = None;
        let mut best_likelihood = -f64::INFINITY;
        let mut best_theta_inits = theta_inits.map(|inits| inits.to_owned());
        for (i, active) in actives.outer_iter().enumerate() {
            let gp = if make_clustering
            /* init || recluster */
            {
                if self.config.coego.activated {
                    match self.config.n_clusters {
                        NbClusters::Auto { max: _ } => log::warn!(
                            "Automated clustering not available with CoEGO: Use CoEGO-FT"
                        ),
                        NbClusters::Fixed { nb } => {
                            let default_init = Array2::from_elem(
                                (nb, xt.ncols()),
                                ThetaTuning::<f64>::DEFAULT_INIT,
                            );
                            let theta_tunings = best_theta_inits
                                .clone()
                                .unwrap_or(default_init)
                                .outer_iter()
                                .map(|init| ThetaTuning::Partial {
                                    init: init.to_owned(),
                                    bounds: Array1::from_vec(vec![
                                        ThetaTuning::<f64>::DEFAULT_BOUNDS;
                                        init.len()
                                    ]),
                                    active: active.to_vec(),
                                })
                                .collect::<Vec<_>>();
                            builder.set_theta_tunings(&theta_tunings);
                        }
                    }
                }
                if i == 0 {
                    info!("{} clustering and training...", model_name);
                }
                let gp = builder
                    .train(xt.view(), yt.view())
                    .expect("GP training failure");

                if i == 0 {
                    info!(
                        "... {} trained ({} / {})",
                        model_name,
                        gp.n_clusters(),
                        gp.recombination()
                    );
                }
                gp
            } else {
                let clustering = clustering.unwrap();

                let mut theta_tunings = if optimize_theta {
                    // set hyperparameters optimization
                    let inits = best_theta_inits
                        .clone()
                        .expect("Theta initialization is provided")
                        .outer_iter()
                        .map(|init| ThetaTuning::Full {
                            init: init.to_owned(),
                            bounds: ThetaTuning::default().bounds().unwrap().to_owned(),
                        })
                        .collect::<Vec<_>>();
                    if i == 0 && model_name == "Objective" {
                        info!("Objective model hyperparameters optim init >>> {inits:?}");
                    }
                    inits
                } else {
                    // just use previous hyperparameters
                    let inits = best_theta_inits
                        .clone()
                        .unwrap()
                        .outer_iter()
                        .map(|init| ThetaTuning::Fixed(init.to_owned()))
                        .collect::<Vec<_>>();
                    if i == 0 && model_name == "Objective" {
                        info!("Objective model hyperparameters reused >>> {inits:?}");
                    }
                    inits
                };

                if self.config.coego.activated {
                    self.set_partial_theta_tuning(&active.to_vec(), &mut theta_tunings);
                }
                builder.set_theta_tunings(&theta_tunings);

                let gp = builder
                    .train_on_clusters(xt.view(), yt.view(), clustering)
                    .expect("GP training failure");
                gp
            };

            // CoEGO only in mono cluster, update theta if better likelihood
            if self.config.coego.activated {
                if self.config.n_clusters.is_mono() {
                    let likelihood = gp.experts()[0].likelihood();
                    // We update only if better likelihood
                    if likelihood > best_likelihood {
                        log::info!("Likelihood = {}", likelihood);
                        best_likelihood = likelihood;
                        best_theta_inits =
                            Some(gp.experts()[0].theta().clone().insert_axis(Axis(0)));
                    }
                } else {
                    log::warn!(
                            "CoEGO theta update wrt likelihood not implemented in multi-cluster setting");
                }
            };
            model = Some(gp)
        }
        model.expect("Surrogate model is trained")
    }

    /// Refresh infill data used to optimize infill criterion
    pub fn refresh_infill_data<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>> + DomainConstraints<C>,
    >(
        &self,
        problem: &mut Problem<O>,
        state: &EgorState<f64>,
        models: &[Box<dyn egobox_moe::MixtureGpSurrogate>],
    ) -> InfillObjData<f64> {
        let y_data = state.data.as_ref().unwrap().1.clone();
        let x_data = state.data.as_ref().unwrap().0.clone();
        let (obj_model, cstr_models) = models.split_first().unwrap();
        let sampling = state.sampling.clone().unwrap();

        let fmin = y_data[[state.best_index.unwrap(), 0]];
        let xbest = x_data.row(state.best_index.unwrap()).to_vec();

        let pb = problem.take_problem().unwrap();
        let fcstrs = pb.fn_constraints();
        let (scale_infill_obj, scale_cstr, _scale_fcstr, scale_wb2) =
            self.compute_scaling(&sampling, obj_model.as_ref(), cstr_models, fcstrs, fmin);
        problem.problem = Some(pb);

        InfillObjData {
            fmin,
            xbest,
            scale_infill_obj,
            scale_cstr: Some(scale_cstr.to_owned()),
            scale_wb2,
        }
    }

    /// Regenerate surrogate models from current state
    /// This method supposes that clustering is done and thetas has to be optimized
    pub fn refresh_surrogates(
        &self,
        state: &EgorState<f64>,
    ) -> Vec<Box<dyn egobox_moe::MixtureGpSurrogate>> {
        info!(
            "Train surrogates with {} points...",
            &state.data.as_ref().unwrap().0.nrows()
        );

        let actives = state
            .activity
            .as_ref()
            .unwrap_or(&self.full_activity())
            .to_owned();

        (0..=self.config.n_cstr)
            .into_par_iter()
            .map(|k| {
                let name = if k == 0 {
                    "Objective".to_string()
                } else {
                    format!("Constraint[{k}]")
                };
                self.make_clustered_surrogate(
                    &name,
                    &state.data.as_ref().unwrap().0,
                    &state.data.as_ref().unwrap().1.slice(s![.., k]).to_owned(),
                    false,
                    true,
                    state.clusterings.as_ref().unwrap()[k].as_ref(),
                    state.theta_inits.as_ref().unwrap()[k].as_ref(),
                    &actives,
                )
            })
            .collect()
    }

    /// This function is the main EGO algorithm iteration:
    /// * Train surrogates
    /// * Find next promising location(s) of optimum
    /// * Update state: Evaluate true function, update doe and optimum
    #[allow(clippy::type_complexity)]
    pub fn ego_step<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>> + DomainConstraints<C>,
    >(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> Result<(EgorState<f64>, InfillObjData<f64>, usize)> {
        let mut new_state = state.clone();
        let mut clusterings = new_state
            .take_clusterings()
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "EgorSolver: No clustering!"
            ))?;
        let mut theta_inits = new_state
            .take_theta_inits()
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "EgorSolver: No theta inits!"
            ))?;
        let activity = new_state.take_activity();
        let sampling = new_state.take_sampling().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "EgorSolver: No sampling!"
        ))?;
        let (mut x_data, mut y_data, mut c_data) = new_state
            .take_data()
            .ok_or_else(argmin_error_closure!(PotentialBug, "EgorSolver: No data!"))?;

        let (rejected_count, infill_data) = loop {
            let recluster = self.have_to_recluster(new_state.added, new_state.prev_added);
            let init = new_state.get_iter() == 0;
            let lhs_optim_seed = if new_state.no_point_added_retries == 0 {
                debug!("Try Lhs optimization!");
                Some(new_state.added as u64)
            } else {
                debug!(
                    "Try point addition {}/{}",
                    MAX_POINT_ADDITION_RETRY - new_state.no_point_added_retries,
                    MAX_POINT_ADDITION_RETRY
                );
                None
            };

            let pb = problem.take_problem().unwrap();
            let fcstrs = pb.fn_constraints();
            let (x_dat, y_dat, c_dat, infill_value, infill_data) = self.select_next_points(
                init,
                state.get_iter(),
                recluster,
                &mut clusterings,
                &mut theta_inits,
                activity.as_ref(),
                &x_data,
                &y_data,
                &c_data,
                &state.cstr_tol,
                &sampling,
                lhs_optim_seed,
                state.best_index.unwrap(),
                fcstrs,
            );
            problem.problem = Some(pb);

            debug!("Try adding {}", x_dat);
            let added_indices = update_data(
                &mut x_data,
                &mut y_data,
                &mut c_data,
                &x_dat,
                &y_dat,
                &c_dat,
            );

            new_state = new_state
                .clusterings(clusterings.clone())
                .theta_inits(theta_inits.clone())
                .data((x_data.clone(), y_data.clone(), c_data.clone()))
                .infill_value(infill_value)
                .sampling(sampling.clone())
                .param(x_dat.row(0).to_owned())
                .cost(y_dat.row(0).to_owned());
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

            let rejected_count = x_dat.nrows() - added_indices.len();
            for i in 0..x_dat.nrows() {
                debug!(
                    "  {} {}",
                    if added_indices.contains(&i) { "A" } else { "R" },
                    x_dat.row(i)
                );
            }
            if rejected_count > 0 {
                info!(
                    "Reject {}/{} point{} too close to previous ones",
                    rejected_count,
                    x_dat.nrows(),
                    if rejected_count > 1 { "s" } else { "" }
                );
            }
            if rejected_count == x_dat.nrows() {
                new_state.no_point_added_retries -= 1;
                if new_state.no_point_added_retries == 0 {
                    info!("Max number of retries ({}) without adding point", 3);
                    info!("Use LHS optimization to ensure a point addition");
                }
                if new_state.no_point_added_retries < 0 {
                    // no luck with LHS optimization
                    warn!("Fail to add another point to improve the surrogate models. Abort!");
                    return Err(EgoError::GlobalStepNoPointError);
                }
            } else {
                // ok point added we can go on, just output number of rejected point
                break (rejected_count, infill_data);
            }
        };
        let add_count = (self.config.q_points - rejected_count) as i32;
        let x_to_eval = x_data.slice(s![-add_count.., ..]).to_owned();
        debug!(
            "Eval {} point{} {}",
            add_count,
            if add_count > 1 { "s" } else { "" },
            if new_state.no_point_added_retries == 0 {
                " from sampling"
            } else {
                ""
            }
        );

        new_state.prev_added = new_state.added;
        new_state.added += add_count as usize;
        info!("+{} point(s), total: {} points", add_count, new_state.added);
        new_state.no_point_added_retries = MAX_POINT_ADDITION_RETRY;
        let y_actual = self.eval_obj(problem, &x_to_eval);
        Zip::from(y_data.slice_mut(s![-add_count.., ..]).rows_mut())
            .and(y_actual.rows())
            .for_each(|mut y, val| y.assign(&val));

        let best_index = find_best_result_index_from(
            state.best_index.unwrap(),
            y_data.nrows() - add_count as usize,
            &y_data,
            &c_data,
            &new_state.cstr_tol,
        );
        new_state.prev_best_index = state.best_index;
        new_state.best_index = Some(best_index);
        new_state = new_state.data((x_data.clone(), y_data.clone(), c_data.clone()));

        Ok((new_state, infill_data, best_index))
    }

    /// Returns next promising x points together with virtual (predicted) y values
    /// from surrogate models (taking into account qei strategy if q_parallel)
    /// infill criterion value is also returned
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn select_next_points(
        &self,
        init: bool,
        iter: u64,
        recluster: bool,
        clusterings: &mut [Option<Clustering>],
        theta_inits: &mut [Option<Array2<f64>>],
        activity: Option<&Array2<usize>>,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        c_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        cstr_tol: &Array1<f64>,
        sampling: &Lhs<f64, Xoshiro256Plus>,
        lhs_optim: Option<u64>,
        best_index: usize,
        cstr_funcs: &[impl CstrFn],
    ) -> (
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        f64,
        InfillObjData<f64>,
    ) {
        debug!("Make surrogate with {}", x_data);
        let mut x_dat = Array2::zeros((0, x_data.ncols()));
        let mut y_dat = Array2::zeros((0, y_data.ncols()));
        let mut c_dat = Array2::zeros((0, c_data.ncols()));
        let mut infill_val = f64::INFINITY;
        let mut infill_data = Default::default();
        for i in 0..self.config.q_points {
            let (xt, yt) = if i == 0 {
                (x_data.to_owned(), y_data.to_owned())
            } else {
                (
                    concatenate![Axis(0), x_data.to_owned(), x_dat.to_owned()],
                    concatenate![Axis(0), y_data.to_owned(), y_dat.to_owned()],
                )
            };

            log::debug!("activity: {:?}", activity);
            let actives = activity.unwrap_or(&self.full_activity()).to_owned();

            info!("Train surrogates with {} points...", xt.nrows());
            let models: Vec<Box<dyn egobox_moe::MixtureGpSurrogate>> = (0..=self.config.n_cstr)
                .into_par_iter()
                .map(|k| {
                    let name = if k == 0 {
                        "Objective".to_string()
                    } else {
                        format!("Constraint[{k}]")
                    };
                    let make_clustering = (init && i == 0) || recluster;
                    let optimize_theta =
                        (iter as usize * self.config.q_points + i) % (self.config.n_optmod) == 0;
                    self.make_clustered_surrogate(
                        &name,
                        &xt,
                        &yt.slice(s![.., k]).to_owned(),
                        make_clustering,
                        optimize_theta,
                        clusterings[k].as_ref(),
                        theta_inits[k].as_ref(),
                        &actives,
                    )
                })
                .collect();

            // Update theta initialization from optimized theta
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

            let fmin = y_data[[best_index, 0]];
            let xbest = x_data.row(best_index).to_owned();
            let (scale_infill_obj, scale_cstr, scale_fcstr, scale_wb2) =
                self.compute_scaling(sampling, obj_model.as_ref(), cstr_models, cstr_funcs, fmin);

            let all_scale_cstr = concatenate![Axis(0), scale_cstr, scale_fcstr];

            infill_data = InfillObjData {
                fmin,
                xbest: xbest.to_vec(),
                scale_infill_obj,
                scale_cstr: Some(all_scale_cstr.to_owned()),
                scale_wb2,
            };

            let cstr_funcs = cstr_funcs
                .iter()
                .enumerate()
                .map(|(i, cstr)| {
                    let scale_fc = scale_fcstr[i];
                    move |x: &[f64],
                          gradient: Option<&mut [f64]>,
                          params: &mut InfillObjData<f64>|
                          -> f64 {
                        let x = if self.config.discrete() {
                            let xary = Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
                            // We have to cast x to folded space as EgorSolver
                            // works internally in the continuous space while
                            // the constraint function expects discrete variable in folded space
                            to_discrete_space(&self.config.xtypes, &xary)
                                .row(0)
                                .into_owned();
                            &xary.into_iter().collect::<Vec<_>>()
                        } else {
                            x
                        };
                        cstr(x, gradient, params) / scale_fc
                    }
                })
                .collect::<Vec<_>>();
            let cstr_funcs = cstr_funcs
                .iter()
                .map(|cstr| cstr as &(dyn ObjFn<InfillObjData<f64>> + Sync))
                .collect::<Vec<_>>();

            let (infill_obj, xk) = self.compute_best_point(
                sampling,
                obj_model.as_ref(),
                cstr_models,
                cstr_tol,
                lhs_optim,
                &infill_data,
                &cstr_funcs,
                (fmin, xbest),
                &actives,
            );

            match self.compute_virtual_point(&xk, y_data, obj_model.as_ref(), cstr_models) {
                Ok(yk) => {
                    let yk = Array2::from_shape_vec((1, 1 + self.config.n_cstr), yk).unwrap();
                    y_dat = concatenate![Axis(0), y_dat, yk];

                    let ck = cstr_funcs
                        .iter()
                        .map(|cstr| cstr(&xk.to_vec(), None, &mut infill_data))
                        .collect::<Vec<_>>();
                    c_dat = concatenate![
                        Axis(0),
                        c_dat,
                        Array2::from_shape_vec((1, cstr_funcs.len()), ck).unwrap()
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
        (x_dat, y_dat, c_dat, infill_val, infill_data)
    }
}
