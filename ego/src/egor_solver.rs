//! Egor implementation as a [argmin::core::Solver] to be used to benefit from
//! features coming with the argmin framework such as checkpointing or observers.
//!
//! Note: Depending on your need you can either use the `EgorSolver` or the provided
//! `EgorBuilder` which allows to build an `Egor` struct which wraps the `argmin::Executor`
//! running an `EgorSolver` on `ObjFun`. See [`crate::EgorBuilder`]
//!
//! ```no_run
//! use ndarray::{array, Array2, ArrayView1, ArrayView2, Zip};
//! use egobox_doe::{Lhs, SamplingMethod};
//! use egobox_ego::{EgorBuilder, EgorConfig, InfillStrategy, InfillOptimizer, ObjFunc, EgorSolver, to_xtypes};
//! use egobox_moe::GpMixtureParams;
//! use rand_xoshiro::Xoshiro256Plus;
//! use ndarray_rand::rand::SeedableRng;
//! use argmin::core::Executor;
//!
//! use argmin_testfunctions::rosenbrock;
//!
//! // Rosenbrock test function: minimum y_opt = 0 at x_opt = (1, 1)
//! fn rosenb(x: &ArrayView2<f64>) -> Array2<f64> {
//!     let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
//!     Zip::from(y.rows_mut())
//!         .and(x.rows())
//!         .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec())]));
//!     y
//! }
//! let rng = Xoshiro256Plus::seed_from_u64(42);
//! let xtypes = to_xtypes(&array![[-2., 2.], [-2., 2.]]);
//! let fobj = ObjFunc::new(rosenb);
//! let config = EgorConfig::default().xtypes(&xtypes);
//! let solver: EgorSolver<GpMixtureParams<f64, Xoshiro256Plus>> = EgorSolver::new(config, rng);
//! let res = Executor::new(fobj, solver)
//!             .configure(|state| state.max_iters(20))
//!             .run()
//!             .unwrap();
//! println!("Rosenbrock min result = {:?}", res.state);
//! ```
//!
//! Constraints are expected to be evaluated with the objective function
//! meaning that the function passed to the optimizer has to return
//! a vector consisting of [obj, cstr_1, ..., cstr_n] and the cstr values
//! are intended to be negative at the end of the optimization.
//! Constraint number should be declared with `n_cstr` setter.
//! A tolerance can be adjust with `cstr_tol` setter for relaxing constraint violation
//! if specified cstr values should be < `cstr_tol` (instead of < 0)
//!
//! ```no_run
//! use ndarray::{array, Array2, ArrayView1, ArrayView2, Zip};
//! use egobox_doe::{Lhs, SamplingMethod};
//! use egobox_ego::{EgorBuilder, EgorConfig, InfillStrategy, InfillOptimizer, ObjFunc, EgorSolver, to_xtypes};
//! use egobox_moe::GpMixtureParams;
//! use rand_xoshiro::Xoshiro256Plus;
//! use ndarray_rand::rand::SeedableRng;
//! use argmin::core::Executor;
//!
//! // Function G24: 1 global optimum y_opt = -5.5080 at x_opt =(2.3295, 3.1785)
//! fn g24(x: &ArrayView1<f64>) -> f64 {
//!    -x[0] - x[1]
//! }
//!
//! // Constraints < 0
//! fn g24_c1(x: &ArrayView1<f64>) -> f64 {
//!     -2.0 * x[0].powf(4.0) + 8.0 * x[0].powf(3.0) - 8.0 * x[0].powf(2.0) + x[1] - 2.0
//! }
//!
//! fn g24_c2(x: &ArrayView1<f64>) -> f64 {
//!     -4.0 * x[0].powf(4.0) + 32.0 * x[0].powf(3.0)
//!     - 88.0 * x[0].powf(2.0) + 96.0 * x[0] + x[1]
//!     - 36.0
//! }
//!
//! // Gouped function : objective + constraints
//! fn f_g24(x: &ArrayView2<f64>) -> Array2<f64> {
//!     let mut y = Array2::zeros((x.nrows(), 3));
//!     Zip::from(y.rows_mut())
//!         .and(x.rows())
//!         .for_each(|mut yi, xi| {
//!             yi.assign(&array![g24(&xi), g24_c1(&xi), g24_c2(&xi)]);
//!         });
//!     y
//! }
//!
//! let rng = Xoshiro256Plus::seed_from_u64(42);
//! let xlimits = array![[0., 3.], [0., 4.]];
//! let doe = Lhs::new(&xlimits).sample(10);
//! let xtypes = to_xtypes(&xlimits);
//!
//! let fobj = ObjFunc::new(f_g24);
//!
//! let config = EgorConfig::default()
//!     .xtypes(&xtypes)
//!     .n_cstr(2)
//!     .infill_strategy(InfillStrategy::EI)
//!     .infill_optimizer(InfillOptimizer::Cobyla)
//!     .doe(&doe)
//!     .target(-5.5080);
//!
//! let solver: EgorSolver<GpMixtureParams<f64, Xoshiro256Plus>> =
//!   EgorSolver::new(config, rng);
//!
//! let res = Executor::new(fobj, solver)
//!             .configure(|state| state.max_iters(40))
//!             .run()
//!             .expect("g24 minimized");
//! println!("G24 min result = {:?}", res.state);
//! ```
//!
use crate::egor_config::EgorConfig;
use crate::egor_state::{find_best_result_index, EgorState, MAX_POINT_ADDITION_RETRY};
use crate::errors::{EgoError, Result};

use crate::mixint::*;

use crate::optimizer::*;

use crate::types::*;
use crate::utils::{compute_cstr_scales, update_data};

use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use egobox_gp::ThetaTuning;
use egobox_moe::{
    Clustering, CorrelationSpec, GpMixtureParams, MixtureGpSurrogate, RegressionSpec,
};
use env_logger::{Builder, Env};
use finitediff::FiniteDiff;
use linfa::ParamGuard;
use log::{debug, info, warn};
use ndarray::{
    concatenate, s, Array, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2, Zip,
};
use ndarray_npy::{read_npy, write_npy};
use ndarray_stats::QuantileExt;

use rand_xoshiro::Xoshiro256Plus;

use argmin::argmin_error_closure;
use argmin::core::{
    CostFunction, Problem, Solver, State, TerminationReason, TerminationStatus, KV,
};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use std::time::Instant;

/// Numpy filename for initial DOE dump
pub const DOE_INITIAL_FILE: &str = "egor_initial_doe.npy";
/// Numpy Filename for current DOE dump
pub const DOE_FILE: &str = "egor_doe.npy";

/// Default tolerance value for constraints to be satisfied (ie cstr < tol)
pub const DEFAULT_CSTR_TOL: f64 = 1e-6;

/// Implementation of `argmin::core::Solver` for Egor optimizer.
/// Therefore this structure can be used with `argmin::core::Executor` and benefit
/// from observers and checkpointing features.
#[derive(Clone, Serialize, Deserialize)]
pub struct EgorSolver<SB: SurrogateBuilder> {
    pub(crate) config: EgorConfig,
    /// Matrix (nx, 2) of [lower bound, upper bound] of the nx components of x
    /// Note: used for continuous variables handling, the optimizer base.
    pub(crate) xlimits: Array2<f64>,
    /// An optional surrogate builder used to model objective and constraint
    /// functions, otherwise [mixture of expert](egobox_moe) is used
    /// Note: if specified takes precedence over individual settings
    pub(crate) surrogate_builder: SB,
    /// A random generator used to get reproductible results.
    /// For instance: Xoshiro256Plus::from_u64_seed(42) for reproducibility
    pub(crate) rng: Xoshiro256Plus,
}

impl SurrogateBuilder for GpMixtureParams<f64, Xoshiro256Plus> {
    /// Constructor from domain space specified with types
    /// **panic** if xtypes contains other types than continuous type `Float`
    fn new_with_xtypes(xtypes: &[XType]) -> Self {
        if crate::utils::discrete(xtypes) {
            panic!("GpMixtureParams cannot be created with discrete types!");
        }
        GpMixtureParams::new()
    }

    /// Sets the allowed regression models used in gaussian processes.
    fn set_regression_spec(&mut self, regression_spec: RegressionSpec) {
        *self = self.clone().regression_spec(regression_spec);
    }

    /// Sets the allowed correlation models used in gaussian processes.
    fn set_correlation_spec(&mut self, correlation_spec: CorrelationSpec) {
        *self = self.clone().correlation_spec(correlation_spec);
    }

    /// Sets the number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
    fn set_kpls_dim(&mut self, kpls_dim: Option<usize>) {
        *self = self.clone().kpls_dim(kpls_dim);
    }

    /// Sets the number of clusters used by the mixture of surrogate experts.
    fn set_n_clusters(&mut self, n_clusters: usize) {
        *self = self.clone().n_clusters(n_clusters);
    }

    /// Sets the number of clusters used by the mixture of surrogate experts.
    fn set_theta_tunings(&mut self, theta_tunings: &[ThetaTuning<f64>]) {
        *self = self.clone().theta_tunings(theta_tunings);
    }

    fn train(
        &self,
        xt: &ArrayView2<f64>,
        yt: &ArrayView2<f64>,
    ) -> Result<Box<dyn MixtureGpSurrogate>> {
        let checked = self.check_ref()?;
        let moe = checked.train(xt, yt)?;
        Ok(moe).map(|moe| Box::new(moe) as Box<dyn MixtureGpSurrogate>)
    }

    fn train_on_clusters(
        &self,
        xt: &ArrayView2<f64>,
        yt: &ArrayView2<f64>,
        clustering: &Clustering,
    ) -> Result<Box<dyn MixtureGpSurrogate>> {
        let checked = self.check_ref()?;
        let moe = checked.train_on_clusters(xt, yt, clustering)?;
        Ok(moe).map(|moe| Box::new(moe) as Box<dyn MixtureGpSurrogate>)
    }
}

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
        let (x_dat, _) = self.next_points(
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

/// Build `xtypes` from simple float bounds of `x` input components when x belongs to R^n.
/// xlimits are bounds of the x components expressed a matrix (dim, 2) where dim is the dimension of x
/// the ith row is the bounds interval [lower, upper] of the ith comonent of `x`.  
pub fn to_xtypes(xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Vec<XType> {
    let mut xtypes: Vec<XType> = vec![];
    Zip::from(xlimits.rows()).for_each(|limits| xtypes.push(XType::Cont(limits[0], limits[1])));
    xtypes
}

impl<O, SB> Solver<O, EgorState<f64>> for EgorSolver<SB>
where
    O: CostFunction<Param = Array2<f64>, Output = Array2<f64>>,
    SB: SurrogateBuilder,
{
    const NAME: &'static str = "Egor";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> std::result::Result<(EgorState<f64>, Option<KV>), argmin::core::Error> {
        let rng = self.rng.clone();
        let sampling = Lhs::new(&self.xlimits).with_rng(rng).kind(LhsKind::Maximin);

        let hstart_doe: Option<Array2<f64>> =
            if self.config.hot_start && self.config.outdir.is_some() {
                let path: &String = self.config.outdir.as_ref().unwrap();
                let filepath = std::path::Path::new(&path).join(DOE_FILE);
                if filepath.is_file() {
                    info!("Reading DOE from {:?}", filepath);
                    Some(read_npy(filepath)?)
                } else if std::path::Path::new(&path).join(DOE_INITIAL_FILE).is_file() {
                    let filepath = std::path::Path::new(&path).join(DOE_INITIAL_FILE);
                    info!("Reading DOE from {:?}", filepath);
                    Some(read_npy(filepath)?)
                } else {
                    None
                }
            } else {
                None
            };

        let doe = hstart_doe.as_ref().or(self.config.doe.as_ref());

        let (y_data, x_data) = if let Some(doe) = doe {
            if doe.ncols() == self.xlimits.nrows() {
                // only x are specified
                info!("Compute initial DOE on specified {} points", doe.nrows());
                (self.eval_obj(problem, doe), doe.to_owned())
            } else {
                // split doe in x and y
                info!("Use specified DOE {} samples", doe.nrows());
                (
                    doe.slice(s![.., self.xlimits.nrows()..]).to_owned(),
                    doe.slice(s![.., ..self.xlimits.nrows()]).to_owned(),
                )
            }
        } else {
            let n_doe = if self.config.n_doe == 0 {
                (self.xlimits.nrows() + 1).max(5)
            } else {
                self.config.n_doe
            };
            info!("Compute initial LHS with {} points", n_doe);
            let x = sampling.sample(n_doe);
            (self.eval_obj(problem, &x), x)
        };
        let doe = concatenate![Axis(1), x_data, y_data];
        if self.config.outdir.is_some() {
            let path = self.config.outdir.as_ref().unwrap();
            std::fs::create_dir_all(path)?;
            let filepath = std::path::Path::new(path).join(DOE_INITIAL_FILE);
            info!("Save initial doe shape {:?} in {:?}", doe.shape(), filepath);
            write_npy(filepath, &doe).expect("Write initial doe");
        }

        let clusterings = vec![None; self.config.n_cstr + 1];
        let theta_inits = vec![None; self.config.n_cstr + 1];
        let no_point_added_retries = MAX_POINT_ADDITION_RETRY;

        let mut initial_state = state
            .data((x_data, y_data))
            .clusterings(clusterings)
            .theta_inits(theta_inits)
            .sampling(sampling);
        initial_state.doe_size = doe.nrows();
        initial_state.max_iters = self.config.max_iters as u64;
        initial_state.added = doe.nrows();
        initial_state.no_point_added_retries = no_point_added_retries;
        initial_state.cstr_tol = self
            .config
            .cstr_tol
            .clone()
            .unwrap_or(Array1::from_elem(self.config.n_cstr, DEFAULT_CSTR_TOL));
        initial_state.target_cost = self.config.target;
        debug!("INITIAL STATE = {:?}", initial_state);
        Ok((initial_state, None))
    }

    fn next_iter(
        &mut self,
        fobj: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> std::result::Result<(EgorState<f64>, Option<KV>), argmin::core::Error> {
        info!(
            "********* Start iteration {}/{}",
            state.get_iter() + 1,
            state.get_max_iters()
        );
        let now = Instant::now();
        // Retrieve Egor internal state
        let mut clusterings =
            state
                .clone()
                .take_clusterings()
                .ok_or_else(argmin_error_closure!(
                    PotentialBug,
                    "EgorSolver: No clustering!"
                ))?;
        let mut theta_inits =
            state
                .clone()
                .take_theta_inits()
                .ok_or_else(argmin_error_closure!(
                    PotentialBug,
                    "EgorSolver: No theta inits!"
                ))?;
        let sampling = state
            .clone()
            .take_sampling()
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "EgorSolver: No sampling!"
            ))?;
        let (mut x_data, mut y_data) = state
            .clone()
            .take_data()
            .ok_or_else(argmin_error_closure!(PotentialBug, "EgorSolver: No data!"))?;

        let mut new_state = state.clone();
        let rejected_count = loop {
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

            let (x_dat, y_dat) = self.next_points(
                init,
                new_state.get_iter(),
                recluster,
                &mut clusterings,
                &mut theta_inits,
                &x_data,
                &y_data,
                &state.cstr_tol,
                &sampling,
                lhs_optim_seed,
            );

            debug!("Try adding {}", x_dat);
            let added_indices = update_data(&mut x_data, &mut y_data, &x_dat, &y_dat);

            new_state = new_state
                .clusterings(clusterings.clone())
                .theta_inits(theta_inits.clone())
                .data((x_data.clone(), y_data.clone()))
                .sampling(sampling.clone())
                .param(x_dat.row(0).to_owned())
                .cost(y_dat.row(0).to_owned());

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
                    return Ok((
                        state.terminate_with(TerminationReason::SolverExit(
                            "Even LHS optimization failed to add a new point".to_string(),
                        )),
                        None,
                    ));
                }
            } else {
                // ok point added we can go on, just output number of rejected point
                break rejected_count;
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
        new_state.no_point_added_retries = MAX_POINT_ADDITION_RETRY; // reset as a point is added

        let y_actual = self.eval_obj(fobj, &x_to_eval);
        Zip::from(y_data.slice_mut(s![-add_count.., ..]).columns_mut())
            .and(y_actual.columns())
            .for_each(|mut y, val| y.assign(&val));
        let doe = concatenate![Axis(1), x_data, y_data];
        if self.config.outdir.is_some() {
            let path = self.config.outdir.as_ref().unwrap();
            std::fs::create_dir_all(path)?;
            let filepath = std::path::Path::new(path).join(DOE_FILE);
            info!("Save doe shape {:?} in {:?}", doe.shape(), filepath);
            write_npy(filepath, &doe).expect("Write current doe");
        }
        let best_index = find_best_result_index(&y_data, &new_state.cstr_tol);
        info!(
            "********* End iteration {}/{} in {:.3}s: Best fun(x)={} at x={}",
            new_state.get_iter() + 1,
            new_state.get_max_iters(),
            now.elapsed().as_secs_f64(),
            y_data.row(best_index),
            x_data.row(best_index)
        );
        new_state = new_state.data((x_data, y_data.clone()));
        Ok((new_state, None))
    }

    fn terminate(&mut self, state: &EgorState<f64>) -> TerminationStatus {
        debug!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> end iteration");
        debug!("Current Cost {:?}", state.get_cost());
        debug!("Best cost {:?}", state.get_best_cost());
        // XXX: Should check target cost taking into account constraints
        // if state.get_best_cost() <= state.get_target_cost() {
        //     info!("Target optimum : {}", self.target);
        //     info!("Expected optimum reached!");
        //     return TerminationReason::TargetCostReached;
        // }
        TerminationStatus::NotTerminated
    }
}

impl<SB> EgorSolver<SB>
where
    SB: SurrogateBuilder,
{
    fn have_to_recluster(&self, added: usize, prev_added: usize) -> bool {
        self.config.n_clusters == 0 && (added != 0 && added % 10 == 0 && added - prev_added > 0)
    }

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

    #[allow(clippy::too_many_arguments)]
    fn next_points(
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
    ) -> (Array2<f64>, Array2<f64>) {
        debug!("Make surrogate with {}", x_data);
        let mut x_dat = Array2::zeros((0, x_data.ncols()));
        let mut y_dat = Array2::zeros((0, y_data.ncols()));
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
            let models: Vec<std::boxed::Box<dyn egobox_moe::MixtureGpSurrogate>> =
                (0..=self.config.n_cstr)
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
            info!("... surrogates trained");

            match self.find_best_point(
                x_data,
                y_data,
                sampling,
                obj_model.as_ref(),
                cstr_models,
                cstr_tol,
                lhs_optim,
            ) {
                Ok(xk) => {
                    match self.get_virtual_point(&xk, y_data, obj_model.as_ref(), cstr_models) {
                        Ok(yk) => {
                            y_dat = concatenate![
                                Axis(0),
                                y_dat,
                                Array2::from_shape_vec((1, 1 + self.config.n_cstr), yk).unwrap()
                            ];
                            x_dat = concatenate![Axis(0), x_dat, xk.insert_axis(Axis(0))];
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
        (x_dat, y_dat)
    }

    /// True whether surrogate gradient computation implemented
    fn is_grad_impl_available(&self) -> bool {
        true
    }

    fn compute_scaling(
        &self,
        sampling: &Lhs<f64, Xoshiro256Plus>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        f_min: f64,
    ) -> (f64, Array1<f64>, f64) {
        let npts = (100 * self.xlimits.nrows()).min(1000);
        info!("Use {npts} to evaluate scalers");
        let scaling_points = sampling.sample(npts);
        let scale_infill_obj =
            self.compute_infill_obj_scale(&scaling_points.view(), obj_model, f_min);
        info!(
            "Infill criterion scaler is updated to {}",
            scale_infill_obj
        );
        let scale_cstr = if cstr_models.is_empty() {
            Array1::zeros((0,))
        } else {
            let scale_cstr = compute_cstr_scales(&scaling_points.view(), cstr_models);
            info!("Constraints scaler is updated to {}", scale_cstr);
            scale_cstr
        };
        let scale_ic =
            self.config
                .infill_criterion
                .scaling(&scaling_points.view(), obj_model, f_min);
        (scale_infill_obj, scale_cstr, scale_ic)
    }

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
    ) -> Result<Array1<f64>> {
        let f_min = y_data.min().unwrap();

        let mut success = false;
        let mut n_optim = 1;
        let n_max_optim = 3;
        let mut best_x = None;

        let (scale_infill_obj, scale_cstr, scale_wb2) =
            self.compute_scaling(sampling, obj_model, cstr_models, *f_min);

        let algorithm = match self.config.infill_optimizer {
            InfillOptimizer::Slsqp => crate::optimizer::Algorithm::Slsqp,
            InfillOptimizer::Cobyla => crate::optimizer::Algorithm::Cobyla,
        };

        let obj = |x: &[f64], gradient: Option<&mut [f64]>, params: &mut ObjData<f64>| -> f64 {
            // Defensive programming NlOpt::Cobyla may pass NaNs
            if x.iter().any(|x| x.is_nan()) {
                return f64::INFINITY;
            }
            let ObjData {
                scale_infill_obj,
                scale_wb2,
                ..
            } = params;
            if let Some(grad) = gradient {
                if self.is_grad_impl_available() {
                    let grd = self
                        .eval_grad_infill_obj(x, obj_model, *f_min, *scale_infill_obj, *scale_wb2)
                        .to_vec();
                    grad[..].copy_from_slice(&grd);
                } else {
                    let f = |x: &Vec<f64>| -> f64 {
                        self.eval_infill_obj(x, obj_model, *f_min, *scale_infill_obj, *scale_wb2)
                    };
                    grad[..].copy_from_slice(&x.to_vec().central_diff(&f));
                }
            }
            self.eval_infill_obj(x, obj_model, *f_min, *scale_infill_obj, *scale_wb2)
        };

        let cstrs: Vec<_> = (0..self.config.n_cstr)
            .map(|i| {
                let index = i;
                let cstr = move |x: &[f64],
                                 gradient: Option<&mut [f64]>,
                                 params: &mut ObjData<f64>|
                      -> f64 {
                    if let Some(grad) = gradient {
                        if self.is_grad_impl_available() {
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
                        } else {
                            let f = |x: &Vec<f64>| -> f64 {
                                cstr_models[i]
                                    .predict(
                                        &Array::from_shape_vec((1, x.len()), x.to_vec())
                                            .unwrap()
                                            .view(),
                                    )
                                    .unwrap()[[0, 0]]
                                    / params.scale_cstr[index]
                            };
                            grad[..].copy_from_slice(&x.to_vec().central_diff(&f));
                        }
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
                    Box::new(cstr) as Box<dyn nlopt::ObjFn<ObjData<f64>> + Sync>
                }
                #[cfg(not(feature = "nlopt"))]
                {
                    Box::new(cstr) as Box<dyn crate::types::ObjFn<ObjData<f64>> + Sync>
                }
            })
            .collect();
        let cstr_refs: Vec<_> = cstrs.iter().map(|c| c.as_ref()).collect();
        info!("Optimize infill criterion...");
        let obj_data = ObjData {
            scale_infill_obj,
            scale_cstr: scale_cstr.to_owned(),
            scale_wb2,
        };
        while !success && n_optim <= n_max_optim {
            let x_start = sampling.sample(self.config.n_start);

            if let Some(seed) = lhs_optim_seed {
                let (_, x_opt) =
                    Optimizer::new(Algorithm::Lhs, &obj, &cstr_refs, &obj_data, &self.xlimits)
                        .cstr_tol(cstr_tol.to_owned())
                        .seed(seed)
                        .minimize();

                info!("LHS optimization best_x {}", x_opt);
                best_x = Some(x_opt);
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
                    best_x = Some(Array::from(res.1.clone()));
                    success = true;
                }
            }

            if n_optim == n_max_optim && best_x.is_none() {
                info!("All optimizations fail => Trigger LHS optimization");
                let (_, x_opt) =
                    Optimizer::new(Algorithm::Lhs, &obj, &cstr_refs, &obj_data, &self.xlimits)
                        .minimize();

                info!("LHS optimization best_x {}", x_opt);
                best_x = Some(x_opt);
                success = true;
            }
            n_optim += 1;
        }
        if best_x.is_some() {
            debug!("... infill criterion optimum found");
        }
        best_x.ok_or_else(|| EgoError::EgoError(String::from("Can not find best point")))
    }

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

    fn eval_infill_obj(
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

    fn eval_obj<O: CostFunction<Param = Array2<f64>, Output = Array2<f64>>>(
        &self,
        pb: &mut Problem<O>,
        x: &Array2<f64>,
    ) -> Array2<f64> {
        let params = if self.config.discrete() {
            // We have to cast x to folded space as EgorSolver
            // works internally in the continuous space
            to_discrete_space(&self.config.xtypes, x)
        } else {
            x.to_owned()
        };
        pb.problem("cost_count", |problem| problem.cost(&params))
            .expect("Objective evaluation")
    }
}
