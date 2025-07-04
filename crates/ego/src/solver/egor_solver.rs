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
//! let xtypes = to_xtypes(&array![[-2., 2.], [-2., 2.]]);
//! let fobj = ObjFunc::new(rosenb);
//! let config = EgorConfig::default().xtypes(&xtypes).seed(42);
//! let solver: EgorSolver<GpMixtureParams<f64>> = EgorSolver::new(config);
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
//!     .seed(42)
//!     .target(-5.5080);
//!
//! let solver: EgorSolver<GpMixtureParams<f64>> =
//!   EgorSolver::new(config);
//!
//! let res = Executor::new(fobj, solver)
//!             .configure(|state| state.max_iters(40))
//!             .run()
//!             .expect("g24 minimized");
//! println!("G24 min result = {:?}", res.state);
//! ```
//!
use crate::utils::find_best_result_index;
use crate::{EgoError, EgorConfig, EgorState, MAX_POINT_ADDITION_RETRY};

use crate::types::*;

use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use log::{debug, info};
use ndarray::{concatenate, s, Array1, Array2, ArrayBase, Axis, Data, Ix2, Zip};
use ndarray_npy::{read_npy, write_npy};

use argmin::core::{
    CostFunction, Problem, Solver, State, TerminationReason, TerminationStatus, KV,
};

use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::marker::PhantomData;
use std::time::Instant;

/// Numpy filename for initial DOE dump
pub const DOE_INITIAL_FILE: &str = "egor_initial_doe.npy";
/// Numpy filename for current DOE dump
pub const DOE_FILE: &str = "egor_doe.npy";

/// Default tolerance value for constraints to be satisfied (ie cstr < tol)
pub const DEFAULT_CSTR_TOL: f64 = 1e-4;

/// Implementation of `argmin::core::Solver` for Egor optimizer.
/// Therefore this structure can be used with `argmin::core::Executor` and benefit
/// from observers and checkpointing features.
#[derive(Clone, Serialize, Deserialize)]
pub struct EgorSolver<SB: SurrogateBuilder, C: CstrFn = Cstr> {
    pub(crate) config: EgorConfig,
    /// Matrix (nx, 2) of [lower bound, upper bound] of the nx components of x
    /// Note: used for continuous variables handling, the optimizer base.
    pub(crate) xlimits: Array2<f64>,
    /// An optional surrogate builder used to model objective and constraint
    /// functions, otherwise [mixture of expert](egobox_moe) is used
    /// Note: if specified takes precedence over individual settings
    pub(crate) surrogate_builder: SB,

    pub phantom: PhantomData<C>,
}

/// Build `xtypes` from simple float bounds of `x` input components when x belongs to R^n.
/// xlimits are bounds of the x components expressed a matrix (dim, 2) where dim is the dimension of x
/// the ith row is the bounds interval [lower, upper] of the ith comonent of `x`.  
pub fn to_xtypes(xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Vec<XType> {
    let mut xtypes: Vec<XType> = vec![];
    Zip::from(xlimits.rows()).for_each(|limits| xtypes.push(XType::Float(limits[0], limits[1])));
    xtypes
}

impl<O, SB, C> Solver<O, EgorState<f64>> for EgorSolver<SB, C>
where
    O: CostFunction<Param = Array2<f64>, Output = Array2<f64>> + DomainConstraints<C>,
    C: CstrFn,
    SB: SurrogateBuilder + DeserializeOwned,
{
    const NAME: &'static str = "Egor";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> std::result::Result<(EgorState<f64>, Option<KV>), argmin::core::Error> {
        let mut rng = if let Some(seed) = self.config.seed {
            Xoshiro256Plus::seed_from_u64(seed)
        } else {
            Xoshiro256Plus::from_entropy()
        };

        let hstart_doe: Option<Array2<f64>> =
            if self.config.warm_start && self.config.outdir.is_some() {
                let path: &String = self.config.outdir.as_ref().unwrap();
                let filepath = std::path::Path::new(&path).join(DOE_FILE);
                if filepath.is_file() {
                    info!("Reading DOE from {filepath:?}");
                    Some(read_npy(filepath)?)
                } else if std::path::Path::new(&path).join(DOE_INITIAL_FILE).is_file() {
                    let filepath = std::path::Path::new(&path).join(DOE_INITIAL_FILE);
                    info!("Reading DOE from {filepath:?}");
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
            info!("Compute initial LHS with {n_doe} points");
            let sampling = Lhs::new(&self.xlimits)
                .with_rng(rng.clone())
                .kind(LhsKind::Maximin);
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

        let c_data = self.eval_problem_fcstrs(problem, &x_data);

        let activity = if self.config.coego.activated {
            let activity = self.get_random_activity(&mut rng);
            debug!("Component activity = {activity:?}");
            Some(activity)
        } else {
            None
        };

        let mut initial_state = state
            .data((x_data, y_data.clone(), c_data.clone()))
            .clusterings(clusterings)
            .theta_inits(theta_inits)
            .rng(rng);

        initial_state.doe_size = doe.nrows();
        initial_state.max_iters = self.config.max_iters as u64;
        initial_state.added = doe.nrows();
        initial_state.no_point_added_retries = no_point_added_retries;
        initial_state.cstr_tol = self.config.cstr_tol.clone().unwrap_or(Array1::from_elem(
            self.config.n_cstr + c_data.ncols(),
            DEFAULT_CSTR_TOL,
        ));
        initial_state.target_cost = self.config.target;

        let best_index = find_best_result_index(&y_data, &c_data, &initial_state.cstr_tol);
        initial_state.best_index = Some(best_index);
        initial_state.prev_best_index = Some(best_index);
        initial_state.last_best_iter = 0;

        initial_state.activity = activity;
        debug!("Initial State = {initial_state:?}");
        Ok((initial_state, None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> std::result::Result<(EgorState<f64>, Option<KV>), argmin::core::Error> {
        debug!(
            "********* Start iteration {}/{}",
            state.get_iter() + 1,
            state.get_max_iters()
        );
        let now = Instant::now();

        let mut res = if self.config.trego.activated {
            self.trego_iteration(problem, state)?
        } else {
            self.ego_iteration(problem, state)?
        };
        let (x_data, y_data, _c_data) = res.0.data.clone().unwrap();

        // Update Coop activity
        let res = if self.config.coego.activated {
            let mut rng = res.0.take_rng().unwrap();
            let activity = self.get_random_activity(&mut rng);
            debug!("Component activity = {activity:?}");
            (res.0.rng(rng).activity(activity), res.1)
        } else {
            res
        };

        info!(
            "********* End iteration {}/{} in {:.3}s: Best fun(x[{}])={} at x={}",
            res.0.get_iter() + 1,
            res.0.get_max_iters(),
            now.elapsed().as_secs_f64(),
            res.0.best_index.unwrap(),
            y_data.row(res.0.best_index.unwrap()),
            x_data.row(res.0.best_index.unwrap())
        );
        Ok(res)
    }

    fn terminate(&mut self, state: &EgorState<f64>) -> TerminationStatus {
        debug!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> end iteration");
        debug!("Current Cost {:?}", state.get_cost());
        debug!("Best cost {:?}", state.get_best_cost());
        debug!("Best index {:?}", state.best_index);
        debug!("Data {:?}", state.data.as_ref().unwrap());

        TerminationStatus::NotTerminated
    }
}

impl<SB, C: CstrFn> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + DeserializeOwned,
{
    /// Iteration of EGO algorithm
    fn ego_iteration<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>> + DomainConstraints<C>,
    >(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> std::result::Result<(EgorState<f64>, Option<KV>), argmin::core::Error> {
        match self.ego_step(problem, state.clone()) {
            Ok(new_state) => Ok((new_state, None)),
            Err(EgoError::NoMorePointToAddError(state)) => Ok((
                state.terminate_with(TerminationReason::SolverConverged),
                None,
            )),
            Err(err) => Err(err.into()),
        }
    }

    /// Iteration of TREGO algorithm
    fn trego_iteration<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>> + DomainConstraints<C>,
    >(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> std::result::Result<(EgorState<f64>, Option<KV>), argmin::core::Error> {
        let rho = |sigma| sigma * sigma;
        let (_, y_data, _) = state.data.as_ref().unwrap(); // initialized in init
        let best = state.best_index.unwrap(); // initialized in init
        let prev_best = state.prev_best_index.unwrap(); // initialized in init

        // Check prev step success
        let diff = y_data[[prev_best, 0]] - rho(state.sigma);
        let last_iter_success = y_data[[best, 0]] < diff;
        info!(
            "success = {} as {} {} {} - {}",
            last_iter_success,
            y_data[[best, 0]],
            if last_iter_success { "<" } else { ">=" },
            y_data[[prev_best, 0]],
            rho(state.sigma)
        );
        let mut new_state = state.clone();
        if !state.prev_step_ego && state.get_iter() != 0 {
            // Adjust trust region wrt local step success
            if last_iter_success {
                let old = state.sigma;
                new_state.sigma *= self.config.trego.gamma;
                info!(
                    "Previous TREGO local step successful: sigma {} -> {}",
                    old, new_state.sigma
                );
            } else {
                let old = state.sigma;
                new_state.sigma *= self.config.trego.beta;
                info!(
                    "Previous TREGO local step progress fail: sigma {} -> {}",
                    old, new_state.sigma
                );
            }
        } else if state.get_iter() != 0 {
            // Adjust trust region wrt global step success
            if last_iter_success {
                let old = state.sigma;
                new_state.sigma *= self.config.trego.gamma;
                info!(
                    "Previous EGO global step successful: sigma {} -> {}",
                    old, new_state.sigma
                );
            } else {
                info!("Previous EGO global step progress fail");
            }
        }

        let is_global_phase = (last_iter_success && state.prev_step_ego)
            || ((state.get_iter() % (1 + self.config.trego.n_local_steps)) == 0);

        if is_global_phase {
            // Global step
            info!(">>> TREGO global step (aka EGO)");
            let mut res = self.ego_iteration(problem, new_state)?;
            res.0.prev_step_ego = true;
            Ok(res)
        } else {
            info!(">>> TREGO local step");
            // Local step
            let models = self.refresh_surrogates(&new_state);
            let infill_data = self.refresh_infill_data(problem, &mut new_state, &models);
            let mut new_state = self.trego_step(problem, new_state, models, &infill_data);
            new_state.prev_step_ego = false;
            Ok((new_state, None))
        }
    }
}
