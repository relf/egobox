//! Egor optimizer implements EGO algorithm with basic handling of constraints.
//!
//! ```no_run
//! # use ndarray::{array, Array2, ArrayView1, ArrayView2, Zip};
//! # use egobox_doe::{Lhs, SamplingMethod};
//! # use egobox_ego::{EgorBuilder, InfillStrategy, InfillOptimizer};
//! # use rand_xoshiro::Xoshiro256Plus;
//! # use ndarray_rand::rand::SeedableRng;
//! use argmin_testfunctions::rosenbrock;
//!
//! // Rosenbrock test function: minimum y_opt = 0 at x_opt = (1, 1)
//! fn rosenb(x: &ArrayView2<f64>) -> Array2<f64> {
//!     let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
//!     Zip::from(y.rows_mut())
//!         .and(x.rows())
//!         .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec(), 1., 100.)]));
//!     y
//! }
//!
//! let xlimits = array![[-2., 2.], [-2., 2.]];
//! let res = EgorBuilder::optimize(rosenb)
//!     .min_within(&xlimits)
//!     .infill_strategy(InfillStrategy::EI)
//!     .n_doe(10)
//!     .target(1e-1)
//!     .n_eval(30)
//!     .run()
//!     .expect("Rosenbrock minimization");
//! println!("Rosenbrock min result = {:?}", res);
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
//! # use ndarray::{array, Array2, ArrayView1, ArrayView2, Zip};
//! # use egobox_doe::{Lhs, SamplingMethod};
//! # use egobox_ego::{EgorBuilder, InfillStrategy, InfillOptimizer};
//! # use rand_xoshiro::Xoshiro256Plus;
//! # use ndarray_rand::rand::SeedableRng;
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
//! let res = EgorBuilder::optimize(f_g24)
//!            .min_within(&xlimits)
//!            .n_cstr(2)
//!            .infill_strategy(InfillStrategy::EI)
//!            .infill_optimizer(InfillOptimizer::Cobyla)
//!            .doe(Some(doe))
//!            .n_eval(40)
//!            .target(-5.5080)
//!            .run()
//!            .expect("g24 minimized");
//! println!("G24 min result = {:?}", res);
//! ```
//!
use crate::egor_state::{find_best_result_index, EgorState, MAX_POINT_ADDITION_RETRY};
use crate::errors::{EgoError, Result};
use crate::lhs_optimizer::LhsOptimizer;
use crate::mixint::*;
use crate::types::*;
use crate::utils::{compute_cstr_scales, compute_obj_scale, compute_wb2s_scale, grad_wbs2};
use crate::utils::{ei, wb2s};
use crate::utils::{grad_ei, update_data};

use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use egobox_moe::{ClusteredSurrogate, Clustering, CorrelationSpec, MoeParams, RegressionSpec};
use env_logger::{Builder, Env};
use finitediff::FiniteDiff;
use linfa::ParamGuard;
use log::{debug, info, warn};
use ndarray::{
    concatenate, s, Array, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2, Zip,
};
use ndarray_npy::{read_npy, write_npy};
use ndarray_rand::rand::SeedableRng;
use ndarray_stats::QuantileExt;
use nlopt::*;
use rand_xoshiro::Xoshiro256Plus;

use argmin::argmin_error_closure;
use argmin::core::{CostFunction, Executor, Problem, Solver, State, TerminationReason, KV};

#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

const DOE_INITIAL_FILE: &str = "egor_initial_doe.npy";
const DOE_FILE: &str = "egor_doe.npy";

/// Implementation of `argmin::core::Solver` for Egor optimizer.
/// Therefore this structure can be used with `argmin::core::Executor` and benefit
/// from observers and checkpointing features.
#[derive(Clone)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct EgorSolver<SB: SurrogateBuilder> {
    /// Number of function evaluations allocated to find the optimum (aka evaluation budget)
    /// Note 1: if the initial doe has to be evaluated, doe size is taken into account in the avaluation budget.
    /// Note 2: Number of iteration is deduced using the following formula (n_eval - initial doe to evaluate) / q_parallel  
    n_eval: usize,
    /// Number of starts for multistart approach used for hyperparameters optimization
    n_start: usize,
    /// Number of parallel points evaluated for each "function evaluation"
    q_parallel: usize,
    /// Number of initial doe drawn using Latin hypercube sampling
    /// Note: n_doe > 0; otherwise n_doe = max(xdim+1, 5)
    n_doe: usize,
    /// Number of Constraints
    /// Note: dim function ouput = 1 objective + n_cstr constraints
    n_cstr: usize,
    /// Constraints violation tolerance meaning cstr < cstr_tol is considered valid
    cstr_tol: f64,
    /// Initial doe can be either \[x\] with x inputs only or an evaluated doe \[x, y\]
    /// Note: x dimension is determined using xlimits
    doe: Option<Array2<f64>>,
    /// Parallel strategy used to define several points (q_parallel) evaluations at each iteration
    q_ei: QEiStrategy,
    /// Criterium to select next point to evaluate
    infill: InfillStrategy,
    /// The optimizer used to optimize infill criterium
    infill_optimizer: InfillOptimizer,
    /// Regression specification for GP models used by mixture of experts (see [egobox_moe])
    regression_spec: RegressionSpec,
    /// Correlation specification for GP models used by mixture of experts (see [egobox_moe])
    correlation_spec: CorrelationSpec,
    /// Optional dimension reduction (see [egobox_moe])
    kpls_dim: Option<usize>,
    /// Number of clusters used by mixture of experts (see [egobox_moe])
    /// When set to 0 the clusters are computes automatically and refreshed
    /// every 10-points (tentative) additions
    n_clusters: Option<usize>,
    /// Specification of a target objective value which is used to stop the algorithm once reached
    target: f64,
    /// Directory to save intermediate results: inital doe + evalutions at each iteration
    outdir: Option<String>,
    /// If true use <outdir> to retrieve and start from previous results
    hot_start: bool,
    /// Matrix (nx, 2) of [lower bound, upper bound] of the nx components of x
    /// Note: used for continuous variables handling, the optimizer base.
    xlimits: Array2<f64>,
    /// List of x types allowing the handling of discrete input variables
    xtypes: Option<Vec<Xtype>>,
    /// Flag for discrete handling, true if mixed-integer type present in xtypes, otherwise false
    no_discrete: bool,
    /// An optional surrogate builder used to model objective and constraint
    /// functions, otherwise [mixture of expert](egobox_moe) is used
    /// Note: if specified takes precedence over individual settings
    surrogate_builder: SB,
    /// A random generator used to get reproductible results.
    /// For instance: Xoshiro256Plus::from_u64_seed(42) for reproducibility
    rng: Xoshiro256Plus,
}

impl SurrogateBuilder for MoeParams<f64, Xoshiro256Plus> {
    fn new_with_xtypes_rng(_xtypes: &[Xtype]) -> Self {
        MoeParams::new()
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

    fn train(
        &self,
        xt: &ArrayView2<f64>,
        yt: &ArrayView2<f64>,
    ) -> Result<Box<dyn ClusteredSurrogate>> {
        let checked = self.check_ref()?;
        let moe = checked.train(xt, yt)?;
        Ok(moe).map(|moe| Box::new(moe) as Box<dyn ClusteredSurrogate>)
    }

    fn train_on_clusters(
        &self,
        xt: &ArrayView2<f64>,
        yt: &ArrayView2<f64>,
        clustering: &Clustering,
    ) -> Result<Box<dyn ClusteredSurrogate>> {
        let checked = self.check_ref()?;
        let moe = checked.train_on_clusters(xt, yt, clustering)?;
        Ok(moe).map(|moe| Box::new(moe) as Box<dyn ClusteredSurrogate>)
    }
}

impl<SB: SurrogateBuilder> EgorSolver<SB> {
    /// Constructor of the optimization of the function `f` with specified random generator
    /// to get reproducibility.
    ///
    /// The function `f` should return an objective value but also constraint values if any.
    /// Design space is specified by the matrix `xlimits` which is `[nx, 2]`-shaped
    /// the ith row contains lower and upper bounds of the ith component of `x`.
    pub fn new(xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>, rng: Xoshiro256Plus) -> Self {
        let env = Env::new().filter_or("EGOBOX_LOG", "info");
        let mut builder = Builder::from_env(env);
        let builder = builder.target(env_logger::Target::Stdout);
        builder.try_init().ok();
        EgorSolver {
            n_eval: 20,
            n_start: 20,
            q_parallel: 1,
            n_doe: 0,
            n_cstr: 0,
            cstr_tol: 1e-6,
            doe: None,
            q_ei: QEiStrategy::KrigingBeliever,
            infill: InfillStrategy::WB2,
            infill_optimizer: InfillOptimizer::Slsqp,
            regression_spec: RegressionSpec::CONSTANT,
            correlation_spec: CorrelationSpec::SQUAREDEXPONENTIAL,
            kpls_dim: None,
            n_clusters: Some(1),
            target: f64::NEG_INFINITY,
            outdir: None,
            hot_start: false,
            xlimits: xlimits.to_owned(),
            xtypes: Some(continuous_xlimits_to_xtypes(xlimits)),
            no_discrete: true,
            surrogate_builder: SB::new_with_xtypes_rng(&continuous_xlimits_to_xtypes(xlimits)),
            rng,
        }
    }

    /// Constructor of the optimization of the function `f` with specified random generator
    /// to get reproducibility. This constructor is used  for mixed-integer optimization
    /// when `f` has discrete inputs to be specified with list of xtypes.
    ///
    /// The function `f` should return an objective but also constraint values if any.
    /// Design space is specified by a list of types for input variables `x` of `f` (see [Xtype]).
    pub fn new_with_xtypes(xtypes: &[Xtype], rng: Xoshiro256Plus) -> Self {
        let env = Env::new().filter_or("EGOBOX_LOG", "info");
        let mut builder = Builder::from_env(env);
        let builder = builder.target(env_logger::Target::Stdout);
        builder.try_init().ok();
        let v_xtypes = xtypes.to_vec();
        let xlimits = unfold_xtypes_as_continuous_limits(xtypes);
        EgorSolver {
            n_eval: 20,
            n_start: 20,
            q_parallel: 1,
            n_doe: 0,
            n_cstr: 0,
            cstr_tol: 1e-6,
            doe: None,
            q_ei: QEiStrategy::KrigingBeliever,
            infill: InfillStrategy::WB2,
            infill_optimizer: InfillOptimizer::Slsqp,
            regression_spec: RegressionSpec::CONSTANT,
            correlation_spec: CorrelationSpec::SQUAREDEXPONENTIAL,
            kpls_dim: None,
            n_clusters: Some(1),
            target: f64::NEG_INFINITY,
            outdir: None,
            hot_start: false,
            xlimits,
            xtypes: Some(v_xtypes),
            surrogate_builder: SB::new_with_xtypes_rng(xtypes),
            no_discrete: !xtypes
                .iter()
                .any(|t| matches!(t, &Xtype::Int(_, _) | &Xtype::Ord(_) | &Xtype::Enum(_))),
            rng,
        }
    }

    /// Sets allowed number of evaluation of the function under optimization
    pub fn n_eval(&mut self, n_eval: usize) -> &mut Self {
        self.n_eval = n_eval;
        self
    }

    /// Sets the number of runs of infill strategy optimizations (best result taken)
    pub fn n_start(&mut self, n_start: usize) -> &mut Self {
        self.n_start = n_start;
        self
    }

    /// Sets Number of parallel evaluations of the function under optimization
    pub fn q_parallel(&mut self, q_parallel: usize) -> &mut Self {
        self.q_parallel = q_parallel;
        self
    }

    /// Number of samples of initial LHS sampling (used when DOE not provided by the user)
    ///
    /// When 0 a number of points is computed automatically regarding the number of input variables
    /// of the function under optimization.
    pub fn n_doe(&mut self, n_doe: usize) -> &mut Self {
        self.n_doe = n_doe;
        self
    }

    /// Sets the number of constraint functions
    pub fn n_cstr(&mut self, n_cstr: usize) -> &mut Self {
        self.n_cstr = n_cstr;
        self
    }

    /// Sets the tolerance on constraints violation (cstr < tol)
    pub fn cstr_tol(&mut self, tol: f64) -> &mut Self {
        self.cstr_tol = tol;
        self
    }

    /// Sets an initial DOE containing ns samples
    ///
    /// Either nt = nx then only x are specified and ns evals are done to get y doe values,
    /// or nt = nx + ny then x = doe(:, :nx) and y = doe(:, nx:) are specified
    pub fn doe(&mut self, doe: Option<Array2<f64>>) -> &mut Self {
        self.doe = doe.map(|x| x.to_owned());
        self
    }

    /// Sets the parallel infill strategy
    ///
    /// Parallel infill criteria to get virtual next promising points in order to allow
    /// n parallel evaluations of the function under optimization.
    pub fn qei_strategy(&mut self, q_ei: QEiStrategy) -> &mut Self {
        self.q_ei = q_ei;
        self
    }

    /// Sets the nfill criteria
    pub fn infill_strategy(&mut self, infill: InfillStrategy) -> &mut Self {
        self.infill = infill;
        self
    }

    /// Sets the infill optimizer
    pub fn infill_optimizer(&mut self, optimizer: InfillOptimizer) -> &mut Self {
        self.infill_optimizer = optimizer;
        self
    }

    /// Sets the allowed regression models used in gaussian processes.
    pub fn regression_spec(&mut self, regression_spec: RegressionSpec) -> &mut Self {
        self.regression_spec = regression_spec;
        self
    }

    /// Sets the allowed correlation models used in gaussian processes.
    pub fn correlation_spec(&mut self, correlation_spec: CorrelationSpec) -> &mut Self {
        self.correlation_spec = correlation_spec;
        self
    }

    /// Sets the number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
    ///
    /// This is used to address high-dimensional problems typically when nx > 9.
    pub fn kpls_dim(&mut self, kpls_dim: Option<usize>) -> &mut Self {
        self.kpls_dim = kpls_dim;
        self
    }

    /// Sets the number of clusters used by the mixture of surrogate experts.
    ///
    /// When set to 0, the number of clusters is determined automatically
    pub fn n_clusters(&mut self, n_clusters: Option<usize>) -> &mut Self {
        self.n_clusters = n_clusters;
        self
    }

    /// Sets a known target minimum to be used as a stopping criterion.
    pub fn target(&mut self, target: f64) -> &mut Self {
        self.target = target;
        self
    }

    /// Sets a directory to write optimization history and used as search path for hot start doe
    pub fn outdir(&mut self, outdir: Option<String>) -> &mut Self {
        self.outdir = outdir;
        self
    }

    /// Whether we start by loading last DOE saved in `outdir` as initial DOE
    pub fn hot_start(&mut self, hot_start: bool) -> &mut Self {
        self.hot_start = hot_start;
        self
    }

    /// Given an evaluated doe (x, y) data, return the next promising x point
    /// where optimum may occurs regarding the infill criterium.
    /// This function inverse the control of the optimization and can used
    /// ask-and-tell interface to the EGO optmizer.
    pub fn suggest(
        &self,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Array2<f64> {
        let rng = self.rng.clone();
        let sampling = Lhs::new(&self.xlimits).with_rng(rng).kind(LhsKind::Maximin);
        let mut clusterings = vec![None; 1 + self.n_cstr];
        let (x_dat, _) = self.next_points(
            true,
            false, // done anyway
            &mut clusterings,
            x_data,
            y_data,
            &sampling,
            None,
        );
        x_dat
    }
}

/// Build ``xtypes` from simple float bounds of `x` input components when x belongs to R^n.
/// xlimits are bounds of the x components expressed a matrix (dim, 2) where dim is the dimension of x
/// the ith row is the bounds interval [lower, upper] of the ith comonent of `x`.  
fn continuous_xlimits_to_xtypes(xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Vec<Xtype> {
    let mut xtypes: Vec<Xtype> = vec![];
    Zip::from(xlimits.rows()).for_each(|limits| xtypes.push(Xtype::Cont(limits[0], limits[1])));
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

        let hstart_doe: Option<Array2<f64>> = if self.hot_start && self.outdir.is_some() {
            let path: &String = self.outdir.as_ref().unwrap();
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

        let doe = hstart_doe.as_ref().or(self.doe.as_ref());

        let (y_data, x_data, n_eval) = if let Some(doe) = doe {
            if doe.ncols() == self.xlimits.nrows() {
                // only x are specified
                info!("Compute initial DOE on specified {} points", doe.nrows());
                (
                    self.eval(problem, doe),
                    doe.to_owned(),
                    self.n_eval.saturating_sub(doe.nrows()),
                )
            } else {
                // split doe in x and y
                info!("Use specified DOE {} samples", doe.nrows());
                (
                    doe.slice(s![.., self.xlimits.nrows()..]).to_owned(),
                    doe.slice(s![.., ..self.xlimits.nrows()]).to_owned(),
                    self.n_eval,
                )
            }
        } else {
            let n_doe = if self.n_doe == 0 {
                (self.xlimits.nrows() + 1).max(5)
            } else {
                self.n_doe
            };
            info!("Compute initial LHS with {} points", n_doe);
            let x = sampling.sample(n_doe);
            (self.eval(problem, &x), x, self.n_eval - n_doe)
        };
        let doe = concatenate![Axis(1), x_data, y_data];
        if self.outdir.is_some() {
            let path = self.outdir.as_ref().unwrap();
            std::fs::create_dir_all(path)?;
            let filepath = std::path::Path::new(path).join(DOE_INITIAL_FILE);
            info!("Save initial doe in {:?}", filepath);
            write_npy(filepath, &doe).expect("Write initial doe");
        }

        let clusterings = vec![None; self.n_cstr + 1];
        let no_point_added_retries = MAX_POINT_ADDITION_RETRY;
        if n_eval / self.q_parallel == 0 {
            warn!(
                "Number of evaluations {} too low (initial doe size={} and q_parallel={})",
                self.n_eval,
                doe.nrows(),
                self.q_parallel
            );
        }
        let n_eval = if n_eval % self.q_parallel != 0 {
            let new_n_eval = (n_eval / self.q_parallel) * self.q_parallel;
            warn!(
                "Number of evaluations out of initial doe {} readjusted to {} to get a multiple of q_parallel={}",
                n_eval,
                new_n_eval,
                self.q_parallel
            );
            new_n_eval
        } else {
            n_eval
        };
        let n_iter = n_eval / self.q_parallel;

        let mut initial_state = state
            .data((x_data, y_data))
            .clusterings(clusterings)
            .sampling(sampling);
        initial_state.max_iters = n_iter as u64;
        initial_state.added = doe.nrows();
        initial_state.no_point_added_retries = no_point_added_retries;
        initial_state.cstr_tol = self.cstr_tol;
        initial_state.target_cost = self.target;
        debug!("INITIAL STATE = {:?}", initial_state);
        Ok((initial_state, None))
    }

    fn next_iter(
        &mut self,
        fobj: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> std::result::Result<(EgorState<f64>, Option<KV>), argmin::core::Error> {
        debug!(
            "********* Start iteration {}/{}",
            state.get_iter() + 1,
            state.get_max_iters()
        );
        // Retrieve Egor internal state
        let mut clusterings =
            state
                .clone()
                .take_clusterings()
                .ok_or_else(argmin_error_closure!(
                    PotentialBug,
                    "EgorSolver: No clustering!"
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
                recluster,
                &mut clusterings,
                &x_data,
                &y_data,
                &sampling,
                lhs_optim_seed,
            );

            debug!("Try adding {}", x_dat);
            let added_indices = update_data(&mut x_data, &mut y_data, &x_dat, &y_dat);

            new_state = new_state
                .clusterings(clusterings.clone())
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
                    info!("Use LHS optimization to hopefully ensure a point addition");
                }
                if new_state.no_point_added_retries < 0 {
                    // no luck with LHS optimization
                    warn!("Fail to add another point to improve the surrogate models. Abort!");
                    return Ok((
                        state.terminate_with(TerminationReason::NoChangeInCost),
                        None,
                    ));
                }
            } else {
                // ok point added we can go on, just output number of rejected point
                break rejected_count;
            }
        };

        let add_count = (self.q_parallel - rejected_count) as i32;
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

        let y_actual = self.eval(fobj, &x_to_eval);
        Zip::from(y_data.slice_mut(s![-add_count.., ..]).columns_mut())
            .and(y_actual.columns())
            .for_each(|mut y, val| y.assign(&val));
        let doe = concatenate![Axis(1), x_data, y_data];
        if self.outdir.is_some() {
            let path = self.outdir.as_ref().unwrap();
            std::fs::create_dir_all(path)?;
            let filepath = std::path::Path::new(path).join(DOE_FILE);
            info!("Save doe in {:?}", filepath);
            write_npy(filepath, &doe).expect("Write current doe");
        }
        let best_index = self.find_best_result_index(&y_data);
        info!(
            "********* End iteration {}/{}: Best fun(x)={} at x={}",
            new_state.get_iter() + 1,
            new_state.get_max_iters(),
            y_data.row(best_index),
            x_data.row(best_index)
        );
        new_state = new_state.data((x_data, y_data.clone()));
        Ok((new_state, None))
    }

    fn terminate(&mut self, state: &EgorState<f64>) -> TerminationReason {
        debug!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TERMINATE");
        debug!("Current Cost {:?}", state.get_cost());
        debug!("Best cost {:?}", state.get_best_cost());
        debug!("Target cost {:?}", state.get_target_cost());
        if state.get_best_cost() <= state.get_target_cost() {
            info!("Target optimum : {}", self.target);
            info!("Expected optimum reached!");
            return TerminationReason::TargetCostReached;
        }
        if state.get_termination_reason() == TerminationReason::Aborted {
            info!("*********************************************************** Keyboard interruption ******");
            return TerminationReason::Aborted;
        }
        TerminationReason::NotTerminated
    }

    /// Checks whether basic termination reasons apply.
    ///
    /// Terminate if
    ///
    /// 1) algorithm was terminated somewhere else in the Executor
    /// 2) iteration count exceeds maximum number of iterations
    /// 3) cost is lower than or equal to the target cost
    ///
    /// This can be overwritten; however it is not advised. It is recommended to implement other
    /// stopping criteria via ([`terminate`](`Solver::terminate`).
    fn terminate_internal(&mut self, state: &EgorState<f64>) -> TerminationReason {
        let solver_terminate =
            <EgorSolver<SB> as Solver<O, EgorState<f64>>>::terminate(self, state);
        if solver_terminate.terminated() {
            return solver_terminate;
        }
        if state.get_iter() >= state.get_max_iters() {
            return TerminationReason::MaxItersReached;
        }
        if state.get_best_cost() <= state.get_target_cost() {
            return TerminationReason::TargetCostReached;
        }
        TerminationReason::NotTerminated
    }
}

impl<SB> EgorSolver<SB>
where
    SB: SurrogateBuilder,
{
    fn have_to_recluster(&self, added: usize, prev_added: usize) -> bool {
        if let Some(nc) = self.n_clusters {
            nc == 0 && (added != 0 && added % 10 == 0 && added - prev_added > 0)
        } else {
            false
        }
    }

    fn make_clustered_surrogate(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        init: bool,
        recluster: bool,
        clustering: &Option<Clustering>,
        model_name: &str,
    ) -> Box<dyn ClusteredSurrogate> {
        let mut builder = self.surrogate_builder.clone();
        builder.set_kpls_dim(self.kpls_dim);
        builder.set_regression_spec(self.regression_spec);
        builder.set_correlation_spec(self.correlation_spec);
        if let Some(nc) = self.n_clusters {
            if nc > 0 {
                builder.set_n_clusters(nc);
            }
        };

        if init || recluster {
            if recluster {
                info!("{} reclustering...", model_name);
            } else {
                info!("{} initial clustering...", model_name);
            }
            let model = builder
                .train(&xt.view(), &yt.view())
                .expect("GP training failure");
            info!(
                "... Best nb of clusters / mixture for {}: {} / {}",
                model_name,
                model.n_clusters(),
                model.recombination()
            );
            model
        } else {
            let clustering = clustering.as_ref().unwrap().clone();
            builder
                .train_on_clusters(&xt.view(), &yt.view(), &clustering)
                .expect("GP training failure")
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn next_points(
        &self,
        init: bool,
        recluster: bool,
        clusterings: &mut [Option<Clustering>],
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        sampling: &Lhs<f64, Xoshiro256Plus>,
        lhs_optim: Option<u64>,
    ) -> (Array2<f64>, Array2<f64>) {
        debug!("Make surrogate with {}", x_data);
        let mut x_dat = Array2::zeros((0, x_data.ncols()));
        let mut y_dat = Array2::zeros((0, y_data.ncols()));
        for i in 0..self.q_parallel {
            let (xt, yt) = if i == 0 {
                (x_data.to_owned(), y_data.to_owned())
            } else {
                (
                    concatenate![Axis(0), x_data.to_owned(), x_dat.to_owned()],
                    concatenate![Axis(0), y_data.to_owned(), y_dat.to_owned()],
                )
            };

            let obj_model = self.make_clustered_surrogate(
                &xt,
                &yt.slice(s![.., 0..1]).to_owned(),
                init && i == 0,
                recluster,
                &clusterings[0],
                "Objective",
            );
            clusterings[0] = Some(obj_model.to_clustering());

            let mut cstr_models: Vec<Box<dyn ClusteredSurrogate>> = Vec::with_capacity(self.n_cstr);
            for k in 1..=self.n_cstr {
                let cstr_model = self.make_clustered_surrogate(
                    &xt,
                    &yt.slice(s![.., k..k + 1]).to_owned(),
                    init && i == 0,
                    recluster,
                    &clusterings[k],
                    &format!("Constraint[{}] reclustering...", k),
                );
                cstr_models.push(cstr_model);
                clusterings[k] = Some(cstr_models[k - 1].to_clustering());
            }

            match self.find_best_point(
                x_data,
                y_data,
                sampling,
                obj_model.as_ref(),
                &cstr_models,
                lhs_optim,
            ) {
                Ok(xk) => {
                    match self.get_virtual_point(&xk, y_data, obj_model.as_ref(), &cstr_models) {
                        Ok(yk) => {
                            y_dat = concatenate![
                                Axis(0),
                                y_dat,
                                Array2::from_shape_vec((1, 1 + self.n_cstr), yk).unwrap()
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

    fn find_best_point(
        &self,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        sampling: &Lhs<f64, Xoshiro256Plus>,
        obj_model: &dyn ClusteredSurrogate,
        cstr_models: &[Box<dyn ClusteredSurrogate>],
        lhs_optim_seed: Option<u64>,
    ) -> Result<Array1<f64>> {
        let f_min = y_data.min().unwrap();

        let obj = |x: &[f64], gradient: Option<&mut [f64]>, params: &mut ObjData<f64>| -> f64 {
            let ObjData {
                scale_obj,
                scale_wb2,
                ..
            } = params;
            if let Some(grad) = gradient {
                if self.is_grad_impl_available() {
                    let grd = self
                        .grad_infill_eval(x, obj_model, *f_min, *scale_obj, *scale_wb2)
                        .to_vec();
                    grad[..].copy_from_slice(&grd);
                } else {
                    let f = |x: &Vec<f64>| -> f64 {
                        self.infill_eval(x, obj_model, *f_min, *scale_obj, *scale_wb2)
                    };
                    grad[..].copy_from_slice(&x.to_vec().central_diff(&f));
                }
            }
            self.infill_eval(x, obj_model, *f_min, *scale_obj, *scale_wb2)
        };

        let mut cstrs: Vec<Box<dyn nlopt::ObjFn<ObjData<f64>>>> = Vec::with_capacity(self.n_cstr);
        for i in 0..self.n_cstr {
            let index = i;
            let cstr =
                move |x: &[f64], gradient: Option<&mut [f64]>, params: &mut ObjData<f64>| -> f64 {
                    if let Some(grad) = gradient {
                        if self.is_grad_impl_available() {
                            let grd = cstr_models[i]
                                .predict_derivatives(
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
                                    .predict_values(
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
                        .predict_values(
                            &Array::from_shape_vec((1, x.len()), x.to_vec())
                                .unwrap()
                                .view(),
                        )
                        .unwrap()[[0, 0]]
                        / params.scale_cstr[index]
                };
            cstrs.push(Box::new(cstr) as Box<dyn nlopt::ObjFn<ObjData<f64>>>);
        }

        let mut success = false;
        let mut n_optim = 1;
        let n_max_optim = 20;
        let mut best_x = None;

        let scaling_points = sampling.sample(100 * self.xlimits.nrows());
        let scale_obj = compute_obj_scale(&scaling_points.view(), obj_model);
        info!("Acquisition function scaling is updated to {}", scale_obj);
        let scale_cstr = if cstr_models.is_empty() {
            Array1::zeros((0,))
        } else {
            let scale_cstr = compute_cstr_scales(&scaling_points.view(), cstr_models);
            info!("Feasibility criterion scaling is updated to {}", scale_cstr);
            scale_cstr
        };
        let scale_wb2 = if self.infill == InfillStrategy::WB2S {
            let scale = compute_wb2s_scale(&scaling_points.view(), obj_model, *f_min);
            info!("WB2S scaling factor is updated to {}", scale);
            scale
        } else {
            1.
        };

        let algorithm = match self.infill_optimizer {
            InfillOptimizer::Slsqp => Algorithm::Slsqp,
            InfillOptimizer::Cobyla => Algorithm::Cobyla,
        };
        while !success && n_optim <= n_max_optim {
            if let Some(seed) = lhs_optim_seed {
                let obj_data = ObjData {
                    scale_obj,
                    scale_wb2,
                    scale_cstr: scale_cstr.to_owned(),
                };
                let cstr_refs = cstrs.iter().map(|c| c.as_ref()).collect();
                let x_opt = LhsOptimizer::new(&self.xlimits, &obj, cstr_refs, &obj_data)
                    .with_rng(Xoshiro256Plus::seed_from_u64(seed))
                    .minimize();
                info!("LHS optimization best_x {}", x_opt);
                best_x = Some(x_opt);
                success = true;
            } else {
                let mut optimizer = Nlopt::new(
                    algorithm,
                    x_data.ncols(),
                    obj,
                    Target::Minimize,
                    ObjData {
                        scale_obj,
                        scale_wb2,
                        scale_cstr: scale_cstr.to_owned(),
                    },
                );
                let lower = self.xlimits.column(0).to_owned();
                optimizer.set_lower_bounds(lower.as_slice().unwrap())?;
                let upper = self.xlimits.column(1).to_owned();
                optimizer.set_upper_bounds(upper.as_slice().unwrap())?;
                optimizer.set_maxeval(200)?;
                optimizer.set_ftol_rel(1e-4)?;
                optimizer.set_ftol_abs(1e-4)?;
                cstrs.iter().enumerate().for_each(|(i, cstr)| {
                    optimizer
                        .add_inequality_constraint(
                            cstr,
                            ObjData {
                                scale_obj,
                                scale_wb2,
                                scale_cstr: scale_cstr.to_owned(),
                            },
                            1e-6 / scale_cstr[i],
                        )
                        .unwrap();
                });

                let mut best_opt = f64::INFINITY;
                let x_start = sampling.sample(self.n_start);

                for i in 0..self.n_start {
                    let mut x_opt = x_start.row(i).to_vec();
                    match optimizer.optimize(&mut x_opt) {
                        Ok((_, opt)) => {
                            if opt < best_opt {
                                best_opt = opt;
                                let res = x_opt.to_vec();
                                best_x = Some(Array::from(res));
                                success = true;
                            }
                        }
                        Err((err, code)) => {
                            debug!("Nlopt Err: {:?} (y_opt={})", err, code);
                        }
                    }
                }
            }
            n_optim += 1;
        }
        best_x.ok_or_else(|| EgoError::EgoError(String::from("Can not find best point")))
    }

    fn find_best_result_index(&self, y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> usize {
        find_best_result_index(y_data, self.cstr_tol)
    }

    fn get_virtual_point(
        &self,
        xk: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        obj_model: &dyn ClusteredSurrogate,
        cstr_models: &[Box<dyn ClusteredSurrogate>],
    ) -> Result<Vec<f64>> {
        let mut res: Vec<f64> = Vec::with_capacity(3);
        if self.q_ei == QEiStrategy::ConstantLiarMinimum {
            let index_min = y_data.slice(s![.., 0_usize]).argmin().unwrap();
            res.push(y_data[[index_min, 0]]);
            for ic in 1..=self.n_cstr {
                res.push(y_data[[index_min, ic]]);
            }
            Ok(res)
        } else {
            let x = &xk.view().insert_axis(Axis(0));
            let pred = obj_model.predict_values(x)?[[0, 0]];
            let var = obj_model.predict_variances(x)?[[0, 0]];
            let conf = match self.q_ei {
                QEiStrategy::KrigingBeliever => 0.,
                QEiStrategy::KrigingBelieverLowerBound => -3.,
                QEiStrategy::KrigingBelieverUpperBound => 3.,
                _ => -1., // never used
            };
            res.push(pred + conf * f64::sqrt(var));
            for cstr_model in cstr_models {
                res.push(cstr_model.predict_values(x)?[[0, 0]]);
            }
            Ok(res)
        }
    }

    fn infill_eval(
        &self,
        x: &[f64],
        obj_model: &dyn ClusteredSurrogate,
        f_min: f64,
        scale: f64,
        scale_wb2: f64,
    ) -> f64 {
        let x_f = x.to_vec();
        let obj = match self.infill {
            InfillStrategy::EI => -ei(&x_f, obj_model, f_min),
            InfillStrategy::WB2 => -wb2s(&x_f, obj_model, f_min, 1.),
            InfillStrategy::WB2S => -wb2s(&x_f, obj_model, f_min, scale_wb2),
        };
        obj / scale
    }

    pub fn grad_infill_eval(
        &self,
        x: &[f64],
        obj_model: &dyn ClusteredSurrogate,
        f_min: f64,
        scale: f64,
        scale_wb2: f64,
    ) -> Vec<f64> {
        let x_f = x.to_vec();
        let grad = match self.infill {
            InfillStrategy::EI => -grad_ei(&x_f, obj_model, f_min),
            InfillStrategy::WB2 => -grad_wbs2(&x_f, obj_model, f_min, 1.),
            InfillStrategy::WB2S => -grad_wbs2(&x_f, obj_model, f_min, scale_wb2),
        };
        (grad / scale).to_vec()
    }

    fn eval<O: CostFunction<Param = Array2<f64>, Output = Array2<f64>>>(
        &self,
        pb: &mut Problem<O>,
        x: &Array2<f64>,
    ) -> Array2<f64> {
        let params = if let Some(xtypes) = &self.xtypes {
            let fold = fold_with_enum_index(xtypes, &x.view());
            cast_to_discrete_values(xtypes, &fold)
        } else {
            x.to_owned()
        };
        pb.problem("cost_count", |problem| problem.cost(&params))
            .expect("Objective evaluation")
    }
}

/// EGO optimizer builder allowing to specify function to be minimized
/// subject to constraints intended to be negative.
///
pub struct EgorBuilder<O: GroupFunc> {
    fobj: O,
    seed: Option<u64>,
}

impl<O: GroupFunc> EgorBuilder<O> {
    /// Function to be minimized domain should be basically R^nx -> R^ny
    /// where nx is the dimension of input x and ny the output dimension
    /// equal to 1 (obj) + n (cstrs).
    /// But function has to be able to evaluate several points in one go
    /// hence take an (p, nx) matrix and return an (p, ny) matrix
    pub fn optimize(fobj: O) -> Self {
        EgorBuilder { fobj, seed: None }
    }

    /// Allow to specify a seed for random number generator to allow
    /// reproducible runs.
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build an Egor optimizer to minimize the function within
    /// the continuous xlimits specified as [[lower, upper], ...]
    /// number of rows gives the dimension of the inputs (continuous optimization).
    pub fn min_within(
        self,
        xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Egor<O, MoeParams<f64, Xoshiro256Plus>> {
        let rng = if let Some(seed) = self.seed {
            Xoshiro256Plus::seed_from_u64(seed)
        } else {
            Xoshiro256Plus::from_entropy()
        };
        Egor {
            fobj: ObjFunc::new(self.fobj),
            solver: EgorSolver::new(xlimits, rng),
        }
    }

    /// Build an Egor optimizer to minimize the function R^n -> R^p taking
    /// inputs specified with given xtypes where some of components may be
    /// discrete variables (mixed-integer optimization).
    pub fn min_within_mixed_space(self, xtypes: &[Xtype]) -> Egor<O, MixintMoeParams> {
        let rng = if let Some(seed) = self.seed {
            Xoshiro256Plus::seed_from_u64(seed)
        } else {
            Xoshiro256Plus::from_entropy()
        };
        Egor {
            fobj: ObjFunc::new(self.fobj),
            solver: EgorSolver::new_with_xtypes(xtypes, rng),
        }
    }
}

/// Egor optimizer structure used to parameterize the underlying `argmin::Solver`
/// and trigger the optimization using `argmin::Executor`.
#[derive(Clone)]
pub struct Egor<O: GroupFunc, SB: SurrogateBuilder> {
    fobj: ObjFunc<O>,
    solver: EgorSolver<SB>,
}

impl<O: GroupFunc, SB: SurrogateBuilder> Egor<O, SB> {
    /// Sets allowed number of evaluation of the function under optimization
    pub fn n_eval(&mut self, n_eval: usize) -> &mut Self {
        self.solver.n_eval(n_eval);
        self
    }

    /// Sets the number of runs of infill strategy optimizations (best result taken)
    pub fn n_start(&mut self, n_start: usize) -> &mut Self {
        self.solver.n_start(n_start);
        self
    }

    /// Sets Number of parallel evaluations of the function under optimization
    pub fn q_parallel(&mut self, q_parallel: usize) -> &mut Self {
        self.solver.q_parallel(q_parallel);
        self
    }

    /// Number of samples of initial LHS sampling (used when DOE not provided by the user)
    ///
    /// When 0 a number of points is computed automatically regarding the number of input variables
    /// of the function under optimization.
    pub fn n_doe(&mut self, n_doe: usize) -> &mut Self {
        self.solver.n_doe(n_doe);
        self
    }

    /// Sets the number of constraint functions
    pub fn n_cstr(&mut self, n_cstr: usize) -> &mut Self {
        self.solver.n_cstr(n_cstr);
        self
    }

    /// Sets the tolerance on constraints violation (cstr < tol)
    pub fn cstr_tol(&mut self, tol: f64) -> &mut Self {
        self.solver.cstr_tol(tol);
        self
    }

    /// Sets an initial DOE containing ns samples
    ///
    /// Either nt = nx then only x are specified and ns evals are done to get y doe values,
    /// or nt = nx + ny then x = doe(:, :nx) and y = doe(:, nx:) are specified
    pub fn doe(&mut self, doe: Option<Array2<f64>>) -> &mut Self {
        self.solver.doe(doe);
        self
    }

    /// Sets the parallel infill strategy
    ///
    /// Parallel infill criteria to get virtual next promising points in order to allow
    /// n parallel evaluations of the function under optimization.
    pub fn qei_strategy(&mut self, q_ei: QEiStrategy) -> &mut Self {
        self.solver.qei_strategy(q_ei);
        self
    }

    /// Sets the nfill criteria
    pub fn infill_strategy(&mut self, infill: InfillStrategy) -> &mut Self {
        self.solver.infill_strategy(infill);
        self
    }

    /// Sets the infill optimizer
    pub fn infill_optimizer(&mut self, optimizer: InfillOptimizer) -> &mut Self {
        self.solver.infill_optimizer(optimizer);
        self
    }

    /// Sets the allowed regression models used in gaussian processes.
    pub fn regression_spec(&mut self, regression_spec: RegressionSpec) -> &mut Self {
        self.solver.regression_spec(regression_spec);
        self
    }

    /// Sets the allowed correlation models used in gaussian processes.
    pub fn correlation_spec(&mut self, correlation_spec: CorrelationSpec) -> &mut Self {
        self.solver.correlation_spec(correlation_spec);
        self
    }

    /// Sets the number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
    ///
    /// This is used to address high-dimensional problems typically when nx > 9.
    pub fn kpls_dim(&mut self, kpls_dim: Option<usize>) -> &mut Self {
        self.solver.kpls_dim(kpls_dim);
        self
    }

    /// Sets the number of clusters used by the mixture of surrogate experts.
    ///
    /// When set to 0, the number of clusters is determined automatically
    pub fn n_clusters(&mut self, n_clusters: Option<usize>) -> &mut Self {
        self.solver.n_clusters(n_clusters);
        self
    }

    /// Sets a known target minimum to be used as a stopping criterion.
    pub fn target(&mut self, target: f64) -> &mut Self {
        self.solver.target(target);
        self
    }

    /// Sets a directory to write optimization history and used as search path for hot start doe
    pub fn outdir(&mut self, outdir: Option<String>) -> &mut Self {
        self.solver.outdir(outdir);
        self
    }

    /// Whether we start by loading last DOE saved in `outdir` as initial DOE
    pub fn hot_start(&mut self, hot_start: bool) -> &mut Self {
        self.solver.hot_start(hot_start);
        self
    }

    /// Given an evaluated doe (x, y) data, return the next promising x point
    /// where optimum may occurs regarding the infill criterium.
    /// This function inverse the control of the optimization and can used
    /// ask-and-tell interface to the EGO optmizer.
    pub fn suggest(
        &self,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Array2<f64> {
        self.solver.suggest(x_data, y_data)
    }

    /// Runs the (constrained) optimization of the objective function.
    pub fn run(&self) -> Result<OptimResult<f64>> {
        let no_discrete = self.solver.no_discrete;
        let xtypes = self.solver.xtypes.clone();

        let result = Executor::new(self.fobj.clone(), self.solver.clone()).run()?;
        debug!("ARGMIN result = {}", result);
        let (x_data, y_data) = result.state().clone().take_data().unwrap();

        let res = if no_discrete {
            info!("History: \n{}", concatenate![Axis(1), x_data, y_data]);
            OptimResult {
                x_opt: result.state.get_best_param().unwrap().to_owned(),
                y_opt: result.state.get_full_best_cost().unwrap().to_owned(),
            }
        } else {
            let xtypes = xtypes.unwrap(); // !no_discrete
            let x_data = cast_to_discrete_values(&xtypes, &x_data);
            let x_data = fold_with_enum_index(&xtypes, &x_data.view());
            info!("History: \n{}", concatenate![Axis(1), x_data, y_data]);

            let x_opt = result
                .state
                .get_best_param()
                .unwrap()
                .to_owned()
                .insert_axis(Axis(0));
            let x_opt = cast_to_discrete_values(&xtypes, &x_opt);
            let x_opt = fold_with_enum_index(&xtypes, &x_opt.view());
            OptimResult {
                x_opt: x_opt.row(0).to_owned(),
                y_opt: result.state.get_full_best_cost().unwrap().to_owned(),
            }
        };
        info!("Optim Result: min f(x)={} at x={}", res.y_opt, res.x_opt);

        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use argmin_testfunctions::rosenbrock;
    use ndarray::{array, ArrayView2};
    use ndarray_npy::read_npy;
    use serial_test::serial;
    use std::time::Instant;

    #[cfg(not(feature = "blas"))]
    use linfa_linalg::norm::*;
    #[cfg(feature = "blas")]
    use ndarray_linalg::Norm;

    fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
        (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
    }

    #[test]
    #[serial]
    fn test_xsinx_ei_quadratic_egor_solver() {
        let initial_doe = array![[0.], [7.], [25.]];
        let res = EgorBuilder::optimize(xsinx)
            .min_within(&array![[0.0, 25.0]])
            .infill_strategy(InfillStrategy::EI)
            .regression_spec(RegressionSpec::QUADRATIC)
            .correlation_spec(CorrelationSpec::ALL)
            .n_eval(30)
            .doe(Some(initial_doe.to_owned()))
            .target(-15.1)
            .outdir(Some("target/tests".to_string()))
            .run()
            .expect("Egor should minimize xsinx");
        let expected = array![-15.1];
        assert_abs_diff_eq!(expected, res.y_opt, epsilon = 0.5);
        let saved_doe: Array2<f64> =
            read_npy(format!("target/tests/{}", DOE_INITIAL_FILE)).unwrap();
        assert_abs_diff_eq!(initial_doe, saved_doe.slice(s![..3, ..1]), epsilon = 1e-6);
    }

    #[test]
    #[serial]
    fn test_xsinx_wb2_egor_solver() {
        let res = EgorBuilder::optimize(xsinx)
            .min_within(&array![[0.0, 25.0]])
            .n_eval(20)
            .regression_spec(RegressionSpec::ALL)
            .correlation_spec(CorrelationSpec::ALL)
            .run()
            .expect("Egor should minimize");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    #[test]
    #[serial]
    fn test_xsinx_auto_clustering_egor_solver() {
        let res = EgorBuilder::optimize(xsinx)
            .min_within(&array![[0.0, 25.0]])
            .n_clusters(Some(0))
            .n_eval(20)
            .run()
            .expect("Egor with auto clustering should minimize xsinx");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    #[test]
    #[serial]
    fn test_xsinx_with_hotstart_egor_solver() {
        let xlimits = array![[0.0, 25.0]];
        let doe = Lhs::new(&xlimits).sample(10);
        let res = EgorBuilder::optimize(xsinx)
            .random_seed(42)
            .min_within(&xlimits)
            .n_eval(15)
            .doe(Some(doe))
            .outdir(Some("target/tests".to_string()))
            .run()
            .expect("Minimize failure");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);

        let res = EgorBuilder::optimize(xsinx)
            .random_seed(42)
            .min_within(&xlimits)
            .n_eval(5)
            .outdir(Some("target/tests".to_string()))
            .hot_start(true)
            .run()
            .expect("Egor should minimize xsinx");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    #[test]
    #[serial]
    fn test_xsinx_suggestions_egor_solver() {
        let mut ego = EgorBuilder::optimize(xsinx)
            .random_seed(42)
            .min_within(&array![[0., 25.]]);
        let ego = ego
            .regression_spec(RegressionSpec::ALL)
            .correlation_spec(CorrelationSpec::ALL)
            .infill_strategy(InfillStrategy::EI);

        let mut doe = array![[0.], [7.], [20.], [25.]];
        let mut y_doe = xsinx(&doe.view());
        for _i in 0..10 {
            let x_suggested = ego.suggest(&doe, &y_doe);

            doe = concatenate![Axis(0), doe, x_suggested];
            y_doe = xsinx(&doe.view());
        }

        let expected = -15.1;
        let y_opt = y_doe.min().unwrap();
        assert_abs_diff_eq!(expected, *y_opt, epsilon = 1e-1);
    }

    fn rosenb(x: &ArrayView2<f64>) -> Array2<f64> {
        let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
        Zip::from(y.rows_mut())
            .and(x.rows())
            .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec(), 1., 100.)]));
        y
    }

    #[test]
    #[serial]
    fn test_rosenbrock_2d_egor_solver() {
        let now = Instant::now();
        let xlimits = array![[-2., 2.], [-2., 2.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(10);
        let res = EgorBuilder::optimize(rosenb)
            .random_seed(42)
            .min_within(&xlimits)
            .doe(Some(doe))
            .n_eval(100)
            .regression_spec(RegressionSpec::ALL)
            .correlation_spec(CorrelationSpec::ALL)
            .target(1e-2)
            .run()
            .expect("Minimize failure");
        println!("Rosenbrock optim result = {:?}", res);
        println!("Elapsed = {:?}", now.elapsed());
        let expected = array![1., 1.];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 5e-1);
    }

    // Objective
    fn g24(x: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> f64 {
        // Function G24: 1 global optimum y_opt = -5.5080 at x_opt =(2.3295, 3.1785)
        -x[0] - x[1]
    }

    // Constraints < 0
    fn g24_c1(x: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> f64 {
        -2.0 * x[0].powf(4.0) + 8.0 * x[0].powf(3.0) - 8.0 * x[0].powf(2.0) + x[1] - 2.0
    }

    fn g24_c2(x: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> f64 {
        -4.0 * x[0].powf(4.0) + 32.0 * x[0].powf(3.0) - 88.0 * x[0].powf(2.0) + 96.0 * x[0] + x[1]
            - 36.0
    }

    fn f_g24(x: &ArrayView2<f64>) -> Array2<f64> {
        let mut y = Array2::zeros((x.nrows(), 3));
        Zip::from(y.rows_mut())
            .and(x.rows())
            .for_each(|mut yi, xi| {
                yi.assign(&array![g24(&xi), g24_c1(&xi), g24_c2(&xi)]);
            });
        y
    }

    #[test]
    #[serial]
    fn test_egor_g24_basic_egor_solver() {
        let xlimits = array![[0., 3.], [0., 4.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(3);
        let res = EgorBuilder::optimize(f_g24)
            .random_seed(42)
            .min_within(&xlimits)
            .n_cstr(2)
            .doe(Some(doe))
            .n_eval(20)
            .run()
            .expect("Minimize failure");
        println!("G24 optim result = {:?}", res);
        let expected = array![2.3295, 3.1785];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 2e-2);
    }

    #[test]
    fn test_egor_g24_qei_egor_solver() {
        let xlimits = array![[0., 3.], [0., 4.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(10);
        let res = EgorBuilder::optimize(f_g24)
            .random_seed(42)
            .min_within(&xlimits)
            .regression_spec(RegressionSpec::ALL)
            .correlation_spec(CorrelationSpec::ALL)
            .n_cstr(2)
            .q_parallel(2)
            .qei_strategy(QEiStrategy::KrigingBeliever)
            .doe(Some(doe))
            .target(-5.508013)
            .n_eval(30)
            .run()
            .expect("Egor minimization");
        println!("G24 optim result = {:?}", res);
        let expected = array![2.3295, 3.1785];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 2e-2);
    }

    // Mixed-integer tests

    fn mixsinx(x: &ArrayView2<f64>) -> Array2<f64> {
        if (x.mapv(|v| v.round()).norm_l2() - x.norm_l2()).abs() < 1e-6 {
            (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
        } else {
            panic!("Error: mixsinx works only on integer, got {:?}", x)
        }
    }

    #[test]
    #[serial]
    fn test_mixsinx_ei_mixint_egor_solver() {
        let n_eval = 30;
        let doe = array![[0.], [7.], [25.]];
        let xtypes = vec![Xtype::Int(0, 25)];

        let res = EgorBuilder::optimize(mixsinx)
            .random_seed(42)
            .min_within_mixed_space(&xtypes)
            .doe(Some(doe))
            .n_eval(n_eval)
            .target(-15.1)
            .infill_strategy(InfillStrategy::EI)
            .run()
            .unwrap();
        assert_abs_diff_eq!(array![18.], res.x_opt, epsilon = 2.);
    }

    #[test]
    fn test_mixsinx_reclustering_mixint_egor_solver() {
        let n_eval = 30;
        let doe = array![[0.], [7.], [25.]];
        let xtypes = vec![Xtype::Int(0, 25)];

        let res = EgorBuilder::optimize(mixsinx)
            .random_seed(42)
            .min_within_mixed_space(&xtypes)
            .doe(Some(doe))
            .n_eval(n_eval)
            .target(-15.1)
            .infill_strategy(InfillStrategy::EI)
            .run()
            .unwrap();
        assert_abs_diff_eq!(array![18.], res.x_opt, epsilon = 2.);
    }

    #[test]
    #[serial]
    fn test_mixsinx_wb2_mixint_egor_solver() {
        let n_eval = 30;
        let xtypes = vec![Xtype::Int(0, 25)];

        let res = EgorBuilder::optimize(mixsinx)
            .random_seed(42)
            .min_within_mixed_space(&xtypes)
            .regression_spec(egobox_moe::RegressionSpec::CONSTANT)
            .correlation_spec(egobox_moe::CorrelationSpec::SQUAREDEXPONENTIAL)
            .n_eval(n_eval)
            .infill_strategy(InfillStrategy::WB2)
            .run()
            .unwrap();
        assert_abs_diff_eq!(&array![18.], &res.x_opt, epsilon = 3.);
    }

    #[test]
    fn test_unfold_xtypes_as_continuous_limits() {
        let xtypes = vec![Xtype::Int(0, 25)];
        let xlimits = unfold_xtypes_as_continuous_limits(&xtypes);
        let expected = array![[0., 25.]];
        assert_abs_diff_eq!(expected, xlimits);
    }
}
