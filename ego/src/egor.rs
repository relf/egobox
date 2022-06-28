//! Egor optimizer implements EGO algorithm with basic handling of constraints.
//!
//! ```no_run
//! # use ndarray::{array, Array2, ArrayView1, ArrayView2, Zip};
//! # use egobox_doe::{Lhs, SamplingMethod};
//! # use egobox_ego::{ApproxValue, Egor, InfillStrategy, InfillOptimizer};
//! # use rand_isaac::Isaac64Rng;
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
//! let res = Egor::new(rosenb, &xlimits)
//!     .infill_strategy(InfillStrategy::EI)
//!     .n_doe(10)
//!     .expect(Some(ApproxValue {  // Known solution, the algo exits if reached
//!         value: 0.0,
//!         tolerance: 1e-1,
//!     }))
//!     .n_eval(30)
//!     .minimize()
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
//! # use egobox_ego::{ApproxValue, Egor, InfillStrategy, InfillOptimizer};
//! # use rand_isaac::Isaac64Rng;
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
//! let res = Egor::new(f_g24, &xlimits)
//!            .n_cstr(2)
//!            .infill_strategy(InfillStrategy::EI)
//!            .infill_optimizer(InfillOptimizer::Cobyla)
//!            .doe(Some(doe))
//!            .n_eval(40)
//!            .expect(Some(ApproxValue {  
//!               value: -5.5080,
//!               tolerance: 1e-3,
//!            }))
//!            .minimize()
//!            .expect("g24 minimized");
//! println!("G24 min result = {:?}", res);
//! ```
//!
use crate::errors::{EgoError, Result};
use crate::lhs_optimizer::LhsOptimizer;
use crate::sort_axis::*;
use crate::types::*;
use crate::utils::update_data;
use crate::utils::{compute_cstr_scales, compute_obj_scale, compute_wb2s_scale};
use crate::utils::{ei, wb2s};

use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use egobox_moe::{ClusteredSurrogate, Clustering, CorrelationSpec, Moe, MoeParams, RegressionSpec};
use env_logger::{Builder, Env};
use finitediff::FiniteDiff;
use linfa::ParamGuard;
use log::{debug, info, warn};
use ndarray::{concatenate, s, Array, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip};
use ndarray_npy::{read_npy, write_npy};
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_stats::QuantileExt;
use nlopt::*;
use rand_isaac::Isaac64Rng;

const DOE_INITIAL_FILE: &str = "egor_initial_doe.npy";
const DOE_FILE: &str = "egor_doe.npy";

impl<R: Rng + SeedableRng + Clone> SurrogateBuilder for MoeParams<f64, R> {
    fn train(&self, xt: &Array2<f64>, yt: &Array2<f64>) -> Result<Box<dyn ClusteredSurrogate>> {
        let checked = self.check_ref()?;
        let moe = checked.train(xt, yt)?;
        Ok(moe).map(|moe| Box::new(moe) as Box<dyn ClusteredSurrogate>)
    }

    fn train_on_clusters(
        &self,
        xt: &Array2<f64>,
        yt: &Array2<f64>,
        clustering: &Clustering,
    ) -> Result<Box<dyn ClusteredSurrogate>> {
        let checked = self.check_ref()?;
        let moe = checked.train_on_clusters(xt, yt, clustering)?;
        Ok(moe).map(|moe| Box::new(moe) as Box<dyn ClusteredSurrogate>)
    }
}

/// EGO optimization parameterization
#[derive(Clone)]
pub struct Egor<'a, O: GroupFunc, R: Rng> {
    /// Number of function evaluations allocated to find the optimum
    /// Note: if the initial doe has to be evaluated, doe size should be substract.  
    pub n_eval: usize,
    /// Number of starts for multistart approach used for hyperparameters optimization
    pub n_start: usize,
    /// Number of parallel points evaluated for each "function evaluation"
    pub q_parallel: usize,
    /// Number of initial doe drawn using Latin hypercube sampling
    /// Note: n_doe > 0; otherwise n_doe = max(xdim+1, 5)
    pub n_doe: usize,
    /// Number of Constraints
    /// Note: dim function ouput = 1 objective + n_cstr constraints
    pub n_cstr: usize,
    /// Constraints violation tolerance meaning cstr < cstr_tol is considered valid
    pub cstr_tol: f64,
    /// Initial doe can be either \[x\] with x inputs only or an evaluated doe \[x, y\]
    /// Note: x dimension is determined using xlimits
    pub doe: Option<Array2<f64>>,
    /// Matrix (nx, 2) of [lower bound, upper bound] of the nx components of x
    pub xlimits: Array2<f64>,
    /// Parallel strategy used to define several points (q_parallel) evaluations at each iteration
    pub q_ei: QEiStrategy,
    /// Criterium to select next point to evaluate
    pub infill: InfillStrategy,
    /// The optimizer used to optimize infill criterium
    pub infill_optimizer: InfillOptimizer,
    /// Regression specification for GP models used by mixture of experts (see [egobox_moe])
    pub regression_spec: RegressionSpec,
    /// Correlation specification for GP models used by mixture of experts (see [egobox_moe])
    pub correlation_spec: CorrelationSpec,
    /// Optional dimension reduction (see [egobox_moe])
    pub kpls_dim: Option<usize>,
    /// Number of clusters used by mixture of experts (see [egobox_moe])
    /// When set to 0 the clusters are computes automatically and refreshed
    /// every 10-points (tentative) additions
    pub n_clusters: Option<usize>,
    /// Specification of an expected solution which is used to stop the algorithm once reached
    pub expected: Option<ApproxValue>,
    /// Directory to save intermediate results: inital doe + evalutions at each iteration
    pub outdir: Option<String>,
    /// If true use <outdir> to retrieve and start from previous results
    pub hot_start: bool,
    /// An optional surrogate builder used to model objective and constraint
    /// functions, otherwise [mixture of expert](egobox_moe) is used
    /// Note: if specified takes precedence over individual settings
    pub surrogate_builder: Option<&'a dyn SurrogateBuilder>,
    /// An optional pre-processor to preprocess continuous input given
    /// to the function under optimization, specially used with mixed integer
    /// function optimization
    pub pre_proc: Option<&'a dyn PreProcessor>,
    /// The function under optimization f(x) = [objective, cstr1, ..., cstrn], (n_cstr+1 size)
    pub obj: O,
    /// A random generator used to get reproductible results.
    /// For instance: Isaac64Rng::from_u64_seed(42) for reproducibility
    pub rng: R,
}

impl<'a, O: GroupFunc> Egor<'a, O, Isaac64Rng> {
    /// Constructor of the optimization of the function `f`
    ///
    /// The function `f` shoud return an objective but also constraint values if any.
    /// Design space is specified by the 2D array `xlimits` which is `[nx, 2]`-shaped and
    /// constains lower and upper bounds of `x` components.
    pub fn new(f: O, xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Egor<'a, O, Isaac64Rng> {
        Self::new_with_rng(f, xlimits, Isaac64Rng::from_entropy())
    }
}

impl<'a, O: GroupFunc, R: Rng + Clone> Egor<'a, O, R> {
    /// Constructor of the optimization of the function `f` with specified random generator
    /// to get reproducibility.
    ///
    /// See [`Egor::new()`]
    pub fn new_with_rng(f: O, xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>, rng: R) -> Self {
        let env = Env::new().filter_or("EGOBOX_LOG", "info");
        let mut builder = Builder::from_env(env);
        let builder = builder.target(env_logger::Target::Stdout);
        builder.try_init().ok();
        Egor {
            n_eval: 20,
            n_start: 20,
            q_parallel: 1,
            n_doe: 0,
            n_cstr: 0,
            cstr_tol: 1e-6,
            doe: None,
            xlimits: xlimits.to_owned(),
            q_ei: QEiStrategy::KrigingBeliever,
            infill: InfillStrategy::WB2,
            infill_optimizer: InfillOptimizer::Slsqp,
            regression_spec: RegressionSpec::ALL,
            correlation_spec: CorrelationSpec::ALL,
            kpls_dim: None,
            n_clusters: Some(1),
            expected: None,
            outdir: None,
            hot_start: false,
            surrogate_builder: None,
            pre_proc: None,
            obj: f,
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

    /// Sets a known minimum to be used as a stopping criterion.
    pub fn expect(&mut self, expected: Option<ApproxValue>) -> &mut Self {
        self.expected = expected;
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

    /// Sets the surrogate builder used to model objective and constraints.
    ///
    /// If none Mixture of Experts [MoeParams] will be used.
    pub fn surrogate_builder(
        &mut self,
        surrogate_builder: Option<&'a dyn SurrogateBuilder>,
    ) -> &mut Self {
        self.surrogate_builder = surrogate_builder;
        self
    }

    /// Sets a pre processor for inputs before being evaluated by the function under optimization
    ///
    /// Used for mixed-integer optimizatio. See [crate::MixintEgor]
    pub fn pre_proc(&mut self, pre_proc: Option<&'a dyn PreProcessor>) -> &mut Self {
        self.pre_proc = pre_proc;
        self
    }

    /// Sets a random generator for reproducibility
    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> Egor<'a, O, R2> {
        Egor {
            n_eval: self.n_eval,
            n_start: self.n_start,
            q_parallel: self.q_parallel,
            n_doe: self.n_doe,
            n_cstr: self.n_cstr,
            cstr_tol: self.cstr_tol,
            doe: self.doe,
            xlimits: self.xlimits,
            q_ei: self.q_ei,
            infill: self.infill,
            infill_optimizer: self.infill_optimizer,
            regression_spec: self.regression_spec,
            correlation_spec: self.correlation_spec,
            kpls_dim: self.kpls_dim,
            n_clusters: self.n_clusters,
            expected: self.expected,
            outdir: self.outdir,
            hot_start: self.hot_start,
            surrogate_builder: self.surrogate_builder,
            pre_proc: self.pre_proc,
            obj: self.obj,
            rng,
        }
    }

    /// Given an evaluated doe (x, y) data, return the next promising x point
    /// where optimum may occurs regarding the infill criterium.
    /// This function inverse the control of the optimization and can used
    /// ask-and-tell interface to the EGO optmizer.
    pub fn suggest(&self, x_data: &Array2<f64>, y_data: &Array2<f64>) -> Array2<f64> {
        let rng = self.rng.clone();
        let sampling = Lhs::new(&self.xlimits).with_rng(rng).kind(LhsKind::Maximin);
        let mut clusterings = vec![None; 1 + self.n_cstr];
        let (x_dat, _) = self.next_points(1, &mut clusterings, x_data, y_data, &sampling, false);
        x_dat
    }

    /// Minimize using EGO algorithm
    pub fn minimize(&self) -> Result<OptimResult<f64>> {
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

        let (mut y_data, mut x_data, n_iter) = if let Some(doe) = doe {
            if doe.ncols() == self.xlimits.nrows() {
                // only x are specified
                info!("Compute initial DOE on specified {} points", doe.nrows());
                (
                    self.eval(doe),
                    doe.to_owned(),
                    self.n_eval.saturating_sub(doe.nrows()),
                )
            } else {
                // split doe in x and y
                info!("Use specified DOE {} samples", doe.nrows());
                (
                    doe.slice(s![.., self.xlimits.nrows()..]).to_owned(),
                    doe.slice(s![.., ..self.xlimits.nrows()]).to_owned(),
                    self.n_eval.saturating_sub(doe.nrows()),
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
            (self.eval(&x), x, self.n_eval - n_doe)
        };
        let doe = concatenate![Axis(1), x_data, y_data];
        if self.outdir.is_some() {
            let path = self.outdir.as_ref().unwrap();
            std::fs::create_dir_all(path)?;
            let filepath = std::path::Path::new(path).join(DOE_INITIAL_FILE);
            info!("Save initial doe in {:?}", filepath);
            write_npy(filepath, &doe).expect("Write initial doe");
        }

        const MAX_RETRY: i32 = 3;
        let mut clusterings = vec![None; self.n_cstr + 1];
        let mut no_point_added_retries = MAX_RETRY;
        let mut lhs_optim = false;
        if n_iter / self.q_parallel == 0 {
            warn!("Number of evaluations {} too low, incompatible with initial doe size {} and q_parallel={} evals/iter", 
                  self.n_eval, doe.nrows(), self.q_parallel);
        }
        let n_iter = n_iter / self.q_parallel;

        for i in 1..=n_iter {
            let (x_dat, y_dat) =
                self.next_points(i, &mut clusterings, &x_data, &y_data, &sampling, lhs_optim);
            debug!("Try adding {}", x_dat);
            let rejected_count = update_data(&mut x_data, &mut y_data, &x_dat, &y_dat);
            if rejected_count > 0 {
                info!(
                    "Reject {}/{} point{} too close to previous ones",
                    rejected_count,
                    x_dat.nrows(),
                    if rejected_count > 1 { "s" } else { "" }
                );
            }
            info!("New pts: {}", x_dat);
            if rejected_count == x_dat.nrows() {
                no_point_added_retries -= 1;
                if no_point_added_retries == 0 {
                    info!("Max number of retries ({}) without adding point", MAX_RETRY);
                    info!("Use LHS optimization to get at least one point");
                    lhs_optim = true;
                }
                info!("End iteration {}/{}", i, n_iter);
                continue;
            }

            no_point_added_retries = MAX_RETRY;
            let count = (self.q_parallel - rejected_count) as i32;
            let x_to_eval = x_data.slice(s![-count.., ..]).to_owned();
            info!(
                "Add {} point{} {}:",
                count,
                if count > 1 { "s" } else { "" },
                if lhs_optim { " from sampling" } else { "" }
            );
            info!("  {}", x_dat);
            lhs_optim = false; // reset as a point is added

            let y_actual = self.eval(&x_to_eval);
            Zip::from(y_data.slice_mut(s![-count.., ..]).columns_mut())
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
                "End iteration {}/{}: Best fun(x)={} at x={}",
                i,
                n_iter,
                y_data.row(best_index),
                x_data.row(best_index)
            );
            if let Some(sol) = self.expected {
                if (y_data[[best_index, 0]] - sol.value).abs() < sol.tolerance {
                    info!("Expected optimum : {:?}", sol);
                    info!("Expected optimum reached!");
                    break;
                }
            }
        }
        let best_index = self.find_best_result_index(&y_data);
        info!("History: \n{}", concatenate![Axis(1), x_data, y_data]);
        let res = OptimResult {
            x_opt: x_data.row(best_index).to_owned(),
            y_opt: y_data.row(best_index).to_owned(),
        };
        info!("Optim Result: min f(x)={} at x= {}", res.y_opt, res.x_opt);
        Ok(res)
    }

    fn should_cluster(&self, n_iter: usize) -> bool {
        n_iter == 1 || {
            if let Some(nc) = self.n_clusters {
                nc == 0 && n_iter % 10 == 1
            } else {
                false
            }
        }
    }

    fn make_default_builder(&self) -> egobox_moe::MoeParams<f64, rand_isaac::Isaac64Rng> {
        let moe = Moe::params()
            .kpls_dim(self.kpls_dim)
            .regression_spec(self.regression_spec)
            .correlation_spec(self.correlation_spec);
        if let Some(nc) = self.n_clusters {
            if nc > 0 {
                moe.n_clusters(nc)
            } else {
                moe
            }
        } else {
            moe
        }
    }

    fn make_clustered_surrogate(
        &self,
        xt: &Array2<f64>,
        yt: &Array2<f64>,
        n_iter: usize,
        clustering: &Option<Clustering>,
    ) -> Box<dyn ClusteredSurrogate> {
        let default_builder = &self.make_default_builder() as &dyn SurrogateBuilder;
        let builder = self.surrogate_builder.unwrap_or(default_builder);
        if self.should_cluster(n_iter) {
            builder.train(xt, yt).expect("GP training failure")
        } else {
            let clustering = clustering.as_ref().unwrap().clone();
            builder
                .train_on_clusters(xt, yt, &clustering)
                .expect("GP training failure")
        }
    }

    fn next_points(
        &'a self,
        n_iter: usize,
        clusterings: &mut [Option<Clustering>],
        x_data: &Array2<f64>,
        y_data: &Array2<f64>,
        sampling: &Lhs<f64, R>,
        lhs_optim: bool,
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
                n_iter,
                &clusterings[0],
            );
            clusterings[0] = Some(obj_model.to_clustering());

            let mut cstr_models: Vec<Box<dyn ClusteredSurrogate>> = Vec::with_capacity(self.n_cstr);
            for k in 1..=self.n_cstr {
                if self.should_cluster(n_iter) {
                    info!("Constraint[{}] reclustering...", k)
                }
                let cstr_model = self.make_clustered_surrogate(
                    &xt,
                    &yt.slice(s![.., k..k + 1]).to_owned(),
                    n_iter,
                    &clusterings[k],
                );
                cstr_models.push(cstr_model);
                if self.should_cluster(n_iter) {
                    info!(
                        "... Best nb of clusters for constraint[{}]: {}",
                        k,
                        cstr_models[k - 1].n_clusters()
                    );
                }
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

    fn find_best_point(
        &self,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        sampling: &Lhs<f64, R>,
        obj_model: &dyn ClusteredSurrogate,
        cstr_models: &[Box<dyn ClusteredSurrogate>],
        lhs_optim: bool,
    ) -> Result<Array1<f64>> {
        let f_min = y_data.min().unwrap();

        let obj = |x: &[f64], gradient: Option<&mut [f64]>, params: &mut ObjData<f64>| -> f64 {
            let ObjData {
                scale_obj,
                scale_wb2,
                ..
            } = params;
            if let Some(grad) = gradient {
                let f = |x: &Vec<f64>| -> f64 {
                    self.infill_eval(x, obj_model, *f_min, *scale_obj, *scale_wb2)
                };
                grad[..].copy_from_slice(&x.to_vec().forward_diff(&f));
            }
            self.infill_eval(x, obj_model, *f_min, *scale_obj, *scale_wb2)
        };

        let mut cstrs: Vec<Box<dyn nlopt::ObjFn<ObjData<f64>>>> = Vec::with_capacity(self.n_cstr);
        for i in 0..self.n_cstr {
            let index = i;
            let cstr =
                move |x: &[f64], gradient: Option<&mut [f64]>, params: &mut ObjData<f64>| -> f64 {
                    if let Some(grad) = gradient {
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
                        grad[..].copy_from_slice(&x.to_vec().forward_diff(&f));
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
        let scale_cstr = compute_cstr_scales(&scaling_points.view(), cstr_models);
        let scale_wb2 = if self.infill == InfillStrategy::WB2S {
            compute_wb2s_scale(&scaling_points.view(), obj_model, *f_min)
        } else {
            1.
        };

        let algorithm = match self.infill_optimizer {
            InfillOptimizer::Slsqp => Algorithm::Slsqp,
            InfillOptimizer::Cobyla => Algorithm::Cobyla,
        };
        while !success && n_optim <= n_max_optim {
            if lhs_optim {
                let obj_data = ObjData {
                    scale_obj,
                    scale_wb2,
                    scale_cstr: scale_cstr.to_owned(),
                };
                let cstr_refs = cstrs.iter().map(|c| c.as_ref()).collect();
                let x_opt = LhsOptimizer::new(&self.xlimits, &obj, cstr_refs, &obj_data)
                    .with_rng(self.rng.clone())
                    .minimize();
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
        if self.n_cstr > 0 {
            let mut index = 0;
            let perm = y_data.sort_axis_by(Axis(0), |i, j| y_data[[i, 0]] < y_data[[j, 0]]);
            let y_sort = y_data.to_owned().permute_axis(Axis(0), &perm);
            for (i, row) in y_sort.axis_iter(Axis(0)).enumerate() {
                if !row.slice(s![1..]).iter().any(|v| *v > self.cstr_tol) {
                    index = i;
                    break;
                }
            }
            perm.indices[index]
        } else {
            y_data.column(0).argmin().unwrap()
        }
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

    fn eval(&self, x: &Array2<f64>) -> Array2<f64> {
        if let Some(pre_proc) = self.pre_proc {
            (self.obj)(&pre_proc.run(x).view())
        } else {
            (self.obj)(&x.view())
        }
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

    fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
        (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
    }

    #[test]
    #[serial]
    fn test_xsinx_ei_egor() {
        let initial_doe = array![[0.], [7.], [25.]];
        let res = Egor::new(xsinx, &array![[0.0, 25.0]])
            .infill_strategy(InfillStrategy::EI)
            .regression_spec(RegressionSpec::QUADRATIC)
            .correlation_spec(CorrelationSpec::ALL)
            .n_eval(20)
            .doe(Some(initial_doe.to_owned()))
            .expect(Some(ApproxValue {
                value: -15.1,
                tolerance: 1e-1,
            }))
            .outdir(Some(".".to_string()))
            .minimize()
            .expect("Minimize failure");
        let expected = array![-15.1];
        assert_abs_diff_eq!(expected, res.y_opt, epsilon = 0.5);
        let saved_doe: Array2<f64> = read_npy(DOE_INITIAL_FILE).unwrap();
        assert_abs_diff_eq!(initial_doe, saved_doe.slice(s![..3, ..1]), epsilon = 1e-6);
    }

    #[test]
    #[serial]
    fn test_xsinx_wb2() {
        let res = Egor::new(xsinx, &array![[0.0, 25.0]])
            .n_eval(20)
            .minimize()
            .expect("Minimize failure");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    #[test]
    #[serial]
    fn test_xsinx_auto_clustering() {
        let res = Egor::new(xsinx, &array![[0.0, 25.0]])
            .n_clusters(Some(0))
            .n_eval(20)
            .minimize()
            .expect("Minimize failure");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    #[test]
    #[serial]
    fn test_xsinx_suggestions() {
        let mut ego = Egor::new(xsinx, &array![[0.0, 25.0]]);
        let ego = ego.infill_strategy(InfillStrategy::EI);

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
    fn test_rosenbrock_2d() {
        let now = Instant::now();
        let xlimits = array![[-2., 2.], [-2., 2.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Isaac64Rng::seed_from_u64(42))
            .sample(10);
        let res = Egor::new(rosenb, &xlimits)
            .with_rng(Isaac64Rng::seed_from_u64(42))
            .infill_strategy(InfillStrategy::EI)
            .doe(Some(doe))
            .n_eval(35)
            .expect(Some(ApproxValue {
                value: 0.0,
                tolerance: 1e-2,
            }))
            .minimize()
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
    fn test_egor_g24_basic() {
        let x = array![[1., 2.]];
        println!("{:?}", f_g24(&x.view()));
        let xlimits = array![[0., 3.], [0., 4.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Isaac64Rng::seed_from_u64(42))
            .sample(5);
        let res = Egor::new(f_g24, &xlimits)
            .with_rng(Isaac64Rng::seed_from_u64(42))
            .n_cstr(2)
            .infill_strategy(InfillStrategy::EI)
            .infill_optimizer(InfillOptimizer::Cobyla) // test passes also with WB2S and Slsqp
            .doe(Some(doe))
            .n_eval(20)
            .minimize()
            .expect("Minimize failure");
        println!("G24 optim result = {:?}", res);
        let expected = array![2.3295, 3.1785];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 2e-2);
    }

    #[test]
    fn test_egor_g24_qei() {
        let xlimits = array![[0., 3.], [0., 4.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Isaac64Rng::seed_from_u64(42))
            .sample(10);
        let res = Egor::new(f_g24, &xlimits)
            .with_rng(Isaac64Rng::seed_from_u64(42))
            .n_cstr(2)
            .q_parallel(2)
            .qei_strategy(QEiStrategy::KrigingBeliever)
            .doe(Some(doe))
            .n_eval(20)
            .minimize()
            .expect("Minimize failure");
        println!("G24 optim result = {:?}", res);
        let expected = array![2.3295, 3.1785];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 2e-2);
    }
}
