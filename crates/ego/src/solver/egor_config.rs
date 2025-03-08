//! Egor optimizer configuration.
use crate::criteria::*;
use crate::types::*;
use crate::HotStartMode;
use egobox_moe::NbClusters;
use egobox_moe::{CorrelationSpec, RegressionSpec};
use ndarray::Array1;
use ndarray::Array2;

use serde::{Deserialize, Serialize};

/// A structure to handle TREGO method parameterization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct TregoConfig {
    pub(crate) activated: bool,
    pub(crate) n_local_steps: u64,
    pub(crate) d: (f64, f64),
    pub(crate) beta: f64,
    pub(crate) gamma: f64,
    pub(crate) sigma0: f64,
}

impl Default for TregoConfig {
    fn default() -> Self {
        TregoConfig {
            activated: false,
            n_local_steps: 4,
            d: (1e-6, 1.),
            beta: 0.9,
            gamma: 10. / 9.,
            sigma0: 1e-1,
        }
    }
}

/// Egor optimizer configuration
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct EgorConfig {
    /// Max number of function iterations allocated to find the optimum (aka iteration budget)
    /// Note 1 : The number of cost function evaluations is deduced using the following formula (n_doe + max_iters)
    /// Note 2 : When q_points > 1, the number of cost function evaluations is (n_doe + max_iters * q_points)
    /// is is an upper bounds as some points may be rejected as being to close to previous ones.   
    pub(crate) max_iters: usize,
    /// Number of starts for multistart approach used for hyperparameters optimization
    pub(crate) n_start: usize,
    /// Interval between two hyperparameters optimizations (as iteration number modulo)
    /// hyperparameters are optimized or re-used from an iteration to another
    pub(crate) n_optmod: usize,
    /// Number of points returned by EGO iteration (aka qEI Multipoint strategy)
    /// Actually as some point determination may fail (at most q_points are returned)
    pub(crate) q_points: usize,
    /// Number of initial doe drawn using Latin hypercube sampling
    /// Note: n_doe > 0; otherwise n_doe = max(xdim + 1, 5)
    pub(crate) n_doe: usize,
    /// Number of Constraints
    /// Note: dim function ouput = 1 objective + n_cstr constraints
    pub(crate) n_cstr: usize,
    /// Optional constraints violation tolerance meaning cstr < cstr_tol is considered valid
    pub(crate) cstr_tol: Option<Array1<f64>>,
    /// Initial doe can be either \[x\] with x inputs only or an evaluated doe \[x, y\]
    /// Note: x dimension is determined using `xlimits.nrows()`
    pub(crate) doe: Option<Array2<f64>>,
    /// Multipoint strategy used to get several points to be evaluated at each iteration
    pub(crate) q_ei: QEiStrategy,
    /// Criterion to select next point to evaluate
    pub(crate) infill_criterion: Box<dyn InfillCriterion>,
    /// The optimizer used to optimize infill criterium
    pub(crate) infill_optimizer: InfillOptimizer,
    /// Regression specification for GP models used by mixture of experts (see [egobox_moe])
    pub(crate) regression_spec: RegressionSpec,
    /// Correlation specification for GP models used by mixture of experts (see [egobox_moe])
    pub(crate) correlation_spec: CorrelationSpec,
    /// Optional dimension reduction (see [egobox_moe])
    pub(crate) kpls_dim: Option<usize>,
    /// Number of clusters used by mixture of experts (see [egobox_moe])
    /// When set to Auto the clusters are computes automatically and refreshed
    /// every 10-points (tentative) additions
    pub(crate) n_clusters: NbClusters,
    /// Specification of a target objective value which is used to stop the algorithm once reached
    pub(crate) target: f64,
    /// Directory to save intermediate results: inital doe + evalutions at each iteration
    pub(crate) outdir: Option<String>,
    /// If true use `outdir` to retrieve and start from previous results
    pub(crate) warm_start: bool,
    /// If some enable checkpointing allowing to restart for given ext_iters number of iteration from last checkpointed iteration
    pub(crate) hot_start: HotStartMode,
    /// List of x types allowing the handling of discrete input variables
    pub(crate) xtypes: Vec<XType>,
    /// A random generator seed used to get reproductible results.
    pub(crate) seed: Option<u64>,
    /// Trego parameterization
    pub(crate) trego: TregoConfig,
    /// Constraints criterion
    pub(crate) cstr_strategy: ConstraintStrategy,
}

impl Default for EgorConfig {
    fn default() -> Self {
        EgorConfig {
            max_iters: 20,
            n_start: 20,
            n_optmod: 1,
            q_points: 1,
            n_doe: 0,
            n_cstr: 0,
            cstr_tol: None,
            doe: None,
            q_ei: QEiStrategy::KrigingBeliever,
            infill_criterion: Box::new(WB2),
            infill_optimizer: InfillOptimizer::Slsqp,
            regression_spec: RegressionSpec::CONSTANT,
            correlation_spec: CorrelationSpec::SQUAREDEXPONENTIAL,
            kpls_dim: None,
            n_clusters: NbClusters::default(),
            target: f64::NEG_INFINITY,
            outdir: None,
            warm_start: false,
            hot_start: HotStartMode::Disabled,
            xtypes: vec![],
            seed: None,
            trego: TregoConfig::default(),
            cstr_strategy: ConstraintStrategy::MeanValue,
        }
    }
}

impl EgorConfig {
    pub fn infill_criterion(mut self, infill_criterion: Box<dyn InfillCriterion>) -> Self {
        self.infill_criterion = infill_criterion;
        self
    }

    /// Sets max number of iterations to optimize the objective function
    pub fn max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Sets the number of runs of infill strategy optimizations (best result taken)
    pub fn n_start(mut self, n_start: usize) -> Self {
        self.n_start = n_start;
        self
    }

    /// Sets the number of iteration interval between two hyperparameter optimization
    pub fn n_optmod(mut self, n_optmod: usize) -> Self {
        self.n_optmod = n_optmod;
        self
    }

    /// Sets Number of parallel evaluations of the function under optimization
    pub fn q_points(mut self, q_points: usize) -> Self {
        self.q_points = q_points;
        self
    }

    /// Number of samples of initial LHS sampling (used when DOE not provided by the user)
    ///
    /// When 0 a number of points is computed automatically regarding the number of input variables
    /// of the function under optimization.
    pub fn n_doe(mut self, n_doe: usize) -> Self {
        self.n_doe = n_doe;
        self
    }

    /// Sets the number of constraint functions
    pub fn n_cstr(mut self, n_cstr: usize) -> Self {
        self.n_cstr = n_cstr;
        self
    }

    /// Sets the tolerance on constraints violation (`cstr < tol`)
    pub fn cstr_tol(mut self, tol: Array1<f64>) -> Self {
        self.cstr_tol = Some(tol);
        self
    }

    /// Sets an initial DOE \['ns', `nt`\] containing `ns` samples.
    ///
    /// Either `nt` = `nx` then only `x` input values are specified and `ns` evals are done to get y ouput doe values,
    /// or `nt = nx + ny` then `x = doe\[:, :nx\]` and `y = doe\[:, nx:\]` are specified
    pub fn doe(mut self, doe: &Array2<f64>) -> Self {
        self.doe = Some(doe.to_owned());
        self
    }

    /// Removes any previously specified initial doe to get the default doe usage
    pub fn default_doe(mut self) -> Self {
        self.doe = None;
        self
    }

    /// Sets the parallel infill strategy
    ///
    /// Parallel infill criterion to get virtual next promising points in order to allow
    /// n parallel evaluations of the function under optimization.
    pub fn qei_strategy(mut self, q_ei: QEiStrategy) -> Self {
        self.q_ei = q_ei;
        self
    }

    /// Sets the infill strategy
    pub fn infill_strategy(mut self, infill: InfillStrategy) -> Self {
        self.infill_criterion = match infill {
            InfillStrategy::EI => Box::new(EI),
            InfillStrategy::WB2 => Box::new(WB2),
            InfillStrategy::WB2S => Box::new(WB2S),
        };
        self
    }

    /// Sets the infill optimizer
    pub fn infill_optimizer(mut self, optimizer: InfillOptimizer) -> Self {
        self.infill_optimizer = optimizer;
        self
    }

    /// Sets the allowed regression models used in gaussian processes.
    pub fn regression_spec(mut self, regression_spec: RegressionSpec) -> Self {
        self.regression_spec = regression_spec;
        self
    }

    /// Sets the allowed correlation models used in gaussian processes.
    pub fn correlation_spec(mut self, correlation_spec: CorrelationSpec) -> Self {
        self.correlation_spec = correlation_spec;
        self
    }

    /// Sets the number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
    ///
    /// This is used to address high-dimensional problems typically when `nx` > 9 wher `nx` is the dimension of `x`.
    pub fn kpls_dim(mut self, kpls_dim: usize) -> Self {
        self.kpls_dim = Some(kpls_dim);
        self
    }

    /// Removes any PLS dimension reduction usage
    pub fn no_kpls(mut self) -> Self {
        self.kpls_dim = None;
        self
    }

    /// Sets the number of clusters used by the mixture of surrogate experts.
    ///
    /// When set to Auto, the number of clusters is determined automatically
    pub fn n_clusters(mut self, n_clusters: NbClusters) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    /// Sets a known target minimum to be used as a stopping criterion.
    pub fn target(mut self, target: f64) -> Self {
        self.target = target;
        self
    }

    /// Sets a directory to write optimization history and used as search path for warm start doe
    pub fn outdir(mut self, outdir: impl Into<String>) -> Self {
        self.outdir = Some(outdir.into());
        self
    }
    /// Do not write optimization history
    pub fn no_outdir(mut self) -> Self {
        self.outdir = None;
        self
    }

    /// Whether we start by loading last DOE saved in `outdir` as initial DOE
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Whether checkpointing is enabled allowing hot start from previous checkpointed iteration if any
    pub fn hot_start(mut self, hot_start: HotStartMode) -> Self {
        self.hot_start = hot_start;
        self
    }

    /// Allow to specify a seed for random number generator to allow
    /// reproducible runs.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Define design space with given x types
    pub fn xtypes(mut self, xtypes: &[XType]) -> Self {
        self.xtypes = xtypes.into();
        self
    }

    /// Activate TREGO method
    pub fn trego(mut self, activated: bool) -> Self {
        self.trego.activated = activated;
        self
    }

    /// Sets the infill strategy
    pub fn cstr_strategy(mut self, cstr_strategy: ConstraintStrategy) -> Self {
        self.cstr_strategy = cstr_strategy;
        self
    }

    /// Check whether we are in a discrete optimization context
    pub fn discrete(&self) -> bool {
        crate::utils::discrete(&self.xtypes)
    }
}
