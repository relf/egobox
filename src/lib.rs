//! `egobox`, Rust toolbox for efficient global optimization
//!
//! Thanks to the [PyO3 project](https://pyo3.rs), which makes Rust well suited for building Python extensions,
//! the EGO algorithm written in Rust (aka `egor`) is binded in Python. You can install the Python package using:
//!
//! ```bash
//! pip install egobox
//! ```
//!
//! See the [tutorial notebook](https://github.com/relf/egobox/doc/TutorialEgor.ipynb) for usage.
//!

use egobox_doe::SamplingMethod;
use linfa::ParamGuard;
use ndarray::{Array2, ArrayView2};
use ndarray_rand::rand::SeedableRng;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand_isaac::Isaac64Rng;

/// Utility function converting `xlimits` float data list specifying bounds of x components
/// to x specified as a list of Vtype.Float types [egobox.Vtype]
///
/// # Parameters
///     xlimits : nx-size list of [lower_bound, upper_bound] where `nx` is the dimension of x
///
/// # Returns
///     xtypes: nx-size list of Vspec(Vtype(FLOAT), [lower_bound, upper_bounds]) where `nx` is the dimension of x
#[pyfunction]
fn to_specs(py: Python, xlimits: Vec<Vec<f64>>) -> PyResult<PyObject> {
    if xlimits.is_empty() || xlimits[0].is_empty() {
        let err = "Error: xspecs argument cannot be empty";
        return Err(PyValueError::new_err(err.to_string()));
    }
    Ok(xlimits
        .iter()
        .map(|xlimit| Vspec::new(Vtype(Vtype::FLOAT), xlimit.clone()))
        .collect::<Vec<Vspec>>()
        .into_py(py))
}

#[pyfunction]
fn lhs(py: Python, xspecs: PyObject, n_samples: usize, seed: Option<u64>) -> &PyArray2<f64> {
    let specs: Vec<Vspec> = xspecs.extract(py).expect("Error in xspecs conversion");
    if specs.is_empty() {
        panic!("Error: xspecs argument cannot be empty")
    }
    let xtypes: Vec<egobox_ego::Xtype> = specs
        .iter()
        .map(|spec| match spec.vtype {
            Vtype(Vtype::FLOAT) => egobox_ego::Xtype::Cont(spec.vlimits[0], spec.vlimits[1]),
            Vtype(Vtype::INT) => {
                egobox_ego::Xtype::Int(spec.vlimits[0] as i32, spec.vlimits[1] as i32)
            }
            Vtype(i) => panic!(
                "Bad variable type: should be either Vtype.FLOAT {} or Vtype.INT {}, got {}",
                Vtype::FLOAT,
                Vtype::INT,
                i
            ),
        })
        .collect();
    let lhs = egobox_ego::MixintContext::new(&xtypes).create_sampling(seed);
    let doe = lhs.sample(n_samples);
    doe.into_pyarray(py)
}

#[pyclass]
struct RegressionSpec(u8);

#[pymethods]
impl RegressionSpec {
    #[classattr]
    const ALL: u8 = egobox_moe::RegressionSpec::ALL.bits();
    #[classattr]
    const CONSTANT: u8 = egobox_moe::RegressionSpec::CONSTANT.bits();
    #[classattr]
    const LINEAR: u8 = egobox_moe::RegressionSpec::LINEAR.bits();
    #[classattr]
    const QUADRATIC: u8 = egobox_moe::RegressionSpec::QUADRATIC.bits();
}

#[pyclass]
struct CorrelationSpec(u8);

#[pymethods]
impl CorrelationSpec {
    #[classattr]
    const ALL: u8 = egobox_moe::CorrelationSpec::ALL.bits();
    #[classattr]
    const SQUARED_EXPONENTIAL: u8 = egobox_moe::CorrelationSpec::SQUAREDEXPONENTIAL.bits();
    #[classattr]
    const ABSOLUTE_EXPONENTIAL: u8 = egobox_moe::CorrelationSpec::ABSOLUTEEXPONENTIAL.bits();
    #[classattr]
    const MATERN32: u8 = egobox_moe::CorrelationSpec::MATERN32.bits();
    #[classattr]
    const MATERN52: u8 = egobox_moe::CorrelationSpec::MATERN52.bits();
}

#[pyclass]
struct InfillStrategy(u8);

#[pymethods]
impl InfillStrategy {
    #[classattr]
    const EI: u8 = 1;
    #[classattr]
    const WB2: u8 = 2;
    #[classattr]
    const WB2S: u8 = 3;
}

#[pyclass]
struct ParInfillStrategy(u8);

#[pymethods]
impl ParInfillStrategy {
    #[classattr]
    const KB: u8 = 1;
    #[classattr]
    const KBLB: u8 = 2;
    #[classattr]
    const KBUB: u8 = 3;
    #[classattr]
    const CLMIN: u8 = 4;
}

#[pyclass]
struct InfillOptimizer(u8);

#[pymethods]
impl InfillOptimizer {
    #[classattr]
    const COBYLA: u8 = 1;
    #[classattr]
    const SLSQP: u8 = 2;
}

#[pyclass]
#[derive(Clone, Copy)]
#[pyo3(text_signature = "(val, tol=1e-6)")]
struct ExpectedOptimum {
    #[pyo3(get)]
    val: f64,
    #[pyo3(get)]
    tol: f64,
}

#[pymethods]
impl ExpectedOptimum {
    #[new]
    #[args(value, tolerance = "1e-6")]
    fn new(val: f64, tol: f64) -> Self {
        ExpectedOptimum { val, tol }
    }
}

#[pyclass]
#[derive(Clone, Copy, Debug)]
struct Vtype(u8);

#[pymethods]
impl Vtype {
    #[classattr]
    const FLOAT: u8 = 1;
    #[classattr]
    const INT: u8 = 2;
    #[new]
    fn new(vtype: u8) -> Self {
        Vtype(vtype)
    }
    fn id(&self) -> u8 {
        self.0
    }
}

#[pyclass]
#[derive(FromPyObject, Debug)]
struct Vspec {
    #[pyo3(get)]
    vtype: Vtype,
    #[pyo3(get)]
    vlimits: Vec<f64>,
}

#[pymethods]
impl Vspec {
    #[new]
    fn new(vtype: Vtype, vlimits: Vec<f64>) -> Self {
        Vspec { vtype, vlimits }
    }
}

/// Optimizer constructor
///
///    fun: array[n, nx]) -> array[n, ny]
///         the function to be minimized
///         fun(x) = [obj(x), cstr_1(x), ... cstr_k(x)] where
///            obj is the objective function [n, nx] -> [n, 1]
///            cstr_i is the ith constraint function [n, nx] -> [n, 1]
///            an k the number of constraints (n_cstr)
///            hence ny = 1 (obj) + k (cstrs)
///         cstr functions are expected be negative (<=0) at the optimum.
///
///     n_cstr (int):
///         the number of constraint functions.
///
///     cstr_tol (float):
///         tolerance on constraints violation (cstr < tol).
///
///     xspecs (list(Vspec)) where Vspec(vtype=FLOAT|INT, vlimits=[lower bound, upper bound]):
///         Bounds of the nx components of the input x (eg. len(xspecs) == nx)
///
///     n_start (int > 0):
///         Number of runs of infill strategy optimizations (best result taken)
///
///     n_doe (int >= 0):
///         Number of samples of initial LHS sampling (used when DOE not provided by the user).
///         When 0 a number of points is computed automatically regarding the number of input variables
///         of the function under optimization.
///
///     doe (array[ns, nt]):
///         Initial DOE containing ns samples:
///             either nt = nx then only x are specified and ns evals are done to get y doe values,
///             or nt = nx + ny then x = doe[:, :nx] and y = doe[:, nx:] are specified  
///
///     regr_spec (RegressionSpec flags, an int in [1, 7]):
///         Specification of regression models used in gaussian processes.
///         Can be RegressionSpec.CONSTANT (1), RegressionSpec.LINEAR (2), RegressionSpec.QUADRATIC (4) or
///         any bit-wise union of these values (e.g. RegressionSpec.CONSTANT | RegressionSpec.LINEAR)
///
///     corr_spec (CorrelationSpec flags, an int in [1, 15]):
///         Specification of correlation models used in gaussian processes.
///         Can be CorrelationSpec.SQUARED_EXPONENTIAL (1), CorrelationSpec.ABSOLUTE_EXPONENTIAL (2),
///         CorrelationSpec.MATERN32 (4), CorrelationSpec.MATERN52 (8) or
///         any bit-wise union of these values (e.g. CorrelationSpec.MATERN32 | CorrelationSpec.MATERN52)
///
///     infill_strategy (InfillStrategy enum, an int in [1, 3])
///         Infill criteria to decide best next promising point.
///         Can be either InfillStrategy.EI (1), InfillStrategy.WB2 (2) or InfillStrategy.WB2S (3)
///
///     q_parallel (int > 0):
///         Number of parallel evaluations of the function under optimization.
///
///     par_infill_strategy (ParInfillStrategy enum, an int in [1, 4])
///         Parallel infill criteria (aka qEI) to get virtual next promising points in order to allow
///         q parallel evaluations of the function under optimization.
///         Can be either ParInfillStrategy.KB (1, Kriging Believer),
///         ParInfillStrategy.KBLB (2, KB Lower Bound), ParInfillStrategy.KBUB (3, KB Lower Bound),
///         ParInfillStrategy.CLMIN (4, Constant Liar Minimum)
///
///     infill_optimizer (InfillOptimizer enum, an int [1, 2])
///         Internal optimizer used to optimize infill criteria.
///         Can be either InfillOptimizer.COBYLA (1) or InfillOptimizer.SLSQP (2)
///
///     kpls_dim (0 < int < nx)
///         Number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
///         This is used to address high-dimensional problems typically when nx > 9.
///
///     n_clusters (int > 0)
///         Number of clusters used by the mixture of surrogate experts.
///   
///     expected (ExpectedOptimum)
///         Known optimum used as stopping criterion.
///
///     outdir (String)
///         Directory to write optimization history and used as search path for hot start doe
///
///     hot_start (bool)
///         Start by loading initial doe from <outdir> directory
///
///     seed (int >= 0)
///         Random generator seed to allow computation reproducibility.
///      
#[pyclass]
#[pyo3(
    text_signature = "(fun, n_cstr=0, cstr_tol=1e-6, n_start=20, n_doe=0, regression_spec=7, correlation_spec=15, infill_strategy=1, q_parallel=1, par_infill_strategy=1, infill_optimizer=1, n_clusters=1)"
)]
struct Egor {
    pub fun: PyObject,
    pub xspecs: PyObject,
    pub n_cstr: usize,
    pub cstr_tol: f64,
    pub n_start: usize,
    pub n_doe: usize,
    pub doe: Option<Array2<f64>>,
    pub regression_spec: RegressionSpec,
    pub correlation_spec: CorrelationSpec,
    pub infill_strategy: InfillStrategy,
    pub q_parallel: usize,
    pub par_infill_strategy: ParInfillStrategy,
    pub infill_optimizer: InfillOptimizer,
    pub kpls_dim: Option<usize>,
    pub n_clusters: Option<usize>,
    pub expected: Option<ExpectedOptimum>,
    pub outdir: Option<String>,
    pub hot_start: bool,
    pub seed: Option<u64>,
}

#[pyclass]
struct OptimResult {
    #[pyo3(get)]
    x_opt: Vec<f64>,
    #[pyo3(get)]
    y_opt: Vec<f64>,
}

#[pymethods]
impl Egor {
    #[new]
    #[args(
        fun,
        xspecs,
        n_cstr = "0",
        cstr_tol = "1e-6",
        n_start = "20",
        n_doe = "0",
        doe = "None",
        regr_spec = "RegressionSpec::ALL",
        corr_spec = "CorrelationSpec::ALL",
        infill_strategy = "InfillStrategy::WB2",
        q_parallel = "1",
        par_infill_strategy = "ParInfillStrategy::KB",
        infill_optimizer = "InfillOptimizer::COBYLA",
        kpls_dim = "None",
        n_clusters = "1",
        expected = "None",
        outdir = "None",
        hot_start = "false",
        seed = "None"
    )]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python,
        fun: PyObject,
        xspecs: PyObject,
        n_cstr: usize,
        cstr_tol: f64,
        n_start: usize,
        n_doe: usize,
        doe: Option<PyReadonlyArray2<f64>>,
        regr_spec: u8,
        corr_spec: u8,
        infill_strategy: u8,
        q_parallel: usize,
        par_infill_strategy: u8,
        infill_optimizer: u8,
        kpls_dim: Option<usize>,
        n_clusters: Option<usize>,
        expected: Option<ExpectedOptimum>,
        outdir: Option<String>,
        hot_start: bool,
        seed: Option<u64>,
    ) -> Self {
        let doe = doe.map(|x| x.to_owned_array());
        Egor {
            fun: fun.to_object(py),
            xspecs,
            n_cstr,
            cstr_tol,
            n_start,
            n_doe,
            doe,
            regression_spec: RegressionSpec(regr_spec),
            correlation_spec: CorrelationSpec(corr_spec),
            infill_strategy: InfillStrategy(infill_strategy),
            q_parallel,
            par_infill_strategy: ParInfillStrategy(par_infill_strategy),
            infill_optimizer: InfillOptimizer(infill_optimizer),
            kpls_dim,
            n_clusters,
            expected,
            outdir,
            hot_start,
            seed,
        }
    }

    /// This function finds the minimum of a given function `fun`
    ///
    /// # Parameters
    ///     n_eval:
    ///         the function evaluation budget, number of fun calls.
    ///
    /// # Returns
    ///     optimization result
    ///         x_opt (array[1, nx]): x value  where fun is at its minimum subject to constraint
    ///         y_opt (array[1, nx]): fun(x_opt)
    ///
    #[args(n_eval = "20")]
    #[pyo3(text_signature = "(n_eval=20)")]
    fn minimize(&self, py: Python, n_eval: usize) -> OptimResult {
        let fun = self.fun.to_object(py);
        let obj = move |x: &ArrayView2<f64>| -> Array2<f64> {
            let gil = Python::acquire_gil();
            let py = gil.python();
            let args = (x.to_owned().into_pyarray(py),);
            let res = fun.call1(py, args).unwrap();
            let pyarray: &PyArray2<f64> = res.extract(py).unwrap();
            pyarray.to_owned_array()
        };

        let infill_strategy = match self.infill_strategy.0 {
            InfillStrategy::EI => egobox_ego::InfillStrategy::EI,
            InfillStrategy::WB2 => egobox_ego::InfillStrategy::WB2,
            InfillStrategy::WB2S => egobox_ego::InfillStrategy::WB2S,
            _ => panic!(
                "InfillStrategy should be either EI ({}), WB2 ({}) or WB2S ({}), got {}",
                InfillStrategy::EI,
                InfillStrategy::WB2,
                InfillStrategy::WB2S,
                self.infill_strategy.0
            ),
        };

        let qei_strategy = match self.par_infill_strategy.0 {
            ParInfillStrategy::KB => egobox_ego::QEiStrategy::KrigingBeliever,
            ParInfillStrategy::KBLB => egobox_ego::QEiStrategy::KrigingBelieverLowerBound,
            ParInfillStrategy::KBUB => egobox_ego::QEiStrategy::KrigingBelieverUpperBound,
            ParInfillStrategy::CLMIN => egobox_ego::QEiStrategy::ConstantLiarMinimum,
            _ => panic!(
                "ParInfillStrategy should be either KB ({}), KBLB ({}), KBUB ({}) or CLMIN ({}), got {}",
                ParInfillStrategy::KB,
                ParInfillStrategy::KBLB,
                ParInfillStrategy::KBUB,
                ParInfillStrategy::CLMIN,
                self.par_infill_strategy.0
            ),
        };

        let infill_optimizer = match self.infill_optimizer.0 {
            InfillOptimizer::COBYLA => egobox_ego::InfillOptimizer::Cobyla,
            InfillOptimizer::SLSQP => egobox_ego::InfillOptimizer::Slsqp,
            _ => panic!(
                "InfillOptimizer should be either COBYLA ({}) or SLSQP ({}), got {}",
                InfillOptimizer::COBYLA,
                InfillOptimizer::SLSQP,
                self.infill_optimizer.0
            ),
        };

        let rng = if let Some(seed) = self.seed {
            Isaac64Rng::seed_from_u64(seed)
        } else {
            Isaac64Rng::from_entropy()
        };

        let expected = self.expected.map(|opt| egobox_ego::ApproxValue {
            value: opt.val,
            tolerance: opt.tol,
        });

        let doe = self.doe.as_ref().map(|v| v.to_owned());

        let xspecs: Vec<Vspec> = self.xspecs.extract(py).expect("Error in xspecs conversion");
        if xspecs.is_empty() {
            panic!("Error: xspecs argument cannot be empty")
        }

        let xtypes: Vec<egobox_ego::Xtype> = xspecs
            .iter()
            .map(|spec| match spec.vtype {
                Vtype(Vtype::FLOAT) => egobox_ego::Xtype::Cont(spec.vlimits[0], spec.vlimits[1]),
                Vtype(Vtype::INT) => {
                    egobox_ego::Xtype::Int(spec.vlimits[0] as i32, spec.vlimits[1] as i32)
                }
                Vtype(i) => panic!(
                    "Bad variable type: should be either Vtype.FLOAT {} or Vtype.INT {}, got {}",
                    Vtype::FLOAT,
                    Vtype::INT,
                    i
                ),
            })
            .collect();

        let surrogate_builder = egobox_moe::MoeParams::default()
            .n_clusters(self.n_clusters.unwrap_or(1))
            .kpls_dim(self.kpls_dim)
            .regression_spec(egobox_moe::RegressionSpec::from_bits(self.regression_spec.0).unwrap())
            .correlation_spec(
                egobox_moe::CorrelationSpec::from_bits(self.correlation_spec.0).unwrap(),
            );
        let surrogate_builder = egobox_ego::MixintMoeParams::new(&xtypes, &surrogate_builder)
            .check()
            .unwrap();
        let pre_proc = egobox_ego::MixintPreProcessor::new(&xtypes);
        let mut mixintegor =
            egobox_ego::MixintEgor::new_with_rng(obj, &surrogate_builder, &pre_proc, rng);
        mixintegor
            .egor
            .n_cstr(self.n_cstr)
            .n_eval(n_eval)
            .n_start(self.n_start)
            .n_doe(self.n_doe)
            .cstr_tol(self.cstr_tol)
            .doe(doe)
            .regression_spec(egobox_moe::RegressionSpec::from_bits(self.regression_spec.0).unwrap())
            .correlation_spec(
                egobox_moe::CorrelationSpec::from_bits(self.correlation_spec.0).unwrap(),
            )
            .infill_strategy(infill_strategy)
            .q_parallel(self.q_parallel)
            .qei_strategy(qei_strategy)
            .infill_optimizer(infill_optimizer)
            .kpls_dim(self.kpls_dim)
            .n_clusters(self.n_clusters)
            .expect(expected)
            .outdir(self.outdir.as_ref().cloned())
            .hot_start(self.hot_start);

        let res = mixintegor.minimize().expect("Minimization failed");

        OptimResult {
            x_opt: res.x_opt.to_vec(),
            y_opt: res.y_opt.to_vec(),
        }
    }
}

#[doc(hidden)]
#[pymodule]
fn egobox(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(to_specs, m)?)?;
    m.add_function(wrap_pyfunction!(lhs, m)?)?;
    m.add_class::<Egor>()?;
    m.add_class::<RegressionSpec>()?;
    m.add_class::<CorrelationSpec>()?;
    m.add_class::<InfillStrategy>()?;
    m.add_class::<ParInfillStrategy>()?;
    m.add_class::<InfillOptimizer>()?;
    m.add_class::<Vtype>()?;
    m.add_class::<Vspec>()?;
    m.add_class::<OptimResult>()?;
    m.add_class::<ExpectedOptimum>()?;
    Ok(())
}
