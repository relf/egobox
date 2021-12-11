use doe::{SamplingMethod, LHS};
use ego::Egor;
use moe;
use ndarray::{Array2, ArrayView2};
use ndarray_rand::rand::SeedableRng;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3_log;
use rand_isaac::Isaac64Rng;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn lhs<'py>(
    py: Python<'py>,
    xlimits: PyReadonlyArray2<f64>,
    n_samples: usize,
    seed: Option<u64>,
) -> &'py PyArray2<f64> {
    let rng = match seed {
        Some(seed) => Isaac64Rng::seed_from_u64(seed),
        None => Isaac64Rng::from_entropy(),
    };
    let actual = LHS::new_with_rng(&xlimits.as_array(), rng).sample(n_samples);
    actual.into_pyarray(py)
}

#[pyclass]
struct RegressionSpec(u8);

#[pymethods]
impl RegressionSpec {
    #[classattr]
    const ALL: u8 = moe::RegressionSpec::ALL.bits();
    #[classattr]
    const CONSTANT: u8 = moe::RegressionSpec::CONSTANT.bits();
    #[classattr]
    const LINEAR: u8 = moe::RegressionSpec::LINEAR.bits();
    #[classattr]
    const QUADRATIC: u8 = moe::RegressionSpec::QUADRATIC.bits();
}

#[pyclass]
struct CorrelationSpec(u8);

#[pymethods]
impl CorrelationSpec {
    #[classattr]
    const ALL: u8 = moe::CorrelationSpec::ALL.bits();
    #[classattr]
    const SQUARED_EXPONENTIAL: u8 = moe::CorrelationSpec::SQUAREDEXPONENTIAL.bits();
    #[classattr]
    const ABSOLUTE_EXPONENTIAL: u8 = moe::CorrelationSpec::ABSOLUTEEXPONENTIAL.bits();
    #[classattr]
    const MATERN32: u8 = moe::CorrelationSpec::MATERN32.bits();
    #[classattr]
    const MATERN52: u8 = moe::CorrelationSpec::MATERN52.bits();
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

/// Optimizer constructor
///
/// Parameters
///
///     xlimits (array[nx, 2]):
///         Bounds of th nx components of the input x (eg. [[lower_1, upper_1], ..., [lower_nx, upper_nx]])
///
///     n_start (int > 0):
///         Number of runs of infill strategy optimizations (best result taken)
///
///     n_doe (int > 0):
///         Number of samples of initial LHS sampling (used when DOE not provided by the user).
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
///     n_parallel (int > 0):
///         Number of parallel evaluations of the function under optimization.
///
///     par_infill_strategy (ParInfillStrategy enum, an int in [1, 4])
///         Parallel infill criteria to get virtual next promising points in order to allow
///         n parallel evaluations of the function under optimization.
///         Can be either ParInfillStrategy.KB (1, Kriging Believer),
///         ParInfillStrategy.KBLB (2, KB Lower Bound), ParInfillStrategy.KBUB (2, KB Lower Bound),
///         ParInfillStrategy.CLMIN (2, Constant Liar Minimum)
///
///     infill_optimizer (InfillOptimizer enum, an int [1, 2])
///         Internal optimizer used to optimize infill criteria.
///         Can be either InfillOptimizer.COBYLA (1) or InfillOptimizer.SLSQP (2)
///
///     kpls_dim (0 < int < nx)
///         Number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
///         This is used to address high-dimensional problems typically when nx > 9.
///
///     n_clusters (0 < int)
///         Number of clusters used by the mixture of surrogate experts.
///         
#[pyclass]
#[pyo3(
    text_signature = "(xlimits, n_start=20, n_doe=10, regression_spec=7, correlation_spec=15, infill_strategy=1, n_parallel=1, par_infill_strategy=1, infill_optimizer=1, n_clusters=1)"
)]
struct Optimizer {
    pub xlimits: Array2<f64>,
    pub n_start: usize,
    pub n_doe: usize,
    pub doe: Option<Array2<f64>>,
    pub regression_spec: RegressionSpec,
    pub correlation_spec: CorrelationSpec,
    pub infill_strategy: InfillStrategy,
    pub n_parallel: usize,
    pub par_infill_strategy: ParInfillStrategy,
    pub infill_optimizer: InfillOptimizer,
    pub kpls_dim: Option<usize>,
    pub n_clusters: Option<usize>,
}

#[pyclass]
struct OptimResult {
    #[pyo3(get)]
    x_opt: Vec<f64>,
    #[pyo3(get)]
    y_opt: Vec<f64>,
}

#[pymethods]
impl Optimizer {
    #[new]
    #[args(
        xlimits,
        n_start = "20",
        n_doe = "10",
        doe = "None",
        regr_spec = "RegressionSpec::ALL",
        corr_spec = "CorrelationSpec::ALL",
        infill_strategy = "InfillStrategy::WB2",
        n_parallel = "1",
        par_infill_strategy = "ParInfillStrategy::KB",
        infill_optimizer = "InfillOptimizer::COBYLA",
        kpls_dim = "None",
        n_clusters = "1"
    )]
    fn new(
        xlimits: PyReadonlyArray2<f64>,
        n_start: usize,
        n_doe: usize,
        doe: Option<PyReadonlyArray2<f64>>,
        regr_spec: u8,
        corr_spec: u8,
        infill_strategy: u8,
        n_parallel: usize,
        par_infill_strategy: u8,
        infill_optimizer: u8,
        kpls_dim: Option<usize>,
        n_clusters: Option<usize>,
    ) -> Self {
        let xlimits = xlimits.to_owned_array();
        let doe = doe.map(|x| x.to_owned_array());
        Optimizer {
            xlimits,
            n_start,
            n_doe,
            doe,
            regression_spec: RegressionSpec(regr_spec),
            correlation_spec: CorrelationSpec(corr_spec),
            infill_strategy: InfillStrategy(infill_strategy),
            n_parallel,
            par_infill_strategy: ParInfillStrategy(par_infill_strategy),
            infill_optimizer: InfillOptimizer(infill_optimizer),
            kpls_dim,
            n_clusters,
        }
    }

    /// This function finds the minimum of a given fun function
    ///
    /// Parameters
    ///
    ///     fun: array[n, nx]) -> array[n, ny]
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
    ///     n_eval (int):
    ///         the function evaluation budget, number of fun calls.
    ///
    /// Returns
    ///
    ///     optimization result
    ///         x_opt (array[1, nx]): x value  where fun is at its minimum subject to constraint
    ///         y_opt (array[1, nx]): fun(x_opt)
    ///
    #[args(n_eval = "20", n_cstr = "0")]
    #[pyo3(text_signature = "(fun, n_eval=20, n_cstr=0)")]
    fn minimize(&self, fun: &PyAny, n_cstr: usize, n_eval: usize) -> OptimResult {
        let obj: PyObject = fun.into();
        let obj = move |x: &ArrayView2<f64>| -> Array2<f64> {
            let gil = Python::acquire_gil();
            let py = gil.python();
            let args = (x.to_owned().into_pyarray(py),);
            let res = obj.call1(py, args).unwrap();
            let pyarray: &PyArray2<f64> = res.extract(py).unwrap();
            let val = pyarray.to_owned_array();
            val
        };

        let infill_strategy = match self.infill_strategy.0 {
            InfillStrategy::EI => ego::InfillStrategy::EI,
            InfillStrategy::WB2 => ego::InfillStrategy::WB2,
            InfillStrategy::WB2S => ego::InfillStrategy::WB2S,
            _ => panic!(
                "InfillStrategy should be either EI ({}), WB2 ({}) or WB2S ({}), got {}",
                InfillStrategy::EI,
                InfillStrategy::WB2,
                InfillStrategy::WB2S,
                self.infill_strategy.0
            ),
        };

        let qei_strategy = match self.par_infill_strategy.0 {
            ParInfillStrategy::KB => ego::QEiStrategy::KrigingBeliever,
            ParInfillStrategy::KBLB => ego::QEiStrategy::KrigingBelieverLowerBound,
            ParInfillStrategy::KBUB => ego::QEiStrategy::KrigingBelieverUpperBound,
            ParInfillStrategy::CLMIN => ego::QEiStrategy::ConstantLiarMinimum,
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
            InfillOptimizer::COBYLA => ego::InfillOptimizer::Cobyla,
            InfillOptimizer::SLSQP => ego::InfillOptimizer::Slsqp,
            _ => panic!(
                "InfillOptimizer should be either COBYLA ({}) or SLSQP ({}), got {}",
                InfillOptimizer::COBYLA,
                InfillOptimizer::SLSQP,
                self.infill_optimizer.0
            ),
        };

        let doe = self.doe.as_ref().map(|v| v.to_owned());
        let res = Egor::new(obj, &self.xlimits)
            .n_cstr(n_cstr)
            .n_eval(n_eval)
            .n_start(self.n_start)
            .n_doe(self.n_doe)
            .doe(doe)
            .regression_spec(moe::RegressionSpec::from_bits(self.regression_spec.0).unwrap())
            .correlation_spec(moe::CorrelationSpec::from_bits(self.correlation_spec.0).unwrap())
            .infill_strategy(infill_strategy)
            .n_parallel(self.n_parallel)
            .qei_strategy(qei_strategy)
            .infill_optimizer(infill_optimizer)
            .kpls_dim(self.kpls_dim)
            .n_clusters(self.n_clusters)
            .minimize();

        OptimResult {
            x_opt: res.x_opt.to_vec(),
            y_opt: res.y_opt.to_vec(),
        }
    }
}

#[pymodule]
fn egobox(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(lhs, m)?)?;
    m.add_class::<Optimizer>()?;
    m.add_class::<RegressionSpec>()?;
    m.add_class::<CorrelationSpec>()?;
    m.add_class::<InfillStrategy>()?;
    m.add_class::<ParInfillStrategy>()?;
    m.add_class::<InfillOptimizer>()?;
    m.add_class::<OptimResult>()?;
    Ok(())
}
