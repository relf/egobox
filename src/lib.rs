use doe::{SamplingMethod, LHS};
use ego::Sego;
use moe;
use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3_log;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn lhs<'py>(py: Python<'py>, xlimits: PyReadonlyArray2<f64>, a: usize) -> &'py PyArray2<f64> {
    let actual = LHS::new(&xlimits.as_array()).sample(a);
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
struct InfillOptimizer(u8);

#[pymethods]
impl InfillOptimizer {
    #[classattr]
    const COBYLA: u8 = 1;
    #[classattr]
    const SLSQP: u8 = 2;
}

#[pyclass]
#[pyo3(text_signature = "(xlimits, n_start=20, n_doe=10, 
    regression_spec=RegressionSpec.ALL, correlation_spec=CorrelationSpec.ALL,
    infill_strategy=InfillStrategy.WBS2, infill_optimizer=InfillOptimizer.COBYLA)")]
struct SegoOptimizer {
    pub xlimits: Array2<f64>,
    pub n_start: usize,
    pub n_doe: usize,
    pub regression_spec: RegressionSpec,
    pub correlation_spec: CorrelationSpec,
    pub infill_strategy: InfillStrategy,
    pub infill_optimizer: InfillOptimizer,
    // pub x_doe: Option<Array2<f64>>,
    // pub q_ei: QEiStrategy,
}

#[pyclass]
struct OptimResult {
    #[pyo3(get)]
    x_opt: Vec<f64>,
    #[pyo3(get)]
    y_opt: Vec<f64>,
}

#[pymethods]
impl SegoOptimizer {
    /// Constructor
    ///
    /// Parameters
    ///
    ///     xlimits (array[nx, 2]):
    ///         bounds of x components (eg. [[lower_1, upper_1], ..., [lower_nx, upper_nx]])
    ///
    ///     n_start (int > 0):
    ///         number of runs of infill strategy optimizations (best result taken)
    ///
    ///     n_doe (int > 0):
    ///         number of samples of initial LHS sampling (used when DOE not provided by the user)
    ///
    ///     regr_spec (RegressionSpec):
    ///         specification of regression models used in gaussian processes
    ///
    ///     corr_spec (CorrelationSpec):
    ///         specification of correlation models used in gaussian processes
    ///
    ///     infill_strategy (InfillStrategy)
    ///         infill criteria either EI, WB2 or WB2S (default WB2S)
    ///
    ///     infill_optimizer (InfillOptimizer)
    ///         intern optimizer used to optimize infill criteria (default COBYLA)
    #[new]
    #[args(
        xlimits,
        n_start = "20",
        n_doe = "10",
        regr_spec = "RegressionSpec::ALL",
        corr_spec = "CorrelationSpec::ALL",
        infill_strategy = "InfillStrategy::WB2",
        infill_optimizer = "InfillOptimizer::COBYLA"
    )]
    fn new(
        xlimits: PyReadonlyArray2<f64>,
        n_start: usize,
        n_doe: usize,
        regr_spec: u8,
        corr_spec: u8,
        infill_strategy: u8,
        infill_optimizer: u8,
    ) -> Self {
        let xlimits = xlimits.to_owned_array();
        SegoOptimizer {
            xlimits,
            n_start,
            n_doe,
            regression_spec: RegressionSpec(regr_spec),
            correlation_spec: CorrelationSpec(corr_spec),
            infill_strategy: InfillStrategy(infill_strategy),
            infill_optimizer: InfillOptimizer(infill_optimizer),
        }
    }

    /// This function finds the minimum of a given function
    ///
    /// Parameters
    ///
    ///     fun: the function to be minimized
    ///          fun: array[n, nx]) -> array[n, ny]
    ///          fun(x) = [obj(x), cstr_1(x), ... cstr_k(x)] where
    ///             obj is the objective function [n, nx] -> [n, 1]
    ///             cstr_i is the ith constraint function [n, nx] -> [n, 1]
    ///             an k the number of constraints (n_cstr)
    ///             hence ny = 1 (obj) + k (cstrs)
    ///          cstr functions are expected be negative (<=0) at the optimum.
    ///
    ///     n_cstr (int): the number of constraints (default 0)
    ///             
    ///     n_eval (int): the function evaluation budget, number of fun calls (default 20)
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
                "InfillOptimizer should be either EI ({}), WB2 ({}) or WB2S ({}), got {}",
                InfillStrategy::EI,
                InfillStrategy::WB2,
                InfillStrategy::WB2S,
                self.infill_optimizer.0
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

        let res = Sego::new(obj, &self.xlimits)
            .n_cstr(n_cstr)
            .n_eval(n_eval)
            .n_start(self.n_start)
            .n_doe(self.n_doe)
            .regression_spec(moe::RegressionSpec::from_bits(self.regression_spec.0).unwrap())
            .correlation_spec(moe::CorrelationSpec::from_bits(self.correlation_spec.0).unwrap())
            .infill_strategy(infill_strategy)
            .infill_optimizer(infill_optimizer)
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
    m.add_class::<SegoOptimizer>()?;
    m.add_class::<RegressionSpec>()?;
    m.add_class::<CorrelationSpec>()?;
    m.add_class::<InfillStrategy>()?;
    m.add_class::<InfillOptimizer>()?;
    m.add_class::<OptimResult>()?;
    Ok(())
}
