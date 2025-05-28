use pyo3::prelude::*;

#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, PartialEq)]
pub enum Recombination {
    /// prediction is taken from the expert with highest responsability
    /// resulting in a model with discontinuities
    Hard = 0,
    /// Prediction is a combination experts prediction wrt their responsabilities,
    /// an optional heaviside factor might be used control steepness of the change between
    /// experts regions.
    Smooth = 1,
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct RegressionSpec(pub(crate) u8);

#[pymethods]
impl RegressionSpec {
    #[classattr]
    pub(crate) const ALL: u8 = egobox_moe::RegressionSpec::ALL.bits();
    #[classattr]
    pub(crate) const CONSTANT: u8 = egobox_moe::RegressionSpec::CONSTANT.bits();
    #[classattr]
    pub(crate) const LINEAR: u8 = egobox_moe::RegressionSpec::LINEAR.bits();
    #[classattr]
    pub(crate) const QUADRATIC: u8 = egobox_moe::RegressionSpec::QUADRATIC.bits();
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct CorrelationSpec(pub(crate) u8);

#[pymethods]
impl CorrelationSpec {
    #[classattr]
    pub(crate) const ALL: u8 = egobox_moe::CorrelationSpec::ALL.bits();
    #[classattr]
    pub(crate) const SQUARED_EXPONENTIAL: u8 =
        egobox_moe::CorrelationSpec::SQUAREDEXPONENTIAL.bits();
    #[classattr]
    pub(crate) const ABSOLUTE_EXPONENTIAL: u8 =
        egobox_moe::CorrelationSpec::ABSOLUTEEXPONENTIAL.bits();
    #[classattr]
    pub(crate) const MATERN32: u8 = egobox_moe::CorrelationSpec::MATERN32.bits();
    #[classattr]
    pub(crate) const MATERN52: u8 = egobox_moe::CorrelationSpec::MATERN52.bits();
}

#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum InfillStrategy {
    Ei = 1,
    Wb2 = 2,
    Wb2s = 3,
    LogEi = 4,
}

#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ConstraintStrategy {
    Mc = 1,
    Utb = 2,
}

#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum QInfillStrategy {
    Kb = 1,
    Kblb = 2,
    Kbub = 3,
    Clmin = 4,
}

#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum InfillOptimizer {
    Cobyla = 1,
    Slsqp = 2,
    Gbnm = 3,
}

#[pyclass]
#[derive(Clone, Copy)]
pub(crate) struct ExpectedOptimum {
    #[pyo3(get)]
    pub(crate) val: f64,
    #[pyo3(get)]
    pub(crate) tol: f64,
}

#[pymethods]
impl ExpectedOptimum {
    #[new]
    #[pyo3(signature = (value, tolerance = 1e-6))]
    fn new(value: f64, tolerance: f64) -> Self {
        ExpectedOptimum {
            val: value,
            tol: tolerance,
        }
    }
}

#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum XType {
    Float = 1,
    Int = 2,
    Ord = 3,
    Enum = 4,
}

#[pyclass]
#[derive(FromPyObject, Debug)]
pub(crate) struct XSpec {
    #[pyo3(get)]
    pub(crate) xtype: XType,
    #[pyo3(get)]
    pub(crate) xlimits: Vec<f64>,
    #[pyo3(get)]
    pub(crate) tags: Vec<String>,
}

#[pymethods]
impl XSpec {
    #[new]
    #[pyo3(signature = (xtype, xlimits=vec![], tags=vec![]))]
    pub(crate) fn new(xtype: XType, xlimits: Vec<f64>, tags: Vec<String>) -> Self {
        XSpec {
            xtype,
            xlimits,
            tags,
        }
    }
}

#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SparseMethod {
    Fitc = 1,
    Vfe = 2,
}
