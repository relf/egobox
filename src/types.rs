use pyo3::prelude::*;

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

#[pyclass]
#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) enum InfillStrategy {
    EI = 1,
    WB2 = 2,
    WB2S = 3,
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) enum ParInfillStrategy {
    KB = 1,
    KBLB = 2,
    KBUB = 3,
    CLMIN = 4,
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) enum InfillOptimizer {
    COBYLA = 1,
    SLSQP = 2,
}

#[pyclass]
#[derive(Clone, Copy)]
#[pyo3(text_signature = "(val, tol=1e-6)")]
pub(crate) struct ExpectedOptimum {
    #[pyo3(get)]
    pub(crate) val: f64,
    #[pyo3(get)]
    pub(crate) tol: f64,
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
pub(crate) struct Vtype(pub(crate) u8);

#[pymethods]
impl Vtype {
    #[classattr]
    pub(crate) const FLOAT: u8 = 1;
    #[classattr]
    pub(crate) const INT: u8 = 2;
    #[new]
    pub(crate) fn new(vtype: u8) -> Self {
        Vtype(vtype)
    }
    pub(crate) fn id(&self) -> u8 {
        self.0
    }
}

#[pyclass]
#[derive(FromPyObject, Debug)]
pub(crate) struct Vspec {
    #[pyo3(get)]
    pub(crate) vtype: Vtype,
    #[pyo3(get)]
    pub(crate) vlimits: Vec<f64>,
}

#[pymethods]
impl Vspec {
    #[new]
    pub(crate) fn new(vtype: Vtype, vlimits: Vec<f64>) -> Self {
        Vspec { vtype, vlimits }
    }
}
