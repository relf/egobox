use crate::types::*;
use pyo3::prelude::*;

/// GP configuration used by `Egor` and `GpMix`
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
///     n_clusters (int)
///         Number of clusters used by the mixture of surrogate experts (default is 1).
///         When set to 0, the number of cluster is determined automatically and refreshed every
///         10-points addition (should say 'tentative addition' because addition may fail for some points
///         but it is counted anyway).
///         When set to negative number -n, the number of clusters is determined automatically in [1, n]
///         this is used to limit the number of trials hence the execution time.
///
///     kpls_dim (0 < int < nx)
///         Number of components to be used when PLS projection is used (a.k.a KPLS method).
///         This is used to address high-dimensional problems typically when nx > 9.
///
#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct GpConfig {
    #[pyo3(get, set)]
    pub regr_spec: RegressionSpec,
    #[pyo3(get, set)]
    pub corr_spec: CorrelationSpec,
    #[pyo3(get, set)]
    pub n_clusters: isize,
    #[pyo3(get, set)]
    pub kpls_dim: Option<usize>,
}

impl Default for GpConfig {
    fn default() -> Self {
        GpConfig::new(
            RegressionSpec::CONSTANT,
            CorrelationSpec::SQUARED_EXPONENTIAL,
            1,
            None,
        )
    }
}

#[pymethods]
impl GpConfig {
    #[new]
    #[pyo3(signature = (regr_spec=RegressionSpec::CONSTANT, corr_spec=CorrelationSpec::SQUARED_EXPONENTIAL, n_clusters=1, kpls_dim=None))]
    pub fn new(regr_spec: u8, corr_spec: u8, n_clusters: isize, kpls_dim: Option<usize>) -> Self {
        GpConfig {
            regr_spec: RegressionSpec(regr_spec),
            corr_spec: CorrelationSpec(corr_spec),
            n_clusters,
            kpls_dim,
        }
    }
}
