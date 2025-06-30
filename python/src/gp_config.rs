use crate::types::*;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyclass;

/// GP configuration used by `Egor` and `GpMix`
#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct GpConfig {
    /// (RegressionSpec flags, an int in [1, 7])
    ///   Specification of regression models used in mixture.
    ///   Can be RegressionSpec.CONSTANT (1), RegressionSpec.LINEAR (2), RegressionSpec.QUADRATIC (4) or
    ///   any bit-wise union of these values (e.g. RegressionSpec.CONSTANT | RegressionSpec.LINEAR)
    #[pyo3(get, set)]
    pub regr_spec: u8,
    /// (CorrelationSpec flags, an int in [1, 15])
    ///   Specification of correlation models used in mixture.
    ///   Can be CorrelationSpec.SQUARED_EXPONENTIAL (1), CorrelationSpec.ABSOLUTE_EXPONENTIAL (2),
    ///   CorrelationSpec.MATERN32 (4), CorrelationSpec.MATERN52 (8) or
    ///   any bit-wise union of these values (e.g. CorrelationSpec.MATERN32 | CorrelationSpec.MATERN52)
    #[pyo3(get, set)]
    pub corr_spec: u8,
    /// (0 < int < nx where nx is the dimension of inputs x)
    ///   Number of components to be used when PLS projection is used (a.k.a KPLS method).
    ///   This is used to address high-dimensional problems typically when nx > 9.
    #[pyo3(get, set)]
    pub kpls_dim: Option<usize>,
    /// (int)
    ///   Number of clusters used by the mixture of surrogate experts (default is 1).
    ///   When set to 0, the number of cluster is determined automatically and refreshed every
    ///   10-points addition (should say 'tentative addition' because addition may fail for some points
    ///   but it is counted anyway).
    ///   When set to negative number -n, the number of clusters is determined automatically in [1, n]
    ///   this is used to limit the number of trials hence the execution time.

    #[pyo3(get, set)]
    pub n_clusters: isize,
    /// (Recombination.Smooth or Recombination.Hard (default))
    ///   Specify how the various experts predictions are recombined
    ///   * Smooth: prediction is a combination of experts prediction wrt their responsabilities,
    ///   the heaviside factor which controls steepness of the change between experts regions is optimized
    ///   to get best mixture quality.
    ///   * Hard: prediction is taken from the expert with highest responsability
    ///   resulting in a model with discontinuities.
    #[pyo3(get, set)]
    pub recombination: Recombination,
    /// ([nx] where nx is the dimension of inputs x)
    ///   Initial guess for GP theta hyperparameters.
    ///   When None the default is 1e-2 for all components
    #[pyo3(get, set)]
    pub theta_init: Option<Vec<f64>>,
    /// ([[lower_1, upper_1], ..., [lower_nx, upper_nx]] where nx is the dimension of inputs x)
    ///   Space search when optimizing theta GP hyperparameters
    ///   When None the default is [1e-6, 1e2] for all components
    #[pyo3(get, set)]
    pub theta_bounds: Option<Vec<Vec<f64>>>,
    /// (int >= 0)
    ///   Number of internal GP hyperpameters optimization restart (multistart)
    ///   When is negative optimization is disabled and theta init value is used
    #[pyo3(get, set)]
    pub n_start: isize,
    /// (int >= 0)
    ///   Max number of likelihood evaluations during GP hyperparameters optimization
    #[pyo3(get, set)]
    pub max_eval: usize,
}

impl Default for GpConfig {
    fn default() -> Self {
        GpConfig::new(
            RegressionSpec::CONSTANT,
            CorrelationSpec::SQUARED_EXPONENTIAL,
            None,
            1,
            Recombination::Hard,
            None,
            None,
            egobox_ego::EGO_GP_OPTIM_N_START as isize,
            egobox_ego::EGO_GP_OPTIM_MAX_EVAL,
        )
    }
}

#[pymethods]
impl GpConfig {
    #[new]
    #[pyo3(signature = (
        regr_spec=GpConfig::default().regr_spec,
        corr_spec=GpConfig::default().corr_spec,
        kpls_dim=GpConfig::default().kpls_dim,
        n_clusters=GpConfig::default().n_clusters,
        recombination=GpConfig::default().recombination,
        theta_init=GpConfig::default().theta_init,
        theta_bounds=GpConfig::default().theta_bounds,
        n_start=GpConfig::default().n_start,
        max_eval=GpConfig::default().max_eval,
))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        regr_spec: u8,
        corr_spec: u8,
        kpls_dim: Option<usize>,
        n_clusters: isize,
        recombination: Recombination,
        theta_init: Option<Vec<f64>>,
        theta_bounds: Option<Vec<Vec<f64>>>,
        n_start: isize,
        max_eval: usize,
    ) -> Self {
        GpConfig {
            regr_spec,
            corr_spec,
            kpls_dim,
            n_clusters,
            recombination,
            theta_init,
            theta_bounds,
            n_start,
            max_eval,
        }
    }
}
