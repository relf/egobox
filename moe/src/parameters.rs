use crate::errors::Result;
use crate::surrogates::Surrogate;
use bitflags::bitflags;
#[allow(unused_imports)]
use egobox_gp::correlation_models::{
    AbsoluteExponentialCorr, Matern32Corr, Matern52Corr, SquaredExponentialCorr,
};
#[allow(unused_imports)]
use egobox_gp::mean_models::{ConstantMean, LinearMean, QuadraticMean};
use linfa::Float;
use linfa_clustering::GaussianMixtureModel;
use ndarray::Array2;
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;

#[cfg(feature = "persistent")]
use serde::{Deserialize, Serialize};

/// Enumeration of recombination modes handled by the mixture
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "persistent", derive(Serialize, Deserialize))]
pub enum Recombination<F: Float> {
    /// prediction is taken from the expert with highest responsability
    /// resulting in discontinuity
    Hard,
    /// prediction is a combination experts prediction wrt their responsabilities
    /// Takes an optional heaviside factor to control steepness of the change between
    /// experts regions.
    Smooth(Option<F>),
}

bitflags! {
    /// Flags to specify tested regression models during experts selection (see [MoeParams::set_regression_spec]).
    ///
    /// Flags can be combine with bit-wise `or` operator to select two or more models.
    /// ```ignore
    /// let spec = RegressionSpec::CONSTANT | RegressionSpec::LINEAR;
    /// ```
    ///
    /// See [bitflags::bitflags]
    pub struct RegressionSpec: u8 {
        /// Constant regression
        const CONSTANT = 0x01;
        /// Linear regression
        const LINEAR = 0x02;
        /// 2-degree polynomial regression
        const QUADRATIC = 0x04;
        /// All regression models available
        const ALL = RegressionSpec::CONSTANT.bits
                    | RegressionSpec::LINEAR.bits
                    | RegressionSpec::QUADRATIC.bits;
    }
}

bitflags! {
    /// Flags to specify tested correlation models during experts selection (see [MoeParams]).
    ///
    /// Flags can be combine with bit-wise `or` operator to select two or more models.
    /// ```ignore
    /// let spec = CorrelationSpec::MATERN32 | CorrelationSpec::Matern52;
    /// ```
    ///
    /// See [bitflags::bitflags]
    pub struct CorrelationSpec: u8 {
        /// Squared exponential correlation model
        const SQUAREDEXPONENTIAL = 0x01;
        /// Absolute exponential correlation model
        const ABSOLUTEEXPONENTIAL = 0x02;
        /// Matern 3/2 correlation model
        const MATERN32 = 0x04;
        /// Matern 5/2 correlation model
        const MATERN52 = 0x08;
        /// All correlation models available
        const ALL = CorrelationSpec::SQUAREDEXPONENTIAL.bits
                    | CorrelationSpec::ABSOLUTEEXPONENTIAL.bits
                    | CorrelationSpec::MATERN32.bits
                    | CorrelationSpec::MATERN52.bits;
    }
}

/// A trait for mixture of experts predictor.
pub trait Expert {
    /// Predict values at a given set of points `x` defined as (n, xdim) matrix
    fn predict_values(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
    /// Predict variances at a given set of points `x` defined as (n, xdim) matrix
    fn predict_variances(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
}

impl<T: Surrogate> Expert for T {
    fn predict_values(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_values(&x.view())
    }

    fn predict_variances(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_variances(&x.view())
    }
}

/// A trait for mixture of experts predictor construction (model fitting)
pub trait MoeFit {
    /// Train the Moe models with given training dataset (x, y)
    fn fit_for_predict(&self, xt: &Array2<f64>, yt: &Array2<f64>) -> Result<Box<dyn Expert>>;
}

/// Mixture of experts parameters
#[derive(Clone)]
pub struct MoeParams<F: Float, R: Rng + Clone> {
    /// Number of clusters (i.e. number of experts)
    n_clusters: usize,
    /// [Recombination] mode
    recombination: Recombination<F>,
    /// Specification of GP regression models to be used
    regression_spec: RegressionSpec,
    /// Specification of GP correlation models to be used
    correlation_spec: CorrelationSpec,
    /// Number of PLS components, should be used when problem size
    /// is over ten variables or so.
    kpls_dim: Option<usize>,
    /// Gaussian Mixture model used to
    gmm: Option<Box<GaussianMixtureModel<F>>>,
    /// Random number generator
    rng: R,
}

impl<F: Float> MoeParams<F, Isaac64Rng> {
    /// Constructor of Moe parameters with `n_clusters`.
    ///
    /// Default values are provided as follows:
    ///
    /// * recombination: `Smooth`
    /// * regression_spec: `ALL`
    /// * correlation_spec: `ALL`
    /// * kpls_dim: `None`
    #[allow(clippy::new_ret_no_self)]
    pub fn new(n_clusters: usize) -> MoeParams<F, Isaac64Rng> {
        Self::new_with_rng(n_clusters, Isaac64Rng::from_entropy())
    }
}

impl<F: Float> Default for MoeParams<F, Isaac64Rng> {
    fn default() -> MoeParams<F, Isaac64Rng> {
        MoeParams::new(1)
    }
}

impl<F: Float, R: Rng + Clone> MoeParams<F, R> {
    /// Constructor of Moe parameters specifying randon number generator for reproducibility
    ///
    /// See [MoeParams::new] for default parameters.
    pub fn new_with_rng(n_clusters: usize, rng: R) -> MoeParams<F, R> {
        MoeParams {
            n_clusters,
            recombination: Recombination::Smooth(Some(F::one())),
            regression_spec: RegressionSpec::ALL,
            correlation_spec: CorrelationSpec::ALL,
            kpls_dim: None,
            gmm: None,
            rng,
        }
    }

    /// The number of clusters, hence the number of experts of the mixture.
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// The recombination mode
    pub fn recombination(&self) -> Recombination<F> {
        self.recombination
    }

    /// The allowed GP regression models in the mixture
    pub fn regression_spec(&self) -> RegressionSpec {
        self.regression_spec
    }

    /// The allowed GP correlation models in the mixture
    pub fn correlation_spec(&self) -> CorrelationSpec {
        self.correlation_spec
    }

    /// The optional number of PLS components
    pub fn kpls_dim(&self) -> Option<usize> {
        self.kpls_dim
    }

    /// The random generator
    pub fn rng(&self) -> R {
        self.rng.clone()
    }

    /// Sets the number of clusters
    pub fn set_nclusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    /// Sets the recombination mode
    pub fn set_recombination(mut self, recombination: Recombination<F>) -> Self {
        self.recombination = recombination;
        self
    }

    /// Sets the regression models used in the mixture.
    ///
    /// Only GP models with regression models allowed by this specification
    /// will be used in the mixture.  
    pub fn set_regression_spec(mut self, regression_spec: RegressionSpec) -> Self {
        self.regression_spec = regression_spec;
        self
    }

    /// Sets the correlation models used in the mixture.
    ///
    /// Only GP models with correlation models allowed by this specification
    /// will be used in the mixture.  
    pub fn set_correlation_spec(mut self, correlation_spec: CorrelationSpec) -> Self {
        self.correlation_spec = correlation_spec;
        self
    }

    /// Sets the number of PLS components in [1, nx]  where nx is the x dimension
    ///
    /// None means no PLS dimension reduction applied.
    pub fn set_kpls_dim(mut self, kpls_dim: Option<usize>) -> Self {
        self.kpls_dim = kpls_dim;
        self
    }

    #[doc(hidden)]
    /// Sets the gaussian mixture (used to find the optimal number of clusters)
    pub(crate) fn set_gmm(mut self, gmm: Option<Box<GaussianMixtureModel<F>>>) -> Self {
        self.gmm = gmm;
        self
    }

    /// Sets the random number generator for reproducibility
    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> MoeParams<F, R2> {
        MoeParams {
            n_clusters: self.n_clusters,
            recombination: self.recombination,
            regression_spec: self.regression_spec,
            correlation_spec: self.correlation_spec,
            kpls_dim: self.kpls_dim,
            gmm: self.gmm,
            rng,
        }
    }
}
