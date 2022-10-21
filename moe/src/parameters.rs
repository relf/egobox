use crate::errors::{MoeError, Result};
use crate::gaussian_mixture::GaussianMixture;
use bitflags::bitflags;
#[allow(unused_imports)]
use egobox_gp::correlation_models::{
    AbsoluteExponentialCorr, Matern32Corr, Matern52Corr, SquaredExponentialCorr,
};
#[allow(unused_imports)]
use egobox_gp::mean_models::{ConstantMean, LinearMean, QuadraticMean};
use linfa::{Float, ParamGuard};
use linfa_clustering::GaussianMixtureModel;
use ndarray::{Array1, Array2, Array3};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;
use std::fmt::Display;

#[cfg(feature = "persistent")]
use serde::{Deserialize, Serialize};

/// Enumeration of recombination modes handled by the mixture
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "persistent", derive(Serialize, Deserialize))]
pub enum Recombination<F: Float> {
    /// prediction is taken from the expert with highest responsability
    /// resulting in a model with discontinuities
    Hard,
    /// Prediction is a combination experts prediction wrt their responsabilities,
    /// an optional heaviside factor might be used control steepness of the change between
    /// experts regions.
    Smooth(Option<F>),
}

impl<F: Float> Display for Recombination<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let recomb = match self {
            Recombination::Hard => "Hard".to_string(),
            Recombination::Smooth(Some(f)) => format!("Smooth({})", f),
            Recombination::Smooth(None) => "Smooth".to_string(),
        };
        write!(f, "Mixture[{}]", &recomb)
    }
}

bitflags! {
    /// Flags to specify tested regression models during experts selection (see [MoeParams::regression_spec]).
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

/// Mixture of experts checked parameters
#[derive(Clone)]
pub struct MoeValidParams<F: Float, R: Rng + Clone> {
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
    /// Gaussian Mixture model used to cluster
    gmm: Option<Box<GaussianMixtureModel<F>>>,
    /// GaussianMixture preset
    gmx: Option<Box<GaussianMixture<F>>>,
    /// Random number generator
    rng: R,
}

impl<F: Float> Default for MoeValidParams<F, Isaac64Rng> {
    fn default() -> MoeValidParams<F, Isaac64Rng> {
        MoeValidParams {
            n_clusters: 1,
            recombination: Recombination::Smooth(Some(F::one())),
            regression_spec: RegressionSpec::ALL,
            correlation_spec: CorrelationSpec::ALL,
            kpls_dim: None,
            gmm: None,
            gmx: None,
            rng: Isaac64Rng::from_entropy(),
        }
    }
}

impl<F: Float, R: Rng + Clone> MoeValidParams<F, R> {
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

    /// An optional gaussian mixture to be fitted to generate multivariate normal
    /// in turns used to cluster
    pub fn gmm(&self) -> &Option<Box<GaussianMixtureModel<F>>> {
        &self.gmm
    }

    /// An optional multivariate normal used to cluster (take precedence over gmm)
    pub fn gmx(&self) -> &Option<Box<GaussianMixture<F>>> {
        &self.gmx
    }

    /// The random generator
    pub fn rng(&self) -> R {
        self.rng.clone()
    }
}

/// Mixture of experts parameters
#[derive(Clone)]
pub struct MoeParams<F: Float, R: Rng + Clone>(MoeValidParams<F, R>);

impl<F: Float> Default for MoeParams<F, Isaac64Rng> {
    fn default() -> MoeParams<F, Isaac64Rng> {
        MoeParams(MoeValidParams::default())
    }
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
    pub fn new() -> MoeParams<F, Isaac64Rng> {
        Self::new_with_rng(Isaac64Rng::from_entropy())
    }
}

impl<F: Float, R: Rng + Clone> MoeParams<F, R> {
    /// Constructor of Moe parameters specifying randon number generator for reproducibility
    ///
    /// See [MoeParams::new] for default parameters.
    pub fn new_with_rng(rng: R) -> MoeParams<F, R> {
        Self(MoeValidParams {
            n_clusters: 1,
            recombination: Recombination::Smooth(Some(F::one())),
            regression_spec: RegressionSpec::ALL,
            correlation_spec: CorrelationSpec::ALL,
            kpls_dim: None,
            gmm: None,
            gmx: None,
            rng,
        })
    }

    /// Sets the number of clusters
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.0.n_clusters = n_clusters;
        self
    }

    /// Sets the recombination mode
    pub fn recombination(mut self, recombination: Recombination<F>) -> Self {
        self.0.recombination = recombination;
        self
    }

    /// Sets the regression models used in the mixture.
    ///
    /// Only GP models with regression models allowed by this specification
    /// will be used in the mixture.  
    pub fn regression_spec(mut self, regression_spec: RegressionSpec) -> Self {
        self.0.regression_spec = regression_spec;
        self
    }

    /// Sets the correlation models used in the mixture.
    ///
    /// Only GP models with correlation models allowed by this specification
    /// will be used in the mixture.  
    pub fn correlation_spec(mut self, correlation_spec: CorrelationSpec) -> Self {
        self.0.correlation_spec = correlation_spec;
        self
    }

    /// Sets the number of PLS components in [1, nx]  where nx is the x dimension
    ///
    /// None means no PLS dimension reduction applied.
    pub fn kpls_dim(mut self, kpls_dim: Option<usize>) -> Self {
        self.0.kpls_dim = kpls_dim;
        self
    }

    #[doc(hidden)]
    /// Sets the gaussian mixture (used to find the optimal number of clusters)
    pub(crate) fn gmm(mut self, gmm: Option<Box<GaussianMixtureModel<F>>>) -> Self {
        self.0.gmm = gmm;
        self
    }

    #[doc(hidden)]
    /// Sets the gaussian mixture (used to find the optimal number of clusters)
    /// Warning: no consistency check is done on the given initialization data
    /// *Panic* if multivariate normal init data not sound
    pub fn gmx(mut self, weights: Array1<F>, means: Array2<F>, covariances: Array3<F>) -> Self {
        self.0.gmx = Some(Box::new(
            GaussianMixture::new(weights, means, covariances).unwrap(),
        ));
        self
    }

    /// Sets the random number generator for reproducibility
    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> MoeParams<F, R2> {
        MoeParams(MoeValidParams {
            n_clusters: self.0.n_clusters(),
            recombination: self.0.recombination(),
            regression_spec: self.0.regression_spec(),
            correlation_spec: self.0.correlation_spec(),
            kpls_dim: self.0.kpls_dim(),
            gmm: self.0.gmm().clone(),
            gmx: self.0.gmx().clone(),
            rng,
        })
    }
}

impl<F: Float, R: Rng + Clone> ParamGuard for MoeParams<F, R> {
    type Checked = MoeValidParams<F, R>;
    type Error = MoeError;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if let Some(d) = self.0.kpls_dim {
            if d == 0 {
                return Err(MoeError::InvalidValueError(
                    "`kpls_dim` canot be 0!".to_string(),
                ));
            }
        }
        Ok(&self.0)
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}

impl<F: Float, R: Rng + Clone> From<MoeValidParams<F, R>> for MoeParams<F, R> {
    fn from(item: MoeValidParams<F, R>) -> Self {
        MoeParams(item)
    }
}
