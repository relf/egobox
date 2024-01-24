use crate::errors::{MoeError, Result};
use crate::gaussian_mixture::GaussianMixture;
use crate::types::*;

#[allow(unused_imports)]
use egobox_gp::correlation_models::{
    AbsoluteExponentialCorr, Matern32Corr, Matern52Corr, SquaredExponentialCorr,
};
#[allow(unused_imports)]
use egobox_gp::mean_models::{ConstantMean, LinearMean, QuadraticMean};
use egobox_gp::Inducings;
use linfa::{Float, ParamGuard};
use linfa_clustering::GaussianMixtureModel;
use ndarray::{Array1, Array2, Array3};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;

#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

/// Mixture of experts checked parameters
#[derive(Clone)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct SgpValidParams<F: Float, R: Rng + Clone> {
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
    /// Inducings
    inducings: Inducings<F>,
    /// Gaussian Mixture model used to cluster
    gmm: Option<Box<GaussianMixtureModel<F>>>,
    /// GaussianMixture preset
    gmx: Option<Box<GaussianMixture<F>>>,
    /// Random number generator
    rng: R,
}

impl<F: Float, R: Rng + SeedableRng + Clone> Default for SgpValidParams<F, R> {
    fn default() -> SgpValidParams<F, R> {
        SgpValidParams {
            n_clusters: 1,
            recombination: Recombination::Smooth(Some(F::one())),
            regression_spec: RegressionSpec::CONSTANT,
            correlation_spec: CorrelationSpec::SQUAREDEXPONENTIAL,
            kpls_dim: None,
            inducings: Inducings::default(),
            gmm: None,
            gmx: None,
            rng: R::from_entropy(),
        }
    }
}

impl<F: Float, R: Rng + Clone> SgpValidParams<F, R> {
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

    /// Inducings points specification
    pub fn inducings(&self) -> &Inducings<F> {
        &self.inducings
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
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct SgpParams<F: Float, R: Rng + Clone>(SgpValidParams<F, R>);

impl<F: Float> Default for SgpParams<F, Xoshiro256Plus> {
    fn default() -> SgpParams<F, Xoshiro256Plus> {
        SgpParams(SgpValidParams::default())
    }
}

impl<F: Float> SgpParams<F, Xoshiro256Plus> {
    /// Constructor of Sgp parameters with `n_clusters`.
    ///
    /// Default values are provided as follows:
    ///
    /// * recombination: `Smooth`
    /// * correlation_spec: `ALL`
    /// * kpls_dim: `None`
    #[allow(clippy::new_ret_no_self)]
    pub fn new(inducings: Inducings<F>) -> SgpParams<F, Xoshiro256Plus> {
        Self::new_with_rng(inducings, Xoshiro256Plus::from_entropy())
    }
}

impl<F: Float, R: Rng + Clone> SgpParams<F, R> {
    /// Constructor of Sgp parameters specifying randon number generator for reproducibility
    ///
    /// See [SgpParams::new] for default parameters.
    pub fn new_with_rng(inducings: Inducings<F>, rng: R) -> SgpParams<F, R> {
        Self(SgpValidParams {
            n_clusters: 1,
            recombination: Recombination::Smooth(Some(F::one())),
            regression_spec: RegressionSpec::CONSTANT,
            correlation_spec: CorrelationSpec::SQUAREDEXPONENTIAL,
            kpls_dim: None,
            inducings,
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

    /// Sets the number of PLS components in [1, nx]  where nx is the x dimension
    ///
    /// None means no PLS dimension reduction applied.
    pub fn inducings(mut self, inducings: Inducings<F>) -> Self {
        self.0.inducings = inducings;
        self
    }

    // #[doc(hidden)]
    // /// Sets the gaussian mixture (used to find the optimal number of clusters)
    // pub(crate) fn gmm(mut self, gmm: Option<Box<GaussianMixtureModel<F>>>) -> Self {
    //     self.0.gmm = gmm;
    //     self
    // }

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
    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> SgpParams<F, R2> {
        SgpParams(SgpValidParams {
            n_clusters: self.0.n_clusters(),
            recombination: self.0.recombination(),
            regression_spec: self.0.regression_spec(),
            correlation_spec: self.0.correlation_spec(),
            kpls_dim: self.0.kpls_dim(),
            inducings: self.0.inducings().clone(),
            gmm: self.0.gmm().clone(),
            gmx: self.0.gmx().clone(),
            rng,
        })
    }
}

impl<F: Float, R: Rng + Clone> ParamGuard for SgpParams<F, R> {
    type Checked = SgpValidParams<F, R>;
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

impl<F: Float, R: Rng + Clone> From<SgpValidParams<F, R>> for SgpParams<F, R> {
    fn from(item: SgpValidParams<F, R>) -> Self {
        SgpParams(item)
    }
}
