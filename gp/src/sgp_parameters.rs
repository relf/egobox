use crate::correlation_models::{CorrelationModel, SquaredExponentialCorr};
use crate::errors::{GpError, Result};
use crate::mean_models::ConstantMean;
use crate::parameters::GpValidParams;
use linfa::{Float, ParamGuard};
use ndarray::Array2;
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VarianceConfig<F: Float> {
    /// Constant variance
    Constant(F),
    /// Variance is optimized between given bounds (lower, upper) starting from the inital guess
    Estimated { initial_guess: F, bounds: (F, F) },
}
impl<F: Float> Default for VarianceConfig<F> {
    fn default() -> VarianceConfig<F> {
        Self::Constant(F::cast(0.01))
    }
}

/// SGP inducing points specification
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub enum Inducings<F: Float> {
    /// usize points are selected randomly in the training dataset
    Randomized(usize),
    /// Points are given as a (npoints, nx) matrix
    Located(Array2<F>),
}
impl<F: Float> Default for Inducings<F> {
    fn default() -> Inducings<F> {
        Self::Randomized(10)
    }
}

/// SGP algorithm method specification
#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub enum SparseMethod {
    #[default]
    /// Fully Independent Training Conditional method
    Fitc,
    /// Variational Free Energy method
    Vfe,
}

/// A set of validated SGP parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SgpValidParams<F: Float, Corr: CorrelationModel<F>> {
    /// gp
    gp_params: GpValidParams<F, ConstantMean, Corr>,
    /// Gaussian homeoscedastic noise variance
    noise: VarianceConfig<F>,
    /// Inducing points
    z: Inducings<F>,
    /// Method
    method: SparseMethod,
    /// Random generator seed
    seed: Option<u64>,
}

impl<F: Float> Default for SgpValidParams<F, SquaredExponentialCorr> {
    fn default() -> SgpValidParams<F, SquaredExponentialCorr> {
        SgpValidParams {
            gp_params: GpValidParams::default(),
            noise: VarianceConfig::default(),
            z: Inducings::default(),
            method: SparseMethod::default(),
            seed: None,
        }
    }
}

impl<F: Float, Corr: CorrelationModel<F>> SgpValidParams<F, Corr> {
    /// Get starting theta value for optimization
    pub fn initial_theta(&self) -> &Option<Vec<F>> {
        &self.gp_params.theta
    }

    /// Get correlation corr k(x, x')
    pub fn corr(&self) -> &Corr {
        &self.gp_params.corr
    }

    /// Get number of components used by PLS
    pub fn kpls_dim(&self) -> &Option<usize> {
        &self.gp_params.kpls_dim
    }

    /// Get number of components used by PLS
    pub fn nugget(&self) -> F {
        self.gp_params.nugget
    }

    /// Get used sparse method
    pub fn method(&self) -> &SparseMethod {
        &self.method
    }

    /// Get inducing points
    pub fn inducings(&self) -> &Inducings<F> {
        &self.z
    }

    /// Get noise variance configuration
    pub fn noise_variance(&self) -> &VarianceConfig<F> {
        &self.noise
    }

    /// Get seed
    pub fn seed(&self) -> &Option<u64> {
        &self.seed
    }
}

#[derive(Clone, Debug)]
/// The set of hyperparameters that can be specified for the execution of
/// the [SGP algorithm](struct.SparseGaussianProcess.html).
pub struct SgpParams<F: Float, Corr: CorrelationModel<F>>(SgpValidParams<F, Corr>);

impl<F: Float, Corr: CorrelationModel<F>> SgpParams<F, Corr> {
    /// A constructor for SGP parameters given mean and correlation models
    pub fn new(corr: Corr) -> SgpParams<F, Corr> {
        Self(SgpValidParams {
            gp_params: GpValidParams {
                theta: None,
                mean: ConstantMean::default(),
                corr,
                kpls_dim: None,
                nugget: F::cast(1000.0) * F::epsilon(),
            },
            noise: VarianceConfig::default(),
            z: Inducings::default(),
            method: SparseMethod::default(),
            seed: None,
        })
    }

    /// Set initial value for theta hyper parameter.
    ///
    /// During training process, the internal optimization is started from `initial_theta`.
    pub fn initial_theta(mut self, theta: Option<Vec<F>>) -> Self {
        self.0.gp_params.theta = theta;
        self
    }

    /// Set correlation model.
    pub fn corr(mut self, corr: Corr) -> Self {
        self.0.gp_params.corr = corr;
        self
    }

    /// Set number of PLS components.
    ///
    /// Should be 0 < n < pb size (i.e. x dimension)
    pub fn kpls_dim(mut self, kpls_dim: Option<usize>) -> Self {
        self.0.gp_params.kpls_dim = kpls_dim;
        self
    }

    /// Set nugget value.
    ///
    /// Nugget is used to improve numerical stability
    pub fn nugget(mut self, nugget: F) -> Self {
        self.0.gp_params.nugget = nugget;
        self
    }

    /// Specify nz inducing points as (nz, x_dim) matrix.
    pub fn inducings(mut self, z: Array2<F>) -> Self {
        self.0.z = Inducings::Located(z);
        self
    }

    /// Specify nz number of inducing points which will be picked randomly in the input training dataset.
    pub fn n_inducings(mut self, nz: usize) -> Self {
        self.0.z = Inducings::Randomized(nz);
        self
    }

    /// Set noise variance configuration defining noise handling.
    pub fn noise_variance(mut self, config: VarianceConfig<F>) -> Self {
        self.0.noise = config;
        self
    }

    /// Set noise variance configuration defining noise handling.
    pub fn seed(mut self, seed: Option<u64>) -> Self {
        self.0.seed = seed;
        self
    }
}

impl<F: Float, Corr: CorrelationModel<F>> ParamGuard for SgpParams<F, Corr> {
    type Checked = SgpValidParams<F, Corr>;
    type Error = GpError;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if let Some(d) = self.0.gp_params.kpls_dim {
            if d == 0 {
                return Err(GpError::InvalidValue("`kpls_dim` canot be 0!".to_string()));
            }
            if let Some(theta) = self.0.initial_theta() {
                if theta.len() > 1 && d > theta.len() {
                    return Err(GpError::InvalidValue(format!(
                        "Dimension reduction ({}) should be smaller than expected
                        training input size infered from given initial theta length ({})",
                        d,
                        theta.len()
                    )));
                };
            }
        }
        Ok(&self.0)
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}
