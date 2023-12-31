use crate::correlation_models::{CorrelationModel, SquaredExponentialCorr};
use crate::errors::{GpError, Result};
use crate::mean_models::{ConstantMean, RegressionModel};
use crate::parameters::GpValidParams;
use linfa::{Float, ParamGuard};
use ndarray::Array2;
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NoiseVarianceConfig<F: Float> {
    Constant(F),
    Estimated {
        initial_guess: F,
        lower_bound: F,
        upper_bound: F,
    },
}
impl<F: Float> Default for NoiseVarianceConfig<F> {
    fn default() -> NoiseVarianceConfig<F> {
        Self::Constant(F::one())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub enum Inducings<F: Float> {
    Randomized(usize),
    Located(Array2<F>),
}
impl<F: Float> Default for Inducings<F> {
    fn default() -> Inducings<F> {
        Self::Randomized(10)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub enum SparseMethod {
    #[default]
    Fitc,
    Vfe,
}

/// A set of validated SGP parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SgpValidParams<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> {
    /// gp
    gp_params: GpValidParams<F, Mean, Corr>,
    /// Gaussian homeoscedastic noise variance
    noise: NoiseVarianceConfig<F>,
    /// Inducing points
    z: Inducings<F>,
    /// Method
    method: SparseMethod,
}

impl<F: Float> Default for SgpValidParams<F, ConstantMean, SquaredExponentialCorr> {
    fn default() -> SgpValidParams<F, ConstantMean, SquaredExponentialCorr> {
        SgpValidParams {
            gp_params: GpValidParams::default(),
            noise: NoiseVarianceConfig::default(),
            z: Inducings::default(),
            method: SparseMethod::default(),
        }
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> SgpValidParams<F, Mean, Corr> {
    /// Get starting theta value for optimization
    pub fn initial_theta(&self) -> &Option<Vec<F>> {
        &self.gp_params.theta
    }

    /// Get mean model  
    pub fn mean(&self) -> &Mean {
        &self.gp_params.mean
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
}

#[derive(Clone, Debug)]
/// The set of hyperparameters that can be specified for the execution of
/// the [GP algorithm](struct.GaussianProcess.html).
pub struct SgpParams<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>>(
    SgpValidParams<F, Mean, Corr>,
);

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> SgpParams<F, Mean, Corr> {
    /// A constructor for GP parameters given mean and correlation models
    pub fn new(mean: Mean, corr: Corr) -> SgpParams<F, Mean, Corr> {
        Self(SgpValidParams {
            gp_params: GpValidParams {
                theta: None,
                mean,
                corr,
                kpls_dim: None,
                nugget: F::cast(1000.0) * F::epsilon(),
            },
            noise: NoiseVarianceConfig::default(),
            z: Inducings::default(),
            method: SparseMethod::default(),
        })
    }

    /// Set initial value for theta hyper parameter.
    ///
    /// During training process, the internal optimization is started from `initial_theta`.
    pub fn initial_theta(mut self, theta: Option<Vec<F>>) -> Self {
        self.0.gp_params.theta = theta;
        self
    }

    /// Set mean model.
    pub fn mean(mut self, mean: Mean) -> Self {
        self.0.gp_params.mean = mean;
        self
    }

    /// Set correlation model.
    pub fn corr(mut self, corr: Corr) -> Self {
        self.0.gp_params.corr = corr;
        self
    }

    /// Set number of PLS components.
    /// Should be 0 < n < pb size (i.e. x dimension)
    pub fn kpls_dim(mut self, kpls_dim: Option<usize>) -> Self {
        self.0.gp_params.kpls_dim = kpls_dim;
        self
    }

    /// Set nugget.
    ///
    /// Nugget is used to improve numerical stability
    pub fn nugget(mut self, nugget: F) -> Self {
        self.0.gp_params.nugget = nugget;
        self
    }

    pub fn inducings(mut self, z: Array2<F>) -> Self {
        self.0.z = Inducings::Located(z);
        self
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> ParamGuard
    for SgpParams<F, Mean, Corr>
{
    type Checked = SgpValidParams<F, Mean, Corr>;
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
