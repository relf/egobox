use crate::correlation_models::{CorrelationModel, SquaredExponentialCorr};
use crate::errors::{GpError, Result};
use crate::mean_models::{ConstantMean, RegressionModel};
use linfa::{Float, ParamGuard};

/// A set of validated GP parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpValidParams<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> {
    /// Parameter of the autocorrelation model
    theta: Option<Vec<F>>,
    /// Regression model representing the mean(x)
    mean: Mean,
    /// Correlation model representing the spatial correlation between errors at e(x) and e(x')
    corr: Corr,
    /// Optionally apply dimension reduction (KPLS) or not
    kpls_dim: Option<usize>,
    /// Parameter to improve numerical stability
    nugget: F,
}

impl<F: Float> Default for GpValidParams<F, ConstantMean, SquaredExponentialCorr> {
    fn default() -> GpValidParams<F, ConstantMean, SquaredExponentialCorr> {
        GpValidParams {
            theta: None,
            mean: ConstantMean(),
            corr: SquaredExponentialCorr(),
            kpls_dim: None,
            nugget: F::cast(100.0) * F::epsilon(),
        }
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> GpValidParams<F, Mean, Corr> {
    /// Get starting theta value for optimization
    pub fn initial_theta(&self) -> &Option<Vec<F>> {
        &self.theta
    }

    /// Get mean model  
    pub fn mean(&self) -> &Mean {
        &self.mean
    }

    /// Get correlation corr k(x, x')
    pub fn corr(&self) -> &Corr {
        &self.corr
    }

    /// Get number of components used by PLS
    pub fn kpls_dim(&self) -> &Option<usize> {
        &self.kpls_dim
    }

    /// Get number of components used by PLS
    pub fn nugget(&self) -> F {
        self.nugget
    }
}

#[derive(Clone, Debug)]
/// The set of hyperparameters that can be specified for the execution of
/// the [GMM algorithm](struct.GaussianMixtureModel.html).
pub struct GpParams<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>>(
    GpValidParams<F, Mean, Corr>,
);

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> GpParams<F, Mean, Corr> {
    /// A constructor for GP parameters given mean and correlation models
    pub fn new(mean: Mean, corr: Corr) -> GpParams<F, Mean, Corr> {
        Self(GpValidParams {
            theta: None,
            mean,
            corr,
            kpls_dim: None,
            nugget: F::cast(100.0) * F::epsilon(),
        })
    }

    /// Set initial value for theta hyper parameter.
    ///
    /// During training process, the internal optimization is started from `initial_theta`.
    pub fn initial_theta(mut self, theta: Option<Vec<F>>) -> Self {
        self.0.theta = theta;
        self
    }

    /// Set mean model.
    pub fn mean(mut self, mean: Mean) -> Self {
        self.0.mean = mean;
        self
    }

    /// Set correlation model.
    pub fn corr(mut self, corr: Corr) -> Self {
        self.0.corr = corr;
        self
    }

    /// Set number of PLS components.
    /// Should be 0 < n < pb size (i.e. x dimension)
    pub fn kpls_dim(mut self, kpls_dim: Option<usize>) -> Self {
        self.0.kpls_dim = kpls_dim;
        self
    }

    /// Set nugget.
    ///
    /// Nugget is used to improve numerical stability
    pub fn nugget(mut self, nugget: F) -> Self {
        self.0.nugget = nugget;
        self
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> ParamGuard
    for GpParams<F, Mean, Corr>
{
    type Checked = GpValidParams<F, Mean, Corr>;
    type Error = GpError;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if let Some(d) = self.0.kpls_dim {
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
