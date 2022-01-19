use crate::correlation_models::{CorrelationModel, SquaredExponentialKernel};
use crate::errors::{GpError, Result};
use crate::mean_models::{ConstantMean, RegressionModel};
use linfa::{Float, ParamGuard};

#[derive(Clone, Debug, PartialEq)]
pub struct GpValidParams<F: Float, Mean: RegressionModel<F>, Kernel: CorrelationModel<F>> {
    /// Parameter of the autocorrelation model
    theta: Option<Vec<F>>,
    /// Regression model representing the mean(x)
    mean: Mean,
    /// Correlation model representing the spatial correlation between errors at e(x) and e(x')
    kernel: Kernel,
    /// Optionally apply dimension reduction (KPLS) or not
    kpls_dim: Option<usize>,
    /// Optionally apply dimension reduction (KPLS) or not
    nugget: F,
}

impl<F: Float> GpValidParams<F, ConstantMean, SquaredExponentialKernel> {
    pub fn default() -> GpValidParams<F, ConstantMean, SquaredExponentialKernel> {
        GpValidParams {
            theta: None,
            mean: ConstantMean(),
            kernel: SquaredExponentialKernel(),
            kpls_dim: None,
            nugget: F::cast(100.0) * F::epsilon(),
        }
    }
}

impl<F: Float, Mean: RegressionModel<F>, Kernel: CorrelationModel<F>>
    GpValidParams<F, Mean, Kernel>
{
    /// Get starting theta value for optimization
    pub fn initial_theta(&self) -> &Option<Vec<F>> {
        &self.theta
    }

    /// Get mean model  
    pub fn mean(&self) -> &Mean {
        &self.mean
    }

    /// Get correlation kernel k(x, x')
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
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
pub struct GpParams<F: Float, Mean: RegressionModel<F>, Kernel: CorrelationModel<F>>(
    GpValidParams<F, Mean, Kernel>,
);

impl<F: Float, Mean: RegressionModel<F>, Kernel: CorrelationModel<F>> GpParams<F, Mean, Kernel> {
    pub fn new(mean: Mean, kernel: Kernel) -> GpParams<F, Mean, Kernel> {
        Self(GpValidParams {
            theta: None,
            mean,
            kernel,
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

    /// Set mean.
    pub fn mean(mut self, mean: Mean) -> Self {
        self.0.mean = mean;
        self
    }

    /// Set kernel.
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.0.kernel = kernel;
        self
    }

    /// Set KPLS.
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

impl<F: Float, Mean: RegressionModel<F>, Kernel: CorrelationModel<F>> ParamGuard
    for GpParams<F, Mean, Kernel>
{
    type Checked = GpValidParams<F, Mean, Kernel>;
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
