use crate::correlation_models::CorrelationModel;
use crate::errors::{GpError, Result};
use crate::mean_models::RegressionModel;
use linfa::dataset::Float;

#[derive(Clone, Copy)]
pub struct GpParams<F: Float, Mean: RegressionModel<F>, Kernel: CorrelationModel<F>> {
    /// Parameter of the autocorrelation model
    theta: F,
    /// Regression model representing the mean(x)
    mean: Mean,
    /// Correlation model representing the spatial correlation between errors at e(x) and e(x')
    kernel: Kernel,
    /// Optionally apply dimension reduction (KPLS) or not
    kpls_dim: Option<usize>,
    /// Optionally apply dimension reduction (KPLS) or not
    nugget: F,
}

impl<F: Float, Mean: RegressionModel<F>, Kernel: CorrelationModel<F>> GpParams<F, Mean, Kernel> {
    pub fn new(mean: Mean, kernel: Kernel) -> GpParams<F, Mean, Kernel> {
        GpParams {
            theta: F::cast(1e-2),
            mean,
            kernel,
            kpls_dim: None,
            nugget: F::cast(100.0) * F::epsilon(),
        }
    }

    /// Get starting theta value for optimization
    pub fn initial_theta(&self) -> F {
        self.theta
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

    /// Set initial value for theta hyper parameter.
    ///
    /// During training process, the internal optimization is started from `initial_theta`.
    pub fn set_initial_theta(mut self, theta: F) -> Self {
        self.theta = theta;
        self
    }

    /// Set mean.
    pub fn with_mean(mut self, mean: Mean) -> Self {
        self.mean = mean;
        self
    }

    /// Set kernel.
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set KPLS.
    pub fn set_kpls_dim(mut self, kpls_dim: Option<usize>) -> Self {
        self.kpls_dim = kpls_dim;
        self
    }

    /// Set nugget.
    ///
    /// Nugget is used to improve numerical stability
    pub fn set_nugget(mut self, nugget: F) -> Self {
        self.nugget = nugget;
        self
    }

    /// Check Gp params consistency
    pub fn validate(&self, xdim: usize) -> Result<()> {
        if let Some(d) = self.kpls_dim {
            if d > xdim {
                return Err(GpError::InvalidValue(format!(
                    "Dimension reduction {} should be smaller than actual \
                    training input dimensions {}",
                    d, xdim
                )));
            };
        }
        Ok(())
    }
}
