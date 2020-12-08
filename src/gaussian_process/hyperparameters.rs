use crate::gaussian_process::{CorrelationModel, RegressionModel};
use ndarray::Array2;

#[derive(Clone)]
pub struct GpHyperParams<Mean: RegressionModel, Kernel: CorrelationModel> {
    /// Parameter of the autocorrelation model
    theta: f64,
    /// Regression model representing the mean(x)
    mean: Mean,
    /// Correlation model representing the spatial correlation between errors at e(x) and e(x')
    kernel: Kernel,
    /// Training inputs
    xtrain: Array2<f64>,
    /// Training outputs
    ytrain: Array2<f64>,
}

impl<Mean: RegressionModel, Kernel: CorrelationModel> GpHyperParams<Mean, Kernel> {
    pub fn new(mean: Mean, kernel: Kernel) -> GpHyperParams<Mean, Kernel> {
        GpHyperParams {
            theta: 1e-2,
            mean,
            kernel,
            xtrain: Array2::default((1, 1)),
            ytrain: Array2::default((1, 1)),
        }
    }

    /// Get starting theta value for optimization
    pub fn initial_theta(&self) -> f64 {
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

    /// Set initial value for theta hyper parameter.
    ///
    /// During training process, the internal optimization is started from `initial_theta`.
    pub fn with_initial_theta(mut self, theta: f64) -> Self {
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
}
