use crate::correlation_models::{CorrelationModel, SquaredExponentialCorr};
use crate::errors::{GpError, Result};
use crate::mean_models::ConstantMean;
use crate::parameters::GpValidParams;
use crate::{ParamTuning, ThetaTuning};
use linfa::{Float, ParamGuard};
use ndarray::Array2;
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

/// Variance estimation method
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamEstimation<F: Float> {
    /// Constant parameter (ie given not estimated)
    Fixed(F),
    /// Parameter is estimated between given bounds (lower, upper) starting from the inital guess
    Estimated { initial_guess: F, bounds: (F, F) },
}
impl<F: Float> Default for ParamEstimation<F> {
    fn default() -> ParamEstimation<F> {
        Self::Estimated {
            initial_guess: F::cast(1e-2),
            bounds: (F::cast(100.0) * F::epsilon(), F::cast(1e10)),
        }
    }
}

/// SGP inducing points specification
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum Inducings<F: Float> {
    /// `usize` points are selected randomly in the training dataset
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
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
    noise: ParamEstimation<F>,
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
            noise: ParamEstimation::default(),
            z: Inducings::default(),
            method: SparseMethod::default(),
            seed: None,
        }
    }
}

impl<F: Float, Corr: CorrelationModel<F>> SgpValidParams<F, Corr> {
    /// Get correlation corr k(x, x')
    pub fn corr(&self) -> &Corr {
        &self.gp_params.corr
    }

    /// Get the number of components used by PLS
    pub fn kpls_dim(&self) -> Option<&usize> {
        self.gp_params.kpls_dim.as_ref()
    }

    /// Get starting theta value for optimization
    pub fn theta_tuning(&self) -> &ThetaTuning<F> {
        &self.gp_params.theta_tuning
    }

    /// Get the number of internal GP hyperparameters optimization restart
    pub fn n_start(&self) -> usize {
        self.gp_params.n_start
    }

    /// Get number of components used by PLS
    pub fn nugget(&self) -> F {
        self.gp_params.nugget
    }

    /// Get used sparse method
    pub fn method(&self) -> SparseMethod {
        self.method
    }

    /// Get inducing points
    pub fn inducings(&self) -> &Inducings<F> {
        &self.z
    }

    /// Get noise variance configuration
    pub fn noise_variance(&self) -> &ParamEstimation<F> {
        &self.noise
    }

    /// Get seed
    pub fn seed(&self) -> Option<&u64> {
        self.seed.as_ref()
    }
}

#[derive(Clone, Debug)]
/// The set of hyperparameters that can be specified for the execution of
/// the [SGP algorithm](struct.SparseGaussianProcess.html).
pub struct SgpParams<F: Float, Corr: CorrelationModel<F>>(SgpValidParams<F, Corr>);

impl<F: Float, Corr: CorrelationModel<F>> SgpParams<F, Corr> {
    /// A constructor for SGP parameters given mean and correlation models
    pub fn new(corr: Corr, inducings: Inducings<F>) -> SgpParams<F, Corr> {
        Self(SgpValidParams {
            gp_params: GpValidParams {
                mean: ConstantMean::default(),
                corr,
                theta_tuning: ThetaTuning::default(),
                kpls_dim: None,
                n_start: 10,
                nugget: F::cast(1000.0) * F::epsilon(),
            },
            noise: ParamEstimation::default(),
            z: inducings,
            method: SparseMethod::default(),
            seed: None,
        })
    }

    /// Set correlation model.
    pub fn corr(mut self, corr: Corr) -> Self {
        self.0.gp_params.corr = corr;
        self
    }

    /// Set initial value for theta hyper parameter.
    ///
    /// During training process, the internal optimization is started from `theta_init`.
    pub fn theta_init(mut self, theta_init: Vec<F>) -> Self {
        self.0.gp_params.theta_tuning = ParamTuning {
            init: theta_init,
            ..(self.0.gp_params.theta_tuning().clone()).into()
        }
        .try_into()
        .unwrap();
        self
    }

    /// Set theta hyper parameter search space.
    pub fn theta_bounds(mut self, theta_bounds: Vec<(F, F)>) -> Self {
        self.0.gp_params.theta_tuning = ParamTuning {
            bounds: theta_bounds,
            ..(self.0.gp_params.theta_tuning()).clone().into()
        }
        .try_into()
        .unwrap();
        self
    }

    /// Get starting theta value for optimization
    pub fn theta_tuning(mut self, theta_tuning: ThetaTuning<F>) -> Self {
        self.0.gp_params.theta_tuning = theta_tuning;
        self
    }

    /// Set number of PLS components.
    ///
    /// Should be 0 < n < pb size (i.e. x dimension)
    pub fn kpls_dim(mut self, kpls_dim: Option<usize>) -> Self {
        self.0.gp_params.kpls_dim = kpls_dim;
        self
    }

    /// Set the number of internal hyperparameters optimization restarts
    pub fn n_start(mut self, n_start: usize) -> Self {
        self.0.gp_params.n_start = n_start;
        self
    }

    /// Set nugget value.
    ///
    /// Nugget is used to improve numerical stability
    pub fn nugget(mut self, nugget: F) -> Self {
        self.0.gp_params.nugget = nugget;
        self
    }

    /// Specify the sparse method
    pub fn sparse_method(mut self, method: SparseMethod) -> Self {
        self.0.method = method;
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
    pub fn noise_variance(mut self, config: ParamEstimation<F>) -> Self {
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
                return Err(GpError::InvalidValueError(
                    "`kpls_dim` canot be 0!".to_string(),
                ));
            }
            let theta = self.0.theta_tuning().theta0();
            if theta.len() > 1 && d > theta.len() {
                return Err(GpError::InvalidValueError(format!(
                    "Dimension reduction ({}) should be smaller than expected
                        training input size infered from given initial theta length ({})",
                    d,
                    theta.len()
                )));
            };
        }
        Ok(&self.0)
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}
