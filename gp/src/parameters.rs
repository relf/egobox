use crate::correlation_models::{CorrelationModel, SquaredExponentialCorr};
use crate::errors::{GpError, Result};
use crate::mean_models::{ConstantMean, RegressionModel};
use linfa::{Float, ParamGuard};
use std::convert::TryFrom;

#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

/// A structure to represent a n-dim parameter estimation
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct ParamTuning<F: Float> {
    pub guess: Vec<F>,
    pub bounds: Vec<(F, F)>,
}

impl<F: Float> TryFrom<ParamTuning<F>> for ThetaTuning<F> {
    type Error = GpError;
    fn try_from(pt: ParamTuning<F>) -> Result<ThetaTuning<F>> {
        if pt.guess.len() != pt.bounds.len() && (pt.guess.len() != 1 || pt.bounds.len() != 1) {
            return Err(GpError::InvalidValueError(
                "Bad theta tuning specification".to_string(),
            ));
        }
        // TODO: check if guess in bounds
        Ok(ThetaTuning(pt))
    }
}

/// As structure for theta hyperparameters guess
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]

pub struct ThetaTuning<F: Float>(ParamTuning<F>);
impl<F: Float> Default for ThetaTuning<F> {
    fn default() -> ThetaTuning<F> {
        ThetaTuning(ParamTuning {
            guess: vec![F::cast(0.01)],
            bounds: vec![(F::cast(1e-6), F::cast(1e2))],
        })
    }
}

impl<F: Float> From<ThetaTuning<F>> for ParamTuning<F> {
    fn from(tt: ThetaTuning<F>) -> ParamTuning<F> {
        ParamTuning {
            guess: tt.0.guess,
            bounds: tt.0.bounds,
        }
    }
}

impl<F: Float> ThetaTuning<F> {
    pub fn theta0(&self) -> &[F] {
        &self.0.guess
    }
    pub fn bounds(&self) -> &[(F, F)] {
        &self.0.bounds
    }
}

/// A set of validated GP parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpValidParams<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> {
    /// Parameter of the autocorrelation model
    pub(crate) theta: Option<Vec<F>>,
    /// Parameter guess of the autocorrelation model
    pub(crate) theta_tuning: ThetaTuning<F>,
    /// Regression model representing the mean(x)
    pub(crate) mean: Mean,
    /// Correlation model representing the spatial correlation between errors at e(x) and e(x')
    pub(crate) corr: Corr,
    /// Optionally apply dimension reduction (KPLS) or not
    pub(crate) kpls_dim: Option<usize>,
    /// Number of optimization restart
    pub(crate) n_start: usize,
    /// Parameter to improve numerical stability
    pub(crate) nugget: F,
}

impl<F: Float> Default for GpValidParams<F, ConstantMean, SquaredExponentialCorr> {
    fn default() -> GpValidParams<F, ConstantMean, SquaredExponentialCorr> {
        GpValidParams {
            theta: None,
            theta_tuning: ThetaTuning::default(),
            mean: ConstantMean(),
            corr: SquaredExponentialCorr(),
            kpls_dim: None,
            n_start: 10,
            nugget: F::cast(100.0) * F::epsilon(),
        }
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> GpValidParams<F, Mean, Corr> {
    /// Get mean model  
    pub fn mean(&self) -> &Mean {
        &self.mean
    }

    /// Get correlation corr k(x, x')
    pub fn corr(&self) -> &Corr {
        &self.corr
    }

    /// Get starting theta value for optimization
    pub fn theta_tuning(&self) -> &ThetaTuning<F> {
        &self.theta_tuning
    }

    /// Get number of components used by PLS
    pub fn kpls_dim(&self) -> &Option<usize> {
        &self.kpls_dim
    }

    /// Get the number of internal optimization restart
    pub fn n_start(&self) -> usize {
        self.n_start
    }

    /// Get number of components used by PLS
    pub fn nugget(&self) -> F {
        self.nugget
    }
}

#[derive(Clone, Debug)]
/// The set of hyperparameters that can be specified for the execution of
/// the [GP algorithm](struct.GaussianProcess.html).
pub struct GpParams<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>>(
    GpValidParams<F, Mean, Corr>,
);

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> GpParams<F, Mean, Corr> {
    /// A constructor for GP parameters given mean and correlation models
    pub fn new(mean: Mean, corr: Corr) -> GpParams<F, Mean, Corr> {
        Self(GpValidParams {
            theta: None,
            theta_tuning: ThetaTuning::default(),
            mean,
            corr,
            kpls_dim: None,
            n_start: 10,
            nugget: F::cast(100.0) * F::epsilon(),
        })
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

    /// Set the number of PLS components.
    /// Should be 0 < n < pb size (i.e. x dimension)
    pub fn kpls_dim(mut self, kpls_dim: Option<usize>) -> Self {
        self.0.kpls_dim = kpls_dim;
        self
    }

    /// Set initial value for theta hyper parameter.
    ///
    /// During training process, the internal optimization is started from `theta_guess`.
    pub fn theta_guess(mut self, theta_guess: Vec<F>) -> Self {
        self.0.theta_tuning = ParamTuning {
            guess: theta_guess,
            ..ThetaTuning::default().into()
        }
        .try_into()
        .unwrap();
        self
    }

    /// Set theta hyper parameter search space.
    pub fn theta_bounds(mut self, theta_bounds: Vec<(F, F)>) -> Self {
        self.0.theta_tuning = ParamTuning {
            bounds: theta_bounds,
            ..ThetaTuning::default().into()
        }
        .try_into()
        .unwrap();
        self
    }

    /// Set theta hyper parameter tuning
    pub fn theta_tuning(mut self, theta_tuning: ThetaTuning<F>) -> Self {
        self.0.theta_tuning = theta_tuning;
        self
    }

    /// Set the number of internal GP hyperparameter theta optimization restarts
    pub fn n_start(mut self, n_start: usize) -> Self {
        self.0.n_start = n_start;
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
