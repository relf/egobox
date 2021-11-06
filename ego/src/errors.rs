use gp::GpError;
use nlopt::FailState;
use std::error::Error;
use std::fmt::{self, Display};

pub type Result<T> = std::result::Result<T, EgoError>;

/// An error when modeling a GMM algorithm
#[derive(Debug)]
pub enum EgoError {
    /// When LikelihoodComputation computation fails
    GpError(String),
    /// When EGO fails
    EgoError(String),
    /// When PLS fails
    InvalidValue(String),
    /// When nlopt fails
    NloptFailure,
}

impl Display for EgoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::GpError(message) => {
                write!(f, "Gaussian process computation error: {}", message)
            }
            Self::EgoError(message) => write!(f, "EGO error: {}", message),
            Self::InvalidValue(message) => write!(f, "Value error: {}", message),
            Self::NloptFailure => write!(f, "NlOpt error"),
        }
    }
}

impl Error for EgoError {}

impl From<GpError> for EgoError {
    fn from(error: GpError) -> EgoError {
        EgoError::GpError(error.to_string())
    }
}

impl From<FailState> for EgoError {
    fn from(_error: FailState) -> EgoError {
        EgoError::NloptFailure
    }
}
