use linfa_pls::PlsError;
use ndarray_linalg::error::LinalgError;
use std::error::Error;
use std::fmt::{self, Display};

pub type Result<T> = std::result::Result<T, GpError>;

/// An error when modeling a GMM algorithm
#[derive(Debug)]
pub enum GpError {
    /// When LikelihoodComputation computation fails
    LikelihoodComputationError(String),
    /// When linear algebra computation fails
    LinalgError(String),
    /// When clustering fails
    EmptyCluster(String),
    /// When PLS fails
    PlsError(String),
    /// When a value is invalid
    InvalidValue(String),
}

impl Display for GpError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::LikelihoodComputationError(message) => {
                write!(f, "LikelihoodComputation computation error: {}", message)
            }
            Self::LinalgError(message) => write!(f, "Linear Algebra error: {}", message),
            Self::EmptyCluster(message) => write!(f, "Empty cluster: {}", message),
            // Self::EgoError(message) => write!(f, "EGO error: {}", message),
            Self::PlsError(message) => write!(f, "PLS error: {}", message),
            Self::InvalidValue(message) => write!(f, "Value error: {}", message),
        }
    }
}

impl Error for GpError {}

impl From<LinalgError> for GpError {
    fn from(error: LinalgError) -> GpError {
        GpError::LinalgError(error.to_string())
    }
}

impl From<PlsError> for GpError {
    fn from(error: PlsError) -> GpError {
        GpError::PlsError(error.to_string())
    }
}
