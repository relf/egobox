use ndarray_linalg::error::LinalgError;
use std::error::Error;
use std::fmt::{self, Display};

pub type Result<T> = std::result::Result<T, EgoboxError>;

/// An error when modeling a GMM algorithm
#[derive(Debug)]
pub enum EgoboxError {
    /// When LikelihoodComputation computation fails
    LikelihoodComputationError(String),
    /// When linear algebra computation fails
    LinalgError(String),
    /// When clustering fails
    EmptyCluster(String),
    /// When EGO fails
    EgoError(String),
    /// When PLS fails
    PlsError(String),
}

impl Display for EgoboxError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::LikelihoodComputationError(message) => {
                write!(f, "LikelihoodComputation computation error: {}", message)
            }
            Self::LinalgError(message) => write!(f, "Linear Algebra error: {}", message),
            Self::EmptyCluster(message) => write!(f, "Empty cluster: {}", message),
            Self::EgoError(message) => write!(f, "EGO error: {}", message),
            Self::PlsError(message) => write!(f, "PLS error: {}", message),
        }
    }
}

impl Error for EgoboxError {}

impl From<LinalgError> for EgoboxError {
    fn from(error: LinalgError) -> EgoboxError {
        EgoboxError::LinalgError(error.to_string())
    }
}
