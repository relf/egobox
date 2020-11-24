use ndarray_linalg::error::LinalgError;
use std::error::Error;
use std::fmt::{self, Display};

pub type Result<T> = std::result::Result<T, EgoboxError>;

/// An error when modeling a GMM algorithm
#[derive(Debug)]
pub enum EgoboxError {
    /// When likelihood computation fails
    LikelihoodError(String),
    /// When linear algebra computation fails
    LinalgError(String),
    /// When clustering fails
    EmptyCluster(String),
}

impl Display for EgoboxError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::LikelihoodError(message) => {
                write!(f, "Likelihood computation error: {}", message)
            }
            Self::LinalgError(message) => write!(f, "Linear Algebra error: {}", message),
            Self::EmptyCluster(message) => write!(f, "Empty cluster: {}", message),
        }
    }
}

impl Error for EgoboxError {}

impl From<LinalgError> for EgoboxError {
    fn from(error: LinalgError) -> EgoboxError {
        EgoboxError::LinalgError(error.to_string())
    }
}
