use ndarray_linalg::error::LinalgError;
use std::error::Error;
use std::fmt::{self, Display};

pub type Result<T> = std::result::Result<T, GpError>;

/// An error when modeling a GMM algorithm
#[derive(Debug)]
pub enum GpError {
    /// When likelihood computation fails
    LikelihoodError(String),
    /// When linear algebra computation fails
    LinalgError(String),
}

impl Display for GpError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::LikelihoodError(message) => {
                write!(f, "Likelihood computation error: {}", message)
            }
            Self::LinalgError(message) => write!(f, "Lilnear Algebra error: {}", message),
        }
    }
}

impl Error for GpError {}

impl From<LinalgError> for GpError {
    fn from(error: LinalgError) -> GpError {
        GpError::LinalgError(error.to_string())
    }
}
