use gp::GpError;
use std::error::Error;
use std::fmt::{self, Display};

pub type Result<T> = std::result::Result<T, EgoboxError>;

/// An error when modeling a GMM algorithm
#[derive(Debug)]
pub enum EgoboxError {
    /// When LikelihoodComputation computation fails
    GpError(String),
    /// When EGO fails
    EgoError(String),
    /// When PLS fails
    InvalidValue(String),
}

impl Display for EgoboxError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::GpError(message) => {
                write!(f, "Gaussian process computation error: {}", message)
            }
            Self::EgoError(message) => write!(f, "EGO error: {}", message),
            Self::InvalidValue(message) => write!(f, "Value error: {}", message),
        }
    }
}

impl Error for EgoboxError {}

impl From<GpError> for EgoboxError {
    fn from(error: GpError) -> EgoboxError {
        EgoboxError::GpError(error.to_string())
    }
}
