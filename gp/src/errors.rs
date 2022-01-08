use linfa_pls::PlsError;
use ndarray_linalg::error::LinalgError;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, GpError>;

/// An error when modeling a GMM algorithm
#[derive(Error, Debug)]
pub enum GpError {
    /// When LikelihoodComputation computation fails
    #[error("LikelihoodComputation computation error: {0}")]
    LikelihoodComputationError(String),
    /// When linear algebra computation fails
    #[error("Linear Algebra error")]
    LinalgError(#[from] LinalgError),
    /// When clustering fails
    #[error("Empty cluster: {0}")]
    EmptyCluster(String),
    /// When PLS fails
    #[error("PLS error: {0}")]
    PlsError(#[from] PlsError),
    /// When a value is invalid
    #[error("PLS error: {0}")]
    InvalidValue(String),
    /// When error during saving
    #[error("Save error: {0}")]
    SaveError(#[from] serde_json::Error),
    /// When error during loading
    #[error("Load IO error")]
    LoadIoError(#[from] std::io::Error),
    /// When error during loading
    #[error("Load error: {0}")]
    LoadError(String),
}
