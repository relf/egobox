use thiserror::Error;

/// A result type for GP regression algorithm
pub type Result<T> = std::result::Result<T, GpError>;

/// An error when using [`GaussianProcess`](crate::GaussianProcess) or a [`SparseGaussianProcess`](crate::SparseGaussianProcess) algorithm
#[derive(Error, Debug)]
pub enum GpError {
    /// When LikelihoodComputation computation fails
    #[error("LikelihoodComputation computation error: {0}")]
    LikelihoodComputationError(String),
    /// When linear algebra computation fails
    #[cfg(feature = "blas")]
    #[error("Linalg BLAS error: {0}")]
    LinalgBlasError(#[from] ndarray_linalg::error::LinalgError),
    #[error(transparent)]
    LinalgError(#[from] linfa_linalg::LinalgError),
    /// When clustering fails
    #[error("Empty cluster: {0}")]
    EmptyCluster(String),
    /// When PLS fails
    #[error("PLS error: {0}")]
    PlsError(#[from] linfa_pls::PlsError),
    /// When a value is invalid
    #[error("PLS error: {0}")]
    InvalidValue(String),
    /// When a linfa error occurs
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
}
