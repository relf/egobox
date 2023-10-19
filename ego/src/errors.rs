use thiserror::Error;

/// A result type for EGO errors
pub type Result<T> = std::result::Result<T, EgoError>;

/// An error for efficient global optimization algorithm
#[derive(Error, Debug)]
pub enum EgoError {
    /// When LikelihoodComputation computation fails
    #[error("GP error")]
    GpError(#[from] egobox_gp::GpError),
    /// When EGO fails
    #[error("EGO error: {0}")]
    EgoError(String),
    /// When an invalid value is encountered
    #[error("Value error: {0}")]
    InvalidValue(String),
    /// When Moe error occurs
    #[error("MOE error")]
    MoeError(#[from] egobox_moe::MoeError),
    /// When IO fails
    #[error("IO error")]
    IoError(#[from] std::io::Error),
    /// When numpy array read fails
    #[error("IO error")]
    ReadNpyError(#[from] ndarray_npy::ReadNpyError),
    /// When numpy array write fails
    #[error("IO error")]
    WriteNpyError(#[from] ndarray_npy::WriteNpyError),
    /// When a `linfa` error occurs
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
    /// When an Argmin framework is raised
    #[error(transparent)]
    ArgminError(#[from] argmin::core::Error),
}
