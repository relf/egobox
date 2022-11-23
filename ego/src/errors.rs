use nlopt::FailState;
use thiserror::Error;

/// A result type for EGO errors
pub type Result<T> = std::result::Result<T, EgoError>;

/// An error when modeling a GMM algorithm
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
    /// When nlopt fails
    #[error("NLOpt optimizer error")]
    NloptFailure,
    /// When Moe error occurs
    #[error("MOE error")]
    MoeError(#[from] egobox_moe::MoeError),
    /// When IO fails
    #[error("IO error")]
    IoError(#[from] std::io::Error),
    /// When IO fails
    #[error("IO error")]
    ReadNpyError(#[from] ndarray_npy::ReadNpyError),
    /// When IO fails
    #[error("IO error")]
    WriteNpyError(#[from] ndarray_npy::WriteNpyError),
    /// When a linfa error occurs
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
    /// When trying to use derivatives with integer variables
    #[error("Can not use derivatives with integer variables")]
    ForbiddenDerivativesError,
}

impl From<FailState> for EgoError {
    fn from(_error: FailState) -> EgoError {
        EgoError::NloptFailure
    }
}
