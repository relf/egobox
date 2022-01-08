use nlopt::FailState;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, EgoError>;

/// An error when modeling a GMM algorithm
#[derive(Error, Debug)]
pub enum EgoError {
    /// When LikelihoodComputation computation fails
    #[error("GP error")]
    GpError(#[from] gp::GpError),
    /// When EGO fails
    #[error("EGO error: {0}")]
    EgoError(String),
    /// When PLS fails
    #[error("Value error: {0}")]
    InvalidValue(String),
    /// When nlopt fails
    #[error("NLOpt optimizer error")]
    NloptFailure,
    /// When Moe error occurs
    #[error("MOE error")]
    MoeError(#[from] moe::MoeError),
}

impl From<FailState> for EgoError {
    fn from(_error: FailState) -> EgoError {
        EgoError::NloptFailure
    }
}
