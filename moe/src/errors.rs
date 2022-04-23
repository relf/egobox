// use egobox_gp::GpError;
// use ndarray_linalg::error::LinalgError;
use thiserror::Error;
// use std::fmt::{self, Display};

/// A result type for Moe algorithm
pub type Result<T> = std::result::Result<T, MoeError>;

/// An error when using MOE algorithm
#[derive(Error, Debug)]
pub enum MoeError {
    /// When linear algebra computation fails
    #[error("Linear Algebra error")]
    LinalgError(#[from] ndarray_linalg::error::LinalgError),
    /// When clustering fails
    #[error("Empty cluster: {0}")]
    EmptyCluster(String),
    /// When Gaussian Process fails
    #[error("GP error")]
    GpError(#[from] egobox_gp::GpError),
    /// When best expert search fails
    #[error("Expert error: {0}")]
    ExpertError(String),
    /// When error on clustering
    #[error("Clustering error: {0}")]
    ClusteringError(String),
    /// When error during saving
    #[cfg(feature = "persistent")]
    #[error("Save error: {0}")]
    SaveError(#[from] serde_json::Error),
    /// When error during loading
    #[error("Load IO error")]
    LoadIoError(#[from] std::io::Error),
    /// When error during loading
    #[error("Load error: {0}")]
    LoadError(String),
    /// When error during loading
    #[error("InvalidValue error: {0}")]
    InvalidValueError(String),
}
