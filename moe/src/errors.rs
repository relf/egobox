// use gp::GpError;
// use ndarray_linalg::error::LinalgError;
use thiserror::Error;
// use std::fmt::{self, Display};

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
    GpError(#[from] gp::GpError),
    /// When best expert search fails
    #[error("Expert error: {0}")]
    ExpertError(String),
    /// When error on clustering
    #[error("Clustering error: {0}")]
    ClusteringError(String),
}
