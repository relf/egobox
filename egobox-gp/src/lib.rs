//! This library implements [Gaussian Process](https://en.wikipedia.org/wiki/Gaussian_process) regression
//! also known as [Kriging](https://en.wikipedia.org/wiki/Kriging) models,
//! it is a port of [SMT Kriging and KPLS surrogate models](https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models.html).
//!
//! A GP regression is an interpolation method where the
//! interpolated values are modeled by a Gaussian process with a mean  
//! governed by a prior covariance kernel, which depends on some
//! parameters to be determined.
//!
//! The interpolated output is modeled as stochastic process as follows:
//!
//! `Y(x) = mu(x) + Z(x)`
//!
//! where:
//! * `mu(x)` is the trend acting as the mean of the process
//! * `Z(x)` the realization of stochastic Gaussian process ~ `Normal(0, sigma^2)`
//!
//! which in turn is written as:
//!
//! `Y(x) = betas.regr(x) + sigma^2*corr(x, x')`
//!
//! where:
//! * `betas` is a vector of linear regression parameters to be determined
//! * `regr(x)` a vector of polynomial basis functions
//! * `sigma^2` is the process variance
//! * `corr(x, x')` is a correlation function which depends on `distance(x, x')`
//! and a set of unknown parameters `thetas` to be determined.
//!
//! Implementation highlights:
//! * This library is based on [ndarray](https://github.com/rust-ndarray/ndarray)
//! and [linfa](https://github.com/rust-ml/linfa) and strive to follow [linfa guidelines](https://github.com/rust-ml/linfa/blob/master/CONTRIBUTE.md)
//! * GP mean model can be constant, linear or quadratic
//! * GP correlation model can be build the following kernels: squared exponential, absolute exponential, matern 3/2, matern 5/2    
//! cf. [SMT Kriging](https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/krg.html)
//! * For high dimensional problems, the classic GP algorithm does not perform well as
//! it depends on the inversion of a correlation (n, n) matrix which is an O(n3) operation.
//! To work around this problem the library implements dimension reduction using
//! Partial Least Squares method upon Kriging method also known as KPLS algorithm
//! * GP models can be saved and loaded using [serde](https://serde.rs/).
//!
//! Reference:
//!
//! * Bouhlel, Mohamed Amine, et al. [Improving kriging surrogates of high-dimensional design
//! models by Partial Least Squares dimension reduction](https://hal.archives-ouvertes.fr/hal-01232938/document)
//! Structural and Multidisciplinary Optimization 53.5 (2016): 935-952.
//!
#![warn(missing_docs)]
mod algorithm;
pub mod correlation_models;
mod errors;
pub mod mean_models;
mod parameters;
mod utils;

pub use algorithm::*;
pub use errors::*;
pub use parameters::*;
pub use utils::NormalizedMatrix;
