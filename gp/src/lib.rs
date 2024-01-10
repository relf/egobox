//! This library implements [Gaussian Process](https://en.wikipedia.org/wiki/Gaussian_process) regression
//! also known as [Kriging](https://en.wikipedia.org/wiki/Kriging) models,
//! it is a port of [SMT Kriging and KPLS surrogate models](https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models.html).
//!
//! It also implements Sparse Gaussian Processes methods (SGPs) which address limitations of Gaussian Processes (GPs)
//! when the number of training points is large. Indeed the complexity of GPs algorithm is in O(N^3) in processing
//! time and O(N^2) in memory where N is the number of training points. The complexity is then respectively reduced
//! to O(N.M^2) and O(NM) where M is the number of so-called inducing points with M < N.
//!   
//! GP methods are implemented by [GaussianProcess] parameterized by [GpParams].
//! SGP methods are implemented by [SparseGaussianProcess] parameterized by [SgpParams].
mod algorithm;
pub mod correlation_models;
mod errors;
pub mod mean_models;
mod sgp_algorithm;

mod parameters;
mod sgp_parameters;
mod utils;

pub use algorithm::*;
pub use errors::*;
pub use parameters::*;
pub use sgp_algorithm::*;
pub use sgp_parameters::*;
