//! This library implements Mixture of Experts method using [GP models](egobox_gp).
//!
//! MoE method aims at increasing the accuracy of a function approximation by replacing
//! a single global model by a weighted sum of local gp regression models (experts).
//! It is based on a partition of the problem domain into several subdomains
//! via clustering algorithms followed by a local expert training on each subdomain.
//!
//! The recombination between the GP models can be either:
//! * `hard`: one GP model is being responsible to provide the predicted value
//! at the given point. GP selection is done by taking the largest probability of the
//! given point being part of the cluster corresponding to the expert GP.
//! In hard mode, transition between models leads to discontinuity.
//! * `smooth`: all GPs models are taken and their predicted values at a given point are
//! weighted regarding their responsability (probability of the given point being part
//! of the cluster corresponding to the expert GP). In this case the MoE model is continuous.
//! The smoothness is automatically adjusted using a factor, the heaviside factor,
//! which can also be set manually.
//!
//! # Implementation
//!
//! * Clusters are defined by clustering the training data with
//! [linfa-clustering](https://docs.rs/linfa-clustering/latest/linfa_clustering/)
//! gaussian mixture model.
//! * This library is a port of the
//! [SMT MoE method](https://smt.readthedocs.io/en/latest/_src_docs/applications/moe.html)
//! using egobox GP models as experts.
//! * It leverages on the egobox GP PLS reduction feature to handle high dimensional problems.
//! * MoE trained model can be save to disk and reloaded. See
//!  
//! # Features
//!
//! ## linfa features
//!
//! End-user project should select a BLAS/Lapack backend depending its environment;
//! it can be either:
//!
//! * Openblas: `linfa/openblas-system` or `linfa/openblas-static`
//! * Netlib: `linfa/netlib-system` or `linfa/netlib-static`
//! * Intel MKL: `linfa/intel-mkl-system` or `linfa/intel-mkl-static`
//!
//! where
//!
//! * `*-system` features: try to find the corresponding backend in your installation.
//! * `*-static` features: try to download and compile the corresponing backend.
//!
//! More information in [linfa features](https://github.com/rust-ml/linfa#blaslapack-backend).
//!
//! ## persistent
//!
//! The `persistent` feature enables `save()`/`load()` methods for a MoE model
//! to/from a json file using the [serde crate](https://serde.rs/).
//!
//! # Example
//!
//! ```no_run
//! use ndarray::{Array2, Array1, Zip, Axis};
//! use egobox_moe::{Moe, Recombination};
//! use ndarray_rand::{RandomExt, rand::SeedableRng, rand_distr::Uniform};
//! use rand_isaac::Isaac64Rng;
//! use linfa::{traits::Fit, ParamGuard, Dataset};
//!
//! // One-dimensional test function with 3 modes
//! fn f3modes(x: &Array2<f64>) -> Array2<f64> {
//!     let mut y = Array2::zeros(x.dim());
//!     Zip::from(&mut y).and(x).for_each(|yi, &xi| {
//!         if xi < 0.4 {
//!             *yi = xi * xi;
//!         } else if (0.4..0.8).contains(&xi) {
//!             *yi = 3. * xi + 1.;
//!         } else {
//!             *yi = f64::sin(10. * xi);
//!         }
//!     });
//!     y
//! }
//!
//! // Training data
//! let mut rng = Isaac64Rng::from_entropy();
//! let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
//! let yt = f3modes(&xt);
//! let ds = Dataset::new(xt, yt);
//!
//! // Predictions
//! let observations = Array1::linspace(0., 1., 100).insert_axis(Axis(1));
//! let predictions = Moe::params(3)
//!                     .recombination(Recombination::Hard)
//!                     .fit(&ds)
//!                     .expect("MoE model training")
//!                     .predict_values(&observations)
//!                     .expect("MoE predictions");
//! ```
//!
//! # Reference
//!
//! Bettebghor, Dimitri, et al. [Surrogate modeling approximation using a mixture of
//! experts based on EM joint estimation](https://hal.archives-ouvertes.fr/hal-01852300/document)
//! Structural and multidisciplinary optimization 43.2 (2011): 243-259.
//!
mod algorithm;
mod clustering;
mod errors;
mod expertise_macros;
mod gaussian_mixture;
mod parameters;
mod surrogates;

pub use algorithm::*;
pub use errors::*;
pub use parameters::*;
pub use surrogates::*;
