//! This library implements Mixture of Experts method using [GP models](egor_gp).
//!
//! MoE method aims at increasing the accuracy of a function approximation by replacing
//! a single global model by a weighted sum of local gp regression models (experts).
//! It is based on a partition of the problem domain into several subdomains
//! via clustering algorithms followed by a local expert training on each subdomain.
//!
//! This library is a port of the [SMT MOE method](https://smt.readthedocs.io/en/latest/_src_docs/applications/moe.html)
//! using Egor GP models as experts.
//! It leverages on the Egor GP KPLS features to handle high dimensional problems.
//!
//! Reference: Bettebghor, Dimitri, et al. [Surrogate modeling approximation using a mixture of
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
