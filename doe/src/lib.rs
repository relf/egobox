//! Design of Experiments (DoE) methods a.k.a. sampling methods
//!
//! A DoE method is a way to generate a set of points (i.e. a DoE) in the design space.
//!
//!
mod full_factorial;
mod lhs;
mod random;
mod traits;
mod utils;

pub use full_factorial::*;
pub use lhs::*;
pub use random::*;
pub use traits::*;
