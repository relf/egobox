/*!
This library implements some Design of Experiments (DoE) methods a.k.a. sampling methods,
specially the [Latin Hypercube sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling)
method which is used by surrogate-based methods.
This library is a port of [SMT sampling methods](https://smt.readthedocs.io/en/latest/_src_docs/sampling_methods.html).

A DoE method is a way to generate a set of points (i.e. a DoE) within a design (or sample) space `xlimits`.
The design space is defined as a 2D ndarray `(nx, 2)`, specifying lower bound and upper bound
of each `nx` components of the samples `x`.

Example:
```
use egobox_doe::{FullFactorial, Lhs, LhsKind, Random, SamplingMethod};
use ndarray::{arr2};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

// Design space is defined as [5., 10.] x [0., 1.], samples are 2-dimensional.
let xlimits = arr2(&[[5., 10.], [0., 1.]]);
// We generate five samples using centered Latin Hypercube sampling.
let samples = Lhs::new(&xlimits).kind(LhsKind::Centered).sample(5);
// or else with FullFactorial sampling
let samples = FullFactorial::new(&xlimits).sample(5);
// or else randomly with random generator for reproducibility
let samples = Random::new(&xlimits).with_rng(Xoshiro256Plus::seed_from_u64(42)).sample(5);
```

This library contains three kinds of sampling methods:
* [Latin Hypercube Sampling](crate::lhs::Lhs),
* [Full Factorial Sampling](crate::full_factorial::FullFactorial),
* [Random Sampling](crate::random::Random)

*/
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
mod full_factorial;
mod lhs;
mod random;
mod traits;
mod utils;

pub use full_factorial::*;
pub use lhs::*;
pub use random::*;
pub use traits::*;
