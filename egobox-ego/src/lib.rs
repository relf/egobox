//! This library implements Efficient Global Optimization method,
//! it is a port of [SMT EGO algorithm](https://smt.readthedocs.io/en/latest/_src_docs/applications/ego.html)
//!
//! The optimizer is able to handle inequality constraints. Objective and contraints
//! are expected to computed grouped at the same time hence the given function
//! should return a vector where the first component is the objective value and
//! the remaining ones constraints values intended to be negative in the end.   
//! The optimizer comes with a set of options to:
//! * specify the initial doe,
//! * parameterize internal optimization,
//! * parameterize mixture of experts,
//! * save intermediate results and allow hot restart,
//!
//! Examples:
//!
//! ```no_run
//! // Here is our unconstrained function to minimize
//! # use ndarray::{array, Array2, ArrayView2};
//! # use egobox_ego::Egor;
//!
//! fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
//!     (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
//! }
//!
//! // We specify an initial x doe
//! let initial_doe = array![[0.], [7.], [25.]];
//! // We ask for 10 evaluation of the objective function to get the result
//! let res = Egor::new(xsinx, &array![[0.0, 25.0]])
//!             .n_eval(10)
//!             .minimize()
//!             .expect("xsinx minimized");
//! println!("Minimum found f(x) = {:?} at x = {:?}", res.x_opt, res.y_opt);
//! ```
//!
//! // Here a function with a constraint
//!
//! ```no_run
//! # use ndarray::{array, Array2, ArrayView1, ArrayView2, Zip};
//! # use egobox_doe::{Lhs, SamplingMethod};
//! # use egobox_ego::{ApproxValue, Egor, InfillStrategy, InfillOptimizer};
//! # use rand_isaac::Isaac64Rng;
//! # use ndarray_rand::rand::SeedableRng;
//!
//! // Function G24: 1 global optimum y_opt = -5.5080 at x_opt =(2.3295, 3.1785)
//! fn g24(x: &ArrayView1<f64>) -> f64 {
//!    -x[0] - x[1]
//! }
//!
//! // Constraints < 0
//! fn g24_c1(x: &ArrayView1<f64>) -> f64 {
//!     -2.0 * x[0].powf(4.0) + 8.0 * x[0].powf(3.0) - 8.0 * x[0].powf(2.0) + x[1] - 2.0
//! }
//!
//! fn g24_c2(x: &ArrayView1<f64>) -> f64 {
//!     -4.0 * x[0].powf(4.0) + 32.0 * x[0].powf(3.0)
//!     - 88.0 * x[0].powf(2.0) + 96.0 * x[0] + x[1]
//!     - 36.0
//! }
//!
//! // Gouped function : objective + constraints
//! fn f_g24(x: &ArrayView2<f64>) -> Array2<f64> {
//!     let mut y = Array2::zeros((x.nrows(), 3));
//!     Zip::from(y.rows_mut())
//!         .and(x.rows())
//!         .for_each(|mut yi, xi| {
//!             yi.assign(&array![g24(&xi), g24_c1(&xi), g24_c2(&xi)]);
//!         });
//!     y
//! }
//!
//! let xlimits = array![[0., 3.], [0., 4.]];
//! let doe = Lhs::new(&xlimits)
//!            .with_rng(Isaac64Rng::seed_from_u64(42))
//!            .sample(10);
//! let res = Egor::new(f_g24, &xlimits)
//!            .with_rng(Isaac64Rng::seed_from_u64(42))
//!            .n_cstr(2)
//!            .infill_strategy(InfillStrategy::EI)
//!            .infill_optimizer(InfillOptimizer::Cobyla)
//!            .doe(Some(doe))
//!            .n_eval(40)
//!            .expect(Some(ApproxValue {  // Known solution, the algo exits if reached
//!               value: -5.5080,
//!               tolerance: 1e-3,
//!            }))
//!            .minimize()
//!            .expect("g24 minimized");
//! println!("G24 optim result = {:?}", res);
//! ```
//!
//! The implementation relies on [Mixture of Experts](egobox_moe).
//! While [crate::Egor] optimizer works with continuous data (i.e floats), the class [crate::MixintEgor]
//! allows to make mixed-integer optimization by decorating `Egor` class.    
//!
//! References:
//!
//! * Bartoli, Nathalie, et al.[Improvement of efficient global optimization with application to
//! aircraft wing design](https://www.researchgate.net/publication/303902935_Improvement_of_efficient_global_optimization_with_application_to_aircraft_wing_design)
//! 7th AIAA/ISSMO Multidisciplinary analysis and optimization conference. 2016.
//!
//!
mod egor;
mod errors;
mod mixint;
mod mixintegor;
mod sort_axis;
mod types;
mod utils;

pub use crate::egor::*;
pub use crate::errors::*;
pub use crate::mixint::*;
pub use crate::mixintegor::*;
pub use crate::types::*;
