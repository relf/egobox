//! This library implements Efficient Global Optimization method,
//! it is a port of [SMT EGO algorithm](https://smt.readthedocs.io/en/stable/_src_docs/applications/ego.html)
//!
//! The optimizer is able to deal with inequality constraints.
//! Objective and contraints are expected to computed grouped at the same time
//! hence the given function should return a vector where the first component
//! is the objective value and the remaining ones constraints values intended
//! to be negative in the end.   
//! The optimizer comes with a set of options to:
//! * specify the initial doe,
//! * parameterize internal optimization,
//! * parameterize mixture of experts,
//! * save intermediate results and allow hot restart,
//!
//! # Examples
//!
//! ```
//! use ndarray::{array, Array2, ArrayView2};
//! use egobox_ego::EgorBuilder2;
//!
//! // A one-dimensional test function, x in [0., 25.] and min xsinx(x) ~ -15.1 at x ~ 18.9
//! fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
//!     (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
//! }
//!
//! // We ask for 10 evaluations of the objective function to get the result
//! let res = EgorBuilder2::optimize(xsinx)
//!             .min_within(&array![[0.0, 25.0]])
//!             .n_eval(10)
//!             .run()
//!             .expect("xsinx minimized");
//! println!("Minimum found f(x) = {:?} at x = {:?}", res.x_opt, res.y_opt);
//! ```
//!
//! The implementation relies on [Mixture of Experts](egobox_moe).
//!
//! While [Egor] optimizer works with continuous data (i.e floats), the optimizer
//! allows to make basic mixed-integer optimization. The configuration of the Optimizer
//! as a mixed_integer optimizer is done though the `EgorBuilder`  
//!
//! As a second example, we define an objective function `mixsinx` taking integer
//! input values from the previous function `xsinx` defined above.
//!  
//! ```
//! use ndarray::{array, Array2, ArrayView2};
//! use linfa::ParamGuard;
//! #[cfg(feature = "blas")]
//! use ndarray_linalg::Norm;
//! #[cfg(not(feature = "blas"))]
//! use linfa_linalg::norm::*;
//! use egobox_ego::{EgorBuilder2, InfillStrategy, Xtype};
//!
//! fn mixsinx(x: &ArrayView2<f64>) -> Array2<f64> {
//!     if (x.mapv(|v| v.round()).norm_l2() - x.norm_l2()).abs() < 1e-6 {
//!         (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
//!     } else {
//!         panic!("Error: mixsinx works only on integer, got {:?}", x)
//!     }
//! }
//!
//! let n_eval = 10;
//! let doe = array![[0.], [7.], [25.]];
//!
//! // We define input as being integer
//! let xtypes = vec![Xtype::Int(0, 25)];
//!
//! let res = EgorBuilder2::optimize(mixsinx)
//!     .random_seed(42)
//!     .min_within_mixed_space(&xtypes)   // We build mixed-integer optimizer
//!     .doe(Some(doe))          // we pass an initial doe
//!     .n_eval(n_eval)
//!     .infill_strategy(InfillStrategy::EI)
//!     .run()
//!     .expect("Egor minimization");
//! println!("min f(x)={} at x={}", res.y_opt, res.x_opt);
//! ```  
//!
//! # Implementation notes
//!
//! * Mixture of experts and PLS dimension reduction is explained in \[[Bartoli2019](#Bartoli2019)\]
//! * Parallel optimization is available through the selection of a qei strategy.
//! More information in \[[Ginsbourger2010](#Ginsbourger2010)\]
//! * Mixed integer approach is imlemented in [MixintEgor].
//! More information in \[[Garrido2018](#Garrido2018)\]
//!
//! # References
//!
//! \[<a id="Bartoli2019">Bartoli2019</a>\]: Bartoli, Nathalie, et al. [Adaptive modeling strategy for constrained global
//! optimization with application to aerodynamic wing design](https://www.sciencedirect.com/science/article/pii/S1270963818306011)
//!  Aerospace Science and technology 90 (2019): 85-102.
//!
//! \[<a id="Ginsbourger2010">Ginsbourger2010</a>\]: Ginsbourger, D., Le Riche, R., & Carraro, L. (2010).
//! Kriging is well-suited to parallelize optimization.
//!  
//! \[<a id="Garrido2018">Garrido2018</a>\]: E.C. Garrido-Merchan and D. Hernandez-Lobato. Dealing with categorical and
//! integer-valued variables in Bayesian Optimization with Gaussian processes.
//!
//!
mod egor_solver;
mod egor_state;
mod errors;
mod lhs_optimizer;
mod mixint;
mod sort_axis;
mod types;
mod utils;

pub use crate::egor_solver::*;
pub use crate::errors::*;
pub use crate::mixint::*;
pub use crate::types::*;
