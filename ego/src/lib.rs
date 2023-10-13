//! This library implements Efficient Global Optimization method,
//! it started as a port of the [EGO algorithm](https://smt.readthedocs.io/en/stable/_src_docs/applications/ego.html)
//! implemented as an application example in [SMT](https://smt.readthedocs.io/en/stable).
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
//! ## Continuous optimization
//!
//! ```
//! use ndarray::{array, Array2, ArrayView2};
//! use egobox_ego::EgorBuilder;
//!
//! // A one-dimensional test function, x in [0., 25.] and min xsinx(x) ~ -15.1 at x ~ 18.9
//! fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
//!     (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
//! }
//!
//! // We ask for 10 evaluations of the objective function to get the result
//! let res = EgorBuilder::optimize(xsinx)
//!             .min_within(&array![[0.0, 25.0]])
//!             .n_iter(10)
//!             .run()
//!             .expect("xsinx minimized");
//! println!("Minimum found f(x) = {:?} at x = {:?}", res.x_opt, res.y_opt);
//! ```
//!
//! The implementation relies on [Mixture of Experts](egobox_moe).
//!
//!
//! ## Mixed-integer optmization
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
//! use egobox_ego::{EgorBuilder, InfillStrategy, XType};
//!
//! fn mixsinx(x: &ArrayView2<f64>) -> Array2<f64> {
//!     if (x.mapv(|v| v.round()).norm_l2() - x.norm_l2()).abs() < 1e-6 {
//!         (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
//!     } else {
//!         panic!("Error: mixsinx works only on integer, got {:?}", x)
//!     }
//! }
//!
//! let n_iter = 10;
//! let doe = array![[0.], [7.], [25.]];   // the initial doe
//!
//! // We define input as being integer
//! let xtypes = vec![XType::Int(0, 25)];
//!
//! let res = EgorBuilder::optimize(mixsinx)
//!     .random_seed(42)
//!     .min_within_mixint_space(&xtypes)  // We build a mixed-integer optimizer
//!     .doe(&doe)                         // we pass the initial doe
//!     .n_iter(n_iter)
//!     .infill_strategy(InfillStrategy::EI)
//!     .run()
//!     .expect("Egor minimization");
//! println!("min f(x)={} at x={}", res.y_opt, res.x_opt);
//! ```  
//!
//! # Usage
//!
//! The [`EgorBuilder`] class is used to build an initial optimizer setting
//! the objective function, an optional random seed (to get reproducible runs) and
//! a design space specifying the domain and dimensions of the inputs `x`.
//!  
//! The `min_within()` and `min_within_mixed_space()` methods return an [`Egor`] object, the optimizer,
//! which can be further configured.
//! The first one is used for continuous input space (eg floats only), the second one for mixed-integer input
//! space (some variables, components of `x`, may be integer, ordered or categorical).
//!
//! Some of the most useful options are:
//!
//! * Specification of the size of the initial DoE. The default is nx+1 where nx is the dimension of x.
//! If your objective function is not expensive you can take `3*nx` to help the optimizer
//! approximating your objective function.
//!
//! ```no_run
//! # use egobox_ego::{EgorBuilder};
//! # use ndarray::{array, Array2, ArrayView2};
//! # fn fobj(x: &ArrayView2<f64>) -> Array2<f64> { x.to_owned() }
//! # let egor = EgorBuilder::optimize(fobj).min_within(&array![[-1., 1.]]);
//!     egor.n_doe(100);
//! ```
//!
//! You can also provide your initial doe though the `egor.doe(your_doe)` method.
//!
//! * As the dimension increase the gaussian process surrogate building may take longer or even fail
//! in this case you can specify a PLS dimension reduction \[[Bartoli2019](#Bartoli2019)\].
//! Gaussian process will be built using the `ndim` (usually 3 or 4) main components in the PLS projected space.
//!
//! ```no_run
//! # use egobox_ego::{EgorBuilder};
//! # use ndarray::{array, Array2, ArrayView2};
//! # fn fobj(x: &ArrayView2<f64>) -> Array2<f64> { x.to_owned() }
//! # let egor = EgorBuilder::optimize(fobj).min_within(&array![[-1., 1.]]);
//!     egor.kpls_dim(3);
//! ```
//!
//! * Specifications of constraints (expected to be negative at the end of the optimization)
//! In this example below we specify that 2 constraints will be computed with the objective values meaning
//! the objective function is expected to return an array '\[nsamples, 1 obj value + 2 const values\]'.
//!
//! ```no_run
//! # use egobox_ego::{EgorBuilder};
//! # use ndarray::{array, Array2, ArrayView2};
//! # fn fobj(x: &ArrayView2<f64>) -> Array2<f64> { x.to_owned() }
//! # let egor = EgorBuilder::optimize(fobj).min_within(&array![[-1., 1.]]);
//!     egor.n_cstr(2);
//! ```
//!
//! * If the default infill strategy (WB2, Watson and Barnes 2nd criterion),
//! you can switch for either EI (Expected Improvement) or WB2S (scaled version of WB2).
//! See \[[Priem2019](#Priem2019)\]
//!
//! ```no_run
//! # use egobox_ego::{EgorBuilder, InfillStrategy};
//! # use ndarray::{array, Array2, ArrayView2};
//! # fn fobj(x: &ArrayView2<f64>) -> Array2<f64> { x.to_owned() }
//! # let egor = EgorBuilder::optimize(fobj).min_within(&array![[-1., 1.]]);
//!     egor.infill_strategy(InfillStrategy::EI);
//! ```
//!
//! * The default gaussian process surrogate is parameterized with a constant trend and a squared exponential correlation kernel, also
//! known as Kriging. The optimizer use such surrogates to approximate objective and constraint functions. The kind of surrogate
//! can be changed using `regression_spec` and `correlation_spec()` methods to specify trend and kernels tested to get the best
//! approximation (quality tested through cross validation).
//!
//! ```no_run
//! # use egobox_ego::{EgorBuilder, RegressionSpec, CorrelationSpec};
//! # use ndarray::{array, Array2, ArrayView2};
//! # fn fobj(x: &ArrayView2<f64>) -> Array2<f64> { x.to_owned() }
//! # let egor = EgorBuilder::optimize(fobj).min_within(&array![[-1., 1.]]);
//!     egor.regression_spec(RegressionSpec::CONSTANT | RegressionSpec::LINEAR)
//!         .correlation_spec(CorrelationSpec::MATERN32 | CorrelationSpec::MATERN52);
//! ```
//! In the above example all GP with combinations of regression and correlation will be tested and the best combination for
//! each modeled function will be retained. You can also simply specify `RegressionSpec::ALL` and `CorrelationSpec::ALL` to
//! test all available combinations but remember that the more you test the slower it runs.
//!  
//! # Implementation notes
//!
//! * Mixture of experts and PLS dimension reduction is explained in \[[Bartoli2019](#Bartoli2019)\]
//! * Parallel optimization is available through the selection of a qei strategy.
//! More information in \[[Ginsbourger2010](#Ginsbourger2010)\]
//! * Mixed integer approach is implemented using continuous relaxation.
//! More information in \[[Garrido2018](#Garrido2018)\]
//!
//! # References
//!
//! \[<a id="Bartoli2019">Bartoli2019</a>\]: Bartoli, Nathalie, et al. [Adaptive modeling strategy for constrained global
//! optimization with application to aerodynamic wing design](https://www.sciencedirect.com/science/article/pii/S1270963818306011)
//!  Aerospace Science and technology 90 (2019): 85-102.
//!
//! \[<a id="Priem2019">Priem2019</a>\] Priem, Rémy, Nathalie Bartoli, and Youssef Diouane.
//! On the use of upper trust bounds in constrained Bayesian optimization infill criteria.
//! AIAA aviation 2019 forum. 2019.
//!
//! \[<a id="Ginsbourger2010">Ginsbourger2010</a>\]: Ginsbourger, D., Le Riche, R., & Carraro, L. (2010).
//! Kriging is well-suited to parallelize optimization.
//!  
//! \[<a id="Garrido2018">Garrido2018</a>\]: E.C. Garrido-Merchan and D. Hernandez-Lobato. Dealing with categorical and
//! integer-valued variables in Bayesian Optimization with Gaussian processes.
//!
//!
mod criteria;
mod egor;
mod egor_solver;
mod egor_state;
mod errors;
mod mixint;
mod types;

#[cfg(not(feature = "nlopt"))]
mod optimizer;

mod lhs_optimizer;
mod sort_axis;
mod utils;

pub use crate::criteria::*;
pub use crate::egor::*;
pub use crate::egor_solver::*;
pub use crate::egor_state::*;
pub use crate::errors::*;
pub use crate::mixint::*;
pub use crate::types::*;
