use crate::errors::Result;
use egobox_moe::ClusteredSurrogate;
use linfa::Float;
use ndarray::{Array1, Array2, ArrayView2};

/// Optimization result
#[derive(Clone, Debug)]
pub struct OptimResult<F: Float> {
    /// Optimum x value
    pub x_opt: Array1<F>,
    /// Optimum y value (e.g. f(x))
    pub y_opt: Array1<F>,
}

/// Infill criterion used to select next promising point
#[derive(Clone, Debug, PartialEq)]
pub enum InfillStrategy {
    /// Expected Improvement
    EI,
    /// Locating the regional extreme
    WB2,
    /// Scaled WB2
    WB2S,
}

/// Optimizer used to optimize the infill criteria
#[derive(Clone, Debug, PartialEq)]
pub enum InfillOptimizer {
    /// SLSQP optimizer (gradient from finite differences)
    Slsqp,
    /// Cobyla optimizer (gradient free)
    Cobyla,
}

/// Strategy to choose several points at each iteration
/// to benefit from parallel evaluation of the objective function
/// (The Multi-points Expected Improvement (q-EI) Criterion)
#[derive(Clone, Debug, PartialEq)]
pub enum QEiStrategy {
    /// Take the mean of the kriging predictor for q points
    KrigingBeliever,
    /// Take the minimum of kriging predictor for q points
    KrigingBelieverLowerBound,
    /// Take the maximum kriging value for q points
    KrigingBelieverUpperBound,
    /// Take the current minimum of the function found so far
    ConstantLiarMinimum,
}

/// A structure to specify an approximative value
#[derive(Clone, Copy, Debug)]
pub struct ApproxValue {
    /// Nominal value
    pub value: f64,
    /// Allowed tolerance for approximation such that (y - value) < tolerance
    pub tolerance: f64,
}

/// An interface for objective function to be optimized
///
/// The function is expected to return a matrix allowing nrows evaluations at once.
/// A row of the output matrix is expected to contain [objective, cstr_1, ... cstr_n] values.
pub trait GroupFunc: Send + Sync + 'static + Clone + Fn(&ArrayView2<f64>) -> Array2<f64> {}
impl<T> GroupFunc for T where T: Send + Sync + 'static + Clone + Fn(&ArrayView2<f64>) -> Array2<f64> {}

/// A trait for surrogate training
///
/// The output surrogate used by [crate::Egor] is expected to model either
/// objective function or constraint functions
pub trait SurrogateBuilder {
    /// Train the surrogate with given training dataset (x, y)
    fn train(&self, xt: &Array2<f64>, yt: &Array2<f64>) -> Result<Box<dyn ClusteredSurrogate>>;
}

/// An interface for preprocessing continuous input init_values
///
/// Special use for [crate::MixintEgor] optimiseur where preprocessing consists in
/// casting continuous values to discrete ones. See [crate::MixintPreProcessor]
pub trait PreProcessor {
    /// Execute the pre processing on given `x` values
    fn run(&self, x: &Array2<f64>) -> Array2<f64>;
}

/// Data used by internal infill criteria to be optimized using NlOpt
#[derive(Clone)]
pub(crate) struct ObjData<F> {
    pub scale_obj: F,
    pub scale_cstr: Array1<F>,
    pub scale_wb2: F,
}
