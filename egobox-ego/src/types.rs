use linfa::Float;
use ndarray::{Array1, Array2, ArrayView2};

/// Optimization result
#[derive(Clone, Debug)]
pub struct OptimResult<F: Float> {
    pub x_opt: Array1<F>,
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
    Slsqp,
    Cobyla,
}

/// Strategy to choose several points at each iteration
/// to benefit from parallel evaluation of the objective function
#[derive(Clone, Debug, PartialEq)]
pub enum QEiStrategy {
    KrigingBeliever,
    KrigingBelieverLowerBound,
    KrigingBelieverUpperBound,
    ConstantLiarMinimum,
}

/// An interface for objective function to be optimized
/// The function is expected to return a matrix allowing nrows evaluations at once.
/// A row of the output matrix is expected to contain [objective, cstr_1, ... cstr_n] values.
pub trait GroupFunc: Send + Sync + 'static + Clone + Fn(&ArrayView2<f64>) -> Array2<f64> {}
impl<T> GroupFunc for T where T: Send + Sync + 'static + Clone + Fn(&ArrayView2<f64>) -> Array2<f64> {}