use crate::gpmix::spec::*;
use crate::{errors::Result, EgorState};
use argmin::core::CostFunction;
use egobox_moe::{Clustering, MixtureGpSurrogate, ThetaTuning};
use linfa::Float;
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// Enums for regression and correlation selection
//pub use egobox_moe::{CorrelationSpec, RegressionSpec};

/// Optimization result
#[derive(Clone, Debug)]
pub struct OptimResult<F: Float> {
    /// Optimum x value
    pub x_opt: Array1<F>,
    /// Optimum y value (e.g. f(x_opt))
    pub y_opt: Array1<F>,
    /// History of successive x values
    pub x_doe: Array2<F>,
    /// History of successive y values (e.g f(x_doe))
    pub y_doe: Array2<F>,
    /// EgorSolver final state
    pub state: EgorState<F>,
}

/// Infill criterion used to select next promising point
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InfillStrategy {
    /// Expected Improvement
    EI,
    /// Locating the regional extreme
    WB2,
    /// Scaled WB2
    WB2S,
}

/// Optimizer used to optimize the infill criteria
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InfillOptimizer {
    /// SLSQP optimizer (gradient from finite differences)
    Slsqp,
    /// Cobyla optimizer (gradient free)
    Cobyla,
}

/// Strategy to choose several points at each iteration
/// to benefit from parallel evaluation of the objective function
/// (The Multi-points Expected Improvement (q-EI) Criterion)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

/// An interface for objective function to be optimized
///
/// The function is expected to return a matrix allowing nrows evaluations at once.
/// A row of the output matrix is expected to contain [objective, cstr_1, ... cstr_n] values.
pub trait GroupFunc: Send + Sync + 'static + Clone + Fn(&ArrayView2<f64>) -> Array2<f64> {}
impl<T> GroupFunc for T where T: Send + Sync + 'static + Clone + Fn(&ArrayView2<f64>) -> Array2<f64> {}

/// As structure to handle the objective and constraints functions for implementing
/// `argmin::CostFunction` to be used with argmin framework.
#[derive(Clone)]
pub struct ObjFunc<O: GroupFunc> {
    fobj: O,
}

impl<O: GroupFunc> ObjFunc<O> {
    pub fn new(fobj: O) -> Self {
        ObjFunc { fobj }
    }
}

impl<O: GroupFunc> CostFunction for ObjFunc<O> {
    /// Type of the parameter vector
    type Param = Array2<f64>;
    /// Type of the return value computed by the cost function
    type Output = Array2<f64>;

    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        // Evaluate 2D Rosenbrock function
        Ok((self.fobj)(&p.view()))
    }
}

/// An enumeration to define the type of an input variable component
/// with its domain definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum XType {
    /// Continuous variable in [lower bound, upper bound]
    Cont(f64, f64),
    /// Integer variable in lower bound .. upper bound
    Int(i32, i32),
    /// An Ordered variable in { float_1, float_2, ..., float_n }
    Ord(Vec<f64>),
    /// An Enum variable in { 1, 2, ..., int_n }
    Enum(usize),
}

/// A trait for surrogate training
///
/// The output surrogate used by [crate::Egor] is expected to model either
/// objective function or constraint functions
pub trait SurrogateBuilder: Clone + Serialize + Sync {
    /// Constructor from domain space specified with types.
    fn new_with_xtypes(xtypes: &[XType]) -> Self;

    /// Sets the allowed regression models used in gaussian processes.
    fn set_regression_spec(&mut self, regression_spec: RegressionSpec);

    /// Sets the allowed correlation models used in gaussian processes.
    fn set_correlation_spec(&mut self, correlation_spec: CorrelationSpec);

    /// Sets the number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
    fn set_kpls_dim(&mut self, kpls_dim: Option<usize>);

    /// Sets the number of clusters used by the mixture of surrogate experts.
    fn set_n_clusters(&mut self, n_clusters: usize);

    /// Sets the hyperparameters tuning strategy
    fn set_theta_tunings(&mut self, theta_tunings: &[ThetaTuning<f64>]);

    /// Train the surrogate with given training dataset (x, y)
    fn train(
        &self,
        xt: &ArrayView2<f64>,
        yt: &ArrayView2<f64>,
    ) -> Result<Box<dyn MixtureGpSurrogate>>;

    /// Train the surrogate with given training dataset (x, y) and given clustering
    fn train_on_clusters(
        &self,
        xt: &ArrayView2<f64>,
        yt: &ArrayView2<f64>,
        clustering: &Clustering,
    ) -> Result<Box<dyn MixtureGpSurrogate>>;
}

pub trait ObjFn<U>: Fn(&[f64], Option<&mut [f64]>, &mut U) -> f64 {}
impl<T, U> ObjFn<U> for T where T: Fn(&[f64], Option<&mut [f64]>, &mut U) -> f64 {}

/// Data used by internal infill criteria optimization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InfillObjData<F: Float> {
    pub fmin: F,
    pub scale_infill_obj: F,
    pub scale_cstr: Option<Array1<F>>,
    pub scale_wb2: F,
}

impl<F: Float> Default for InfillObjData<F> {
    fn default() -> Self {
        Self {
            fmin: F::infinity(),
            scale_infill_obj: F::one(),
            scale_cstr: None,
            scale_wb2: F::one(),
        }
    }
}
