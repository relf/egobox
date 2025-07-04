#![allow(deprecated)]
use crate::gpmix::spec::*;
use crate::{errors::Result, EgorState};
use argmin::core::CostFunction;
use egobox_moe::{Clustering, MixtureGpSurrogate, NbClusters, Recombination, ThetaTuning};
use linfa::Float;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

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
    /// Log of Expected Improvement
    LogEI,
    /// Locating the regional extreme
    WB2,
    /// Scaled WB2
    WB2S,
}

/// Constraint criterion used to select next promising point
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintStrategy {
    /// Use the mean value
    MeanConstraint,
    /// Use the upper bound (ie mean + 3*sigma)
    UpperTrustBound,
}

/// Optimizer used to optimize the infill criteria
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InfillOptimizer {
    /// SLSQP optimizer (gradient based)
    Slsqp,
    /// Cobyla optimizer (gradient free)
    Cobyla,
    /// GBNM optimizer (gradient free but do not handle constraints)
    /// Use with constrained infill criterion in presence of constraints
    Gbnm,
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
pub trait GroupFunc: Clone + Fn(&ArrayView2<f64>) -> Array2<f64> {}
impl<T> GroupFunc for T where T: Clone + Fn(&ArrayView2<f64>) -> Array2<f64> {}

/// A trait to retrieve functions constraints specifying
/// the domain of the input variables.
pub trait DomainConstraints<C: CstrFn> {
    /// Returns the list of constraints functions
    fn fn_constraints(&self) -> &[impl CstrFn];
}

/// As structure to handle the objective and constraints functions for implementing
/// `argmin::CostFunction` to be used with argmin framework.
#[derive(Clone)]
pub struct ObjFunc<O: GroupFunc, C: CstrFn> {
    fobj: O,
    fcstrs: Vec<C>,
}

impl<O: GroupFunc, C: CstrFn> ObjFunc<O, C> {
    pub fn new(fobj: O) -> Self {
        ObjFunc {
            fobj,
            fcstrs: vec![],
        }
    }

    pub fn subject_to(mut self, fcstrs: Vec<C>) -> Self {
        self.fcstrs = fcstrs;
        self
    }
}

impl<O: GroupFunc, C: CstrFn> CostFunction for ObjFunc<O, C> {
    /// Type of the parameter vector
    type Param = Array2<f64>;
    /// Type of the return value computed by the cost function
    type Output = Array2<f64>;

    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        // Evaluate objective function
        Ok((self.fobj)(&p.view()))
    }
}

impl<O: GroupFunc, C: CstrFn> DomainConstraints<C> for ObjFunc<O, C> {
    fn fn_constraints(&self) -> &[impl CstrFn] {
        &self.fcstrs
    }
}

/// An enumeration to define the type of an input variable component
/// with its domain definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum XType {
    /// Continuous variable in [lower bound, upper bound]
    #[deprecated(note = "Use `Xtype::Float` instead", since = "0.30.0")]
    Cont(f64, f64),
    /// Continuous variable in [lower bound, upper bound]
    Float(f64, f64),
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
    fn set_n_clusters(&mut self, n_clusters: NbClusters);

    /// Sets the mode of recombination to get the output prediction from experts prediction
    fn set_recombination(&mut self, recombination: Recombination<f64>);

    /// Sets the hyperparameters tuning strategy
    fn set_theta_tunings(&mut self, theta_tunings: &[ThetaTuning<f64>]);

    /// Set likelihood optimization parameters
    fn set_optim_params(&mut self, n_start: usize, max_eval: usize);

    /// Train the surrogate with given training dataset (x, y)
    fn train(
        &self,
        xt: ArrayView2<f64>,
        yt: ArrayView1<f64>,
    ) -> Result<Box<dyn MixtureGpSurrogate>>;

    /// Train the surrogate with given training dataset (x, y) and given clustering
    fn train_on_clusters(
        &self,
        xt: ArrayView2<f64>,
        yt: ArrayView1<f64>,
        clustering: &Clustering,
    ) -> Result<Box<dyn MixtureGpSurrogate>>;
}

/// A trait for functions used by internal optimizers
/// Functions are expected to be defined as `g(x, g, u)` where
/// * `x` is the input information,
/// * `g` an optional gradient information to be updated if present
/// * `u` information provided by the user
#[cfg(not(feature = "nlopt"))]
pub trait ObjFn<U>: Fn(&[f64], Option<&mut [f64]>, &mut U) -> f64 {}
#[cfg(feature = "nlopt")]
use nlopt::ObjFn;

#[cfg(not(feature = "nlopt"))]
impl<T, U> ObjFn<U> for T where T: Fn(&[f64], Option<&mut [f64]>, &mut U) -> f64 {}

/// A function trait for domain constraints used by the internal optimizer
/// It is a specialized version of [`ObjFn`] with [`InfillObjData`] as user information
pub trait CstrFn: Clone + ObjFn<InfillObjData<f64>> + Sync {}
impl<T> CstrFn for T where T: Clone + ObjFn<InfillObjData<f64>> + Sync {}

/// A function type for domain constraints which will be used by the internal optimizer
/// which is the default value for [`crate::EgorFactory`] generic `C` parameter.
pub type Cstr = fn(&[f64], Option<&mut [f64]>, &mut InfillObjData<f64>) -> f64;

/// Data used by internal infill criteria optimization
/// Internally this type is used to carry the information required to
/// compute the various infill criteria implemented by [`crate::Egor`].
///
/// See [`crate::criteria`]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InfillObjData<F: Float> {
    pub fmin: F,
    pub xbest: Vec<F>,
    pub scale_infill_obj: F,
    pub scale_cstr: Option<Array1<F>>,
    pub scale_wb2: F,
}

impl<F: Float> Default for InfillObjData<F> {
    fn default() -> Self {
        Self {
            fmin: F::infinity(),
            xbest: vec![],
            scale_infill_obj: F::one(),
            scale_cstr: None,
            scale_wb2: F::one(),
        }
    }
}
