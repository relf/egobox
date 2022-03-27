use linfa::Float;
use ndarray::{Array1, Array2, ArrayView2};

pub trait ObjFunc: Send + Sync + 'static + Fn(&[f64]) -> f64 {}
impl<T> ObjFunc for T where T: Send + Sync + 'static + Fn(&[f64]) -> f64 {}

#[derive(Clone, Debug)]
pub struct OptimResult<F: Float> {
    pub x_opt: Array1<F>,
    pub y_opt: Array1<F>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum InfillStrategy {
    EI,
    WB2,
    WB2S,
}

#[derive(Clone, Debug, PartialEq)]
pub enum QEiStrategy {
    KrigingBeliever,
    KrigingBelieverLowerBound,
    KrigingBelieverUpperBound,
    ConstantLiarMinimum,
}

pub struct ObjData<F> {
    pub scale_obj: F,
    pub scale_cstr: Array1<F>,
    pub scale_wb2: F,
}

pub trait GroupFunc: Send + Sync + 'static + Clone + Fn(&ArrayView2<f64>) -> Array2<f64> {}
impl<T> GroupFunc for T where T: Send + Sync + 'static + Clone + Fn(&ArrayView2<f64>) -> Array2<f64> {}

#[derive(Clone, Debug, PartialEq)]
pub enum CstrStatus {
    Respected,
    Violated,
    Active,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Constraint {
    pub value: f64,
    pub status: CstrStatus,
}

#[derive(Clone, Debug, PartialEq)]
pub enum InfillOptimizer {
    Slsqp,
    Cobyla,
}
