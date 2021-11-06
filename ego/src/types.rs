use gp::Float;
use ndarray::Array1;

pub const SQRT_2PI: f64 = 2.5066282746310007;

pub trait ObjFunc: Send + Sync + 'static + Fn(&[f64]) -> f64 {}
impl<T> ObjFunc for T where T: Send + Sync + 'static + Fn(&[f64]) -> f64 {}

#[derive(Debug)]
pub struct OptimResult<F: Float> {
    pub x_opt: Array1<F>,
    pub y_opt: F,
}

#[derive(Debug, PartialEq)]
pub enum AcqStrategy {
    EI,
    WB2,
    WB2S,
}

#[derive(Debug, PartialEq)]
pub enum QEiStrategy {
    KrigingBeliever,
    KrigingBelieverLowerBound,
    KrigingBelieverUpperBound,
    ConstantLiarMinimum,
}

/// A structure to pass data to objective acquisition function
pub struct ObjData<F> {
    pub scale: F,
    pub scale_wb2: Option<F>,
}
