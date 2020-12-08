use ndarray::{Array2, ArrayBase, Data, Ix2};

pub trait RegressionModel: Clone + Copy {
    fn eval(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64>;
}

#[derive(Clone, Copy)]
pub struct ConstantMean();

impl RegressionModel for ConstantMean {
    fn eval(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        Array2::<f64>::ones((x.nrows(), 1))
    }
}

impl ConstantMean {
    pub fn new() -> Self {
        Self {}
    }
}
