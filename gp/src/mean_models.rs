use linfa::Float;
use ndarray::{concatenate, s, Array2, ArrayBase, Axis, Data, Ix2};

pub trait RegressionModel<F: Float>: Clone + Copy {
    fn apply(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F>;
}

#[derive(Clone, Copy)]
pub struct ConstantMean();

impl<F: Float> RegressionModel<F> for ConstantMean {
    fn apply(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        Array2::<F>::ones((x.nrows(), 1))
    }
}

impl ConstantMean {
    pub fn default() -> Self {
        Self {}
    }
}

#[derive(Clone, Copy)]
pub struct LinearMean();

impl<F: Float> RegressionModel<F> for LinearMean {
    fn apply(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        let res = concatenate![Axis(1), Array2::ones((x.nrows(), 1)), x.to_owned()];
        res
    }
}

impl LinearMean {
    pub fn default() -> Self {
        Self {}
    }
}

#[derive(Clone, Copy)]
pub struct QuadraticMean();

impl<F: Float> RegressionModel<F> for QuadraticMean {
    fn apply(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        let mut res = concatenate![Axis(1), Array2::ones((x.nrows(), 1)), x.to_owned()];
        for k in 0..x.ncols() {
            let part = x.slice(s![.., k..]).to_owned() * x.slice(s![.., k..k + 1]);
            res = concatenate![Axis(1), res, part]
        }
        res
    }
}

impl QuadraticMean {
    pub fn default() -> Self {
        Self {}
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_quadratic() {
        let a = array![[1., 2., 3.], [3., 4., 5.]];
        let actual = QuadraticMean::default().apply(&a);
        let expected = array![
            [1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0],
            [1.0, 3.0, 4.0, 5.0, 9.0, 12.0, 15.0, 16.0, 20.0, 25.0]
        ];
        assert_abs_diff_eq!(expected, actual);
    }
}
