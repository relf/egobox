use ndarray::{s, stack, Array2, ArrayBase, Axis, Data, Ix2};

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
    pub fn default() -> Self {
        Self {}
    }
}

#[derive(Clone, Copy)]
pub struct LinearMean();

impl RegressionModel for LinearMean {
    fn eval(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        stack![Axis(1), Array2::ones((x.nrows(), 1)), x.to_owned()].reversed_axes()
    }
}

impl LinearMean {
    pub fn default() -> Self {
        Self {}
    }
}

#[derive(Clone, Copy)]
pub struct QuadraticMean();

impl RegressionModel for QuadraticMean {
    fn eval(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        let mut res = stack![Axis(1), Array2::ones((x.nrows(), 1)), x.to_owned()];
        for k in 0..x.ncols() {
            let part = x.slice(s![.., k..]).to_owned() * x.slice(s![.., k..k + 1]);
            res = stack![Axis(1), res, part]
        }
        res.reversed_axes()
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
        let actual = QuadraticMean::default().eval(&a);
        let expected = array![
            [1.0, 1.0],
            [1.0, 3.0],
            [2.0, 4.0],
            [3.0, 5.0],
            [1.0, 9.0],
            [2.0, 12.0],
            [3.0, 15.0],
            [4.0, 16.0],
            [6.0, 20.0],
            [9.0, 25.0]
        ];
        assert_abs_diff_eq!(expected, actual);
    }
}
