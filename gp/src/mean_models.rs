//! A module for regression models to model the mean term of the GP model.
//! In practice small degree (<= 2) polynomial regression models are used,
//! as the gaussian process is then fitted using the [correlated error term](struct.CorrelationModel).
//!
//! The following models are implemented:
//! * constant,
//! * linear,
//! * quadratic

use linfa::Float;
use ndarray::{concatenate, s, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

/// A trait for mean models used in GP regression
pub trait RegressionModel<F: Float>: Clone + Copy + Default {
    /// Compute regression coefficients defining the mean behaviour of the GP model
    /// for the given `x` data points specified as (n, nx) matrix.
    fn apply(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F>;

    /// Compute regression derivative coefficients
    /// at the given `x` data point specified as (nx,) vector.
    fn jac(&self, x: &ArrayBase<impl Data<Elem = F>, Ix1>) -> Array2<F>;
}

/// A constant function as mean of the GP
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(into = "String"),
    serde(try_from = "String")
)]
pub struct ConstantMean();

impl<F: Float> RegressionModel<F> for ConstantMean {
    /// Zero order polynomial (constant) regression model.
    /// regr(x) = [1, ..., 1].T
    fn apply(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        Array2::<F>::ones((x.nrows(), 1))
    }

    fn jac(&self, x: &ArrayBase<impl Data<Elem = F>, Ix1>) -> Array2<F> {
        Array2::<F>::zeros((1, x.len()))
    }
}

impl From<ConstantMean> for String {
    fn from(_item: ConstantMean) -> Self {
        "Constant".to_string()
    }
}

impl TryFrom<String> for ConstantMean {
    type Error = &'static str;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s == "Constant" {
            Ok(Self::default())
        } else {
            Err("Bad string value for ConstantMean, should be \'Constant\'")
        }
    }
}

/// An affine function as mean of the GP
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(into = "String"),
    serde(try_from = "String")
)]
pub struct LinearMean();

impl<F: Float> RegressionModel<F> for LinearMean {
    /// First order polynomial (linear) regression model.
    /// regr(x) = [ 1, x_1, ..., x_n ].T
    fn apply(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        let res = concatenate![Axis(1), Array2::ones((x.nrows(), 1)), x.to_owned()];
        res
    }

    fn jac(&self, x: &ArrayBase<impl Data<Elem = F>, Ix1>) -> Array2<F> {
        let nx = x.len();
        let mut jac = Array2::<F>::zeros((nx + 1, nx));
        jac.slice_mut(s![1.., ..]).assign(&Array2::eye(x.len()));
        jac
    }
}

impl From<LinearMean> for String {
    fn from(_item: LinearMean) -> Self {
        "Linear".to_string()
    }
}

impl TryFrom<String> for LinearMean {
    type Error = &'static str;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s == "Linear" {
            Ok(Self::default())
        } else {
            Err("Bad string value for LinearMean, should be \'Linear\'")
        }
    }
}

/// A 2-degree polynomial as mean of the GP
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(into = "String"),
    serde(try_from = "String")
)]
pub struct QuadraticMean();

impl<F: Float> RegressionModel<F> for QuadraticMean {
    /// Second order polynomial (quadratic) regression model.
    /// regr(x) = [ 1, { x_i, i = 1,...,n }, { x_i * x_j,  (i,j) = 1,...,n  , i > j } ].T
    fn apply(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        let mut res = concatenate![Axis(1), Array2::ones((x.nrows(), 1)), x.to_owned()];
        for k in 0..x.ncols() {
            let part = x.slice(s![.., k..]).to_owned() * x.slice(s![.., k..k + 1]);
            res = concatenate![Axis(1), res, part]
        }
        res
    }

    fn jac(&self, x: &ArrayBase<impl Data<Elem = F>, Ix1>) -> Array2<F> {
        let nx = x.len();
        let mut jac = Array2::<F>::zeros((nx * (nx + 1) / 2 + nx + 1, nx));
        jac.slice_mut(s![1..nx + 1, 0..nx]).assign(&Array2::eye(nx));
        let mut o = 1 + nx;
        let mut p = nx;
        for i in 0..nx {
            let mut part_i = Array2::zeros((p, p));
            part_i.column_mut(0).assign(&x.slice(s![i..]));
            part_i = part_i + Array2::eye(p).mapv(|v: F| v * x[i]);

            jac.slice_mut(s![o..(o + nx - i), i..nx]).assign(&part_i);

            o += p;
            p -= 1;
        }
        jac
    }
}

impl From<QuadraticMean> for String {
    fn from(_item: QuadraticMean) -> Self {
        "Quadratic".to_string()
    }
}

impl TryFrom<String> for QuadraticMean {
    type Error = &'static str;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s == "Quadratic" {
            Ok(Self::default())
        } else {
            Err("Bad string value for QuadraticMean, should be \'Quadratic\'")
        }
    }
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn test_quadratic2() {
        let a = array![[0.], [7.], [25.]];
        let actual = QuadraticMean::default().apply(&a);
        let expected = array![[1., 0., 0.], [1., 7., 49.], [1., 25., 625.]];
        assert_abs_diff_eq!(expected, actual);
    }

    #[test]
    fn test_save_load() {
        let data = r#""ConstantMean""#;
        let v: serde_json::Value = serde_json::from_str(data).unwrap();
        println!("{}", v);
    }

    #[test]
    fn test_quadratic_jac() {
        let expected = array![
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [2., 0., 0.],
            [2., 1., 0.],
            [3., 0., 1.],
            [0., 4., 0.],
            [0., 3., 2.],
            [0., 0., 6.]
        ];
        assert_abs_diff_eq!(expected, QuadraticMean::default().jac(&array![1., 2., 3.]));
    }
}
