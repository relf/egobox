use std::convert::TryFrom;

use linfa::Float;
use ndarray::{Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use serde::{Deserialize, Serialize};

pub trait CorrelationModel<F: Float>: Clone + Copy + Default + Serialize {
    fn apply(
        &self,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F>;
}

#[derive(Clone, Copy, Default, Serialize, Deserialize, Debug)]
#[serde(into = "String")]
pub struct SquaredExponentialKernel();

impl From<SquaredExponentialKernel> for String {
    fn from(_item: SquaredExponentialKernel) -> String {
        "SquaredExponential".to_string()
    }
}

impl TryFrom<String> for SquaredExponentialKernel {
    type Error = &'static str;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s == "SquaredExponential" {
            Ok(Self::default())
        } else {
            Err("Bad string value for SquaredExponentialKernel, should be \'SquaredExponential\'")
        }
    }
}

impl<F: Float> CorrelationModel<F> for SquaredExponentialKernel {
    fn apply(
        &self,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let wd = d.mapv(|v| v * v).dot(&weights.mapv(|v| v * v));
        let theta_r = theta.to_owned().insert_axis(Axis(0));
        let r = (theta_r * wd).sum_axis(Axis(1)).mapv(|v| F::exp(-v));
        r.into_shape((d.nrows(), 1)).unwrap()
    }
}

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
#[serde(into = "String")]
pub struct AbsoluteExponentialKernel();

impl From<AbsoluteExponentialKernel> for String {
    fn from(_item: AbsoluteExponentialKernel) -> String {
        "AbsoluteExponential".to_string()
    }
}

impl<F: Float> CorrelationModel<F> for AbsoluteExponentialKernel {
    fn apply(
        &self,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let wd = d.mapv(|v| v.abs()).dot(&weights.mapv(|v| v.abs()));
        let theta_r = theta.to_owned().insert_axis(Axis(0));
        let r = (theta_r * wd).sum_axis(Axis(1)).mapv(|v| F::exp(-v));
        r.into_shape((d.nrows(), 1)).unwrap()
    }
}

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
#[serde(into = "String")]
pub struct Matern32Kernel();

impl From<Matern32Kernel> for String {
    fn from(_item: Matern32Kernel) -> String {
        "Matern32".to_string()
    }
}

impl<F: Float> CorrelationModel<F> for Matern32Kernel {
    fn apply(
        &self,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let theta = theta.to_owned().insert_axis(Axis(0));
        let wd = d.mapv(|v| v.abs()).dot(&weights.mapv(|v| v.abs()));
        let theta_wd = (theta.to_owned() * &wd).mapv(|v| F::cast(3).sqrt() * v);
        let a = theta_wd
            .to_owned()
            .mapv(|v| F::one() + v)
            .map_axis(Axis(1), |row| row.product());
        let b = theta_wd.sum_axis(Axis(1)).mapv(|v| F::exp(-v));
        let r = a * b;
        r.into_shape((d.nrows(), 1)).unwrap()
    }
}

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
#[serde(into = "String")]
pub struct Matern52Kernel();

impl From<Matern52Kernel> for String {
    fn from(_item: Matern52Kernel) -> String {
        "Matern52".to_string()
    }
}

impl<F: Float> CorrelationModel<F> for Matern52Kernel {
    fn apply(
        &self,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let theta = theta.to_owned().insert_axis(Axis(0));
        let wd = d.mapv(|v| v.abs()).dot(&weights.mapv(|v| v.abs()));
        let theta_wd = (theta.to_owned() * &wd).mapv(|v| F::cast(5.).sqrt() * v);
        let a = theta_wd
            .to_owned()
            .mapv(|v| (F::one() + v + v * v / F::cast(3.)))
            .map_axis(Axis(1), |row| row.product());
        let b = theta_wd.sum_axis(Axis(1)).mapv(|v| F::exp(-v));
        let r = a * b;
        r.into_shape((d.nrows(), 1)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::DistanceMatrix;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, array};

    #[test]
    fn test_squared_exponential() {
        let xt = array![[4.5], [1.2], [2.0], [3.0], [4.0]];
        let dm = DistanceMatrix::new(&xt);
        let res = SquaredExponentialKernel::default().apply(&arr1(&[0.1]), &dm.d, &array![[1.]]);
        let expected = array![
            [0.336552878364737],
            [0.5352614285189903],
            [0.7985162187593771],
            [0.9753099120283326],
            [0.9380049995307295],
            [0.7232502423798424],
            [0.4565760496233148],
            [0.9048374180359595],
            [0.6703200460356393],
            [0.9048374180359595]
        ];
        assert_abs_diff_eq!(res, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_squared_exponential_2d() {
        let xt = array![[0., 1.], [2., 3.], [4., 5.]];
        let dm = DistanceMatrix::new(&xt);
        dbg!(&dm);
        let res = SquaredExponentialKernel::default().apply(
            &arr1(&[1., 2.]),
            &dm.d,
            &array![[1., 0.], [0., 1.]],
        );
        let expected = array![[6.14421235e-06], [1.42516408e-21], [6.14421235e-06]];
        assert_abs_diff_eq!(res, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_matern32_2d() {
        let xt = array![[0., 1.], [2., 3.], [4., 5.]];
        let dm = DistanceMatrix::new(&xt);
        dbg!(&dm);
        let res =
            Matern32Kernel::default().apply(&arr1(&[1., 2.]), &dm.d, &array![[1., 0.], [0., 1.]]);
        let expected = array![[1.08539595e-03], [1.10776401e-07], [1.08539595e-03]];
        assert_abs_diff_eq!(res, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_matern52_2d() {
        let xt = array![[0., 1.], [2., 3.], [4., 5.]];
        let dm = DistanceMatrix::new(&xt);
        let res =
            Matern52Kernel::default().apply(&arr1(&[1., 2.]), &dm.d, &array![[1., 0.], [0., 1.]]);
        let expected = array![[6.62391590e-04], [1.02117882e-08], [6.62391590e-04]];
        assert_abs_diff_eq!(res, expected, epsilon = 1e-6);
    }
}
