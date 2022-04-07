//! A module for correlation models which implements PLS weighting used by GP models.
//! The following kernels are implemented:
//! * squared exponential,
//! * absolute exponential,
//! * matern 3/2,
//! * matern 5/2.

use linfa::Float;
use ndarray::{Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

/// A trait for using a correlation model in GP regression
pub trait CorrelationModel<F: Float>: Clone + Copy + Default {
    /// Use the correlation model to compute correlation matrix given
    /// `theta` parameters, distances `d` between data points and PLS `weights`.
    fn apply(
        &self,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F>;
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(into = "String")]
#[serde(try_from = "String")]

/// Squared exponential correlation models
pub struct SquaredExponentialCorr();

impl From<SquaredExponentialCorr> for String {
    fn from(_item: SquaredExponentialCorr) -> String {
        "SquaredExponential".to_string()
    }
}

impl TryFrom<String> for SquaredExponentialCorr {
    type Error = &'static str;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s == "SquaredExponential" {
            Ok(Self::default())
        } else {
            Err("Bad string value for SquaredExponentialCorr, should be \'SquaredExponential\'")
        }
    }
}

impl<F: Float> CorrelationModel<F> for SquaredExponentialCorr {
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
/// Absolute exponential correlation models
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(into = "String")]
#[serde(try_from = "String")]
pub struct AbsoluteExponentialCorr();

impl From<AbsoluteExponentialCorr> for String {
    fn from(_item: AbsoluteExponentialCorr) -> String {
        "AbsoluteExponential".to_string()
    }
}

impl TryFrom<String> for AbsoluteExponentialCorr {
    type Error = &'static str;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s == "AbsoluteExponential" {
            Ok(Self::default())
        } else {
            Err("Bad string value for AbsoluteExponentialCorr, should be \'AbsoluteExponential\'")
        }
    }
}

impl<F: Float> CorrelationModel<F> for AbsoluteExponentialCorr {
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

/// Matern 3/2 correlation model
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(into = "String")]
#[serde(try_from = "String")]
pub struct Matern32Corr();

impl From<Matern32Corr> for String {
    fn from(_item: Matern32Corr) -> String {
        "Matern32".to_string()
    }
}

impl TryFrom<String> for Matern32Corr {
    type Error = &'static str;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s == "Matern32" {
            Ok(Self::default())
        } else {
            Err("Bad string value for Matern32Corr, should be \'Matern32\'")
        }
    }
}

impl<F: Float> CorrelationModel<F> for Matern32Corr {
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

/// Matern 5/2 correlation model
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(into = "String")]
#[serde(try_from = "String")]
pub struct Matern52Corr();

impl From<Matern52Corr> for String {
    fn from(_item: Matern52Corr) -> String {
        "Matern52".to_string()
    }
}

impl TryFrom<String> for Matern52Corr {
    type Error = &'static str;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s == "Matern52" {
            Ok(Self::default())
        } else {
            Err("Bad string value for Matern52Corr, should be \'Matern52\'")
        }
    }
}

impl<F: Float> CorrelationModel<F> for Matern52Corr {
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
        let res = SquaredExponentialCorr::default().apply(&arr1(&[0.1]), &dm.d, &array![[1.]]);
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
        let res = SquaredExponentialCorr::default().apply(
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
            Matern32Corr::default().apply(&arr1(&[1., 2.]), &dm.d, &array![[1., 0.], [0., 1.]]);
        let expected = array![[1.08539595e-03], [1.10776401e-07], [1.08539595e-03]];
        assert_abs_diff_eq!(res, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_matern52_2d() {
        let xt = array![[0., 1.], [2., 3.], [4., 5.]];
        let dm = DistanceMatrix::new(&xt);
        let res =
            Matern52Corr::default().apply(&arr1(&[1., 2.]), &dm.d, &array![[1., 0.], [0., 1.]]);
        let expected = array![[6.62391590e-04], [1.02117882e-08], [6.62391590e-04]];
        assert_abs_diff_eq!(res, expected, epsilon = 1e-6);
    }
}
