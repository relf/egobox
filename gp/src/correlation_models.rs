//! A module for correlation models with PLS weighting to model the error term of the GP model.
//!
//! The following correlation models are implemented:
//! * squared exponential,
//! * absolute exponential,
//! * matern 3/2,
//! * matern 5/2.

use linfa::Float;
use ndarray::{Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip};
use ndarray_einsum_beta::einsum;
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::fmt;

use crate::utils::pairwise_differences;

/// A trait for using a correlation model in GP regression
pub trait CorrelationModel<F: Float>: Clone + Copy + Default + fmt::Display {
    /// Compute correlation matrix given `theta` parameters,
    /// distances `d` between x data points and PLS `weights`.
    fn apply(
        &self,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F>;

    fn jac(
        &self,
        xnorm: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F>;
}

/// Squared exponential correlation models
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(into = "String"),
    serde(try_from = "String")
)]
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

    fn jac(
        &self,
        xnorm: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let xn = xnorm.to_owned().insert_axis(Axis(0));
        let d = pairwise_differences(&xn, xtrain);

        // correlation r
        let wd = d.mapv(|v| v * v).dot(&weights.mapv(|v| v * v));
        let theta_r = theta.to_owned().insert_axis(Axis(0));
        let r = (theta_r * wd).sum_axis(Axis(1)).mapv(|v| F::exp(-v));

        // correlation dr/dx(xnorm)
        let wd = d.dot(&weights.mapv(|v| v * v));
        let dr = einsum("j,ij->ij", &[theta, &wd])
            .unwrap()
            .mapv(|v| F::cast(-2) * v);
        einsum("i,ij->ij", &[&r, &dr])
            .unwrap()
            .into_shape((xtrain.nrows(), xtrain.ncols()))
            .unwrap()
    }
}

impl fmt::Display for SquaredExponentialCorr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SquaredExponential")
    }
}

/// Absolute exponential correlation models
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(into = "String"),
    serde(try_from = "String")
)]
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
        let wd = d.mapv(|v| v.abs()).dot(weights).mapv(|v| v.abs());
        let theta_r = theta.to_owned().insert_axis(Axis(0));
        let r = (theta_r * wd).sum_axis(Axis(1)).mapv(|v| F::exp(-v));
        r.into_shape((d.nrows(), 1)).unwrap()
    }

    fn jac(
        &self,
        xnorm: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let xn = xnorm.to_owned().insert_axis(Axis(0));
        let d = pairwise_differences(&xn, xtrain);

        // correlation r
        let wd = (d.mapv(|v| v.abs()).dot(weights)).mapv(|v| v.abs());
        let theta_r = theta.to_owned().insert_axis(Axis(0));
        let r = (theta_r * wd).sum_axis(Axis(1)).mapv(|v| F::exp(-v));

        // correlation dr/dx(xnorm)
        // (x - mean).weights
        // (1, nx).(nx, ncomp) -> shape(1 x ncomp)   (ncomp=nx when no PLS)
        let sign_wd = (d.dot(weights)).mapv(|v| v.signum());

        // - (theta * wd)
        // (ncomp,) * (nx, ncomp)
        let dr = -einsum("j,ij->ij", &[theta, &sign_wd]).unwrap();
        einsum("i,ij->ij", &[&r, &dr])
            .unwrap()
            .into_shape((xtrain.nrows(), xtrain.ncols()))
            .unwrap()
    }
}

impl fmt::Display for AbsoluteExponentialCorr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "AbsoluteExponential")
    }
}

/// Matern 3/2 correlation model
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(into = "String"),
    serde(try_from = "String")
)]
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
        let wd = d.mapv(|v| v.abs()).dot(&weights.mapv(|v| v.abs()));

        let theta_wd = theta.to_owned() * &wd;
        let a = theta_wd
            .to_owned()
            .mapv(|v| F::one() + F::cast(3).sqrt() * v)
            .map_axis(Axis(1), |row| row.product());
        let b = theta_wd
            .sum_axis(Axis(1))
            .mapv(|v| F::exp(-F::cast(3).sqrt() * v));
        let r = a * b;
        r.into_shape((d.nrows(), 1)).unwrap()
    }

    fn jac(
        &self,
        xnorm: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let xn = xnorm.to_owned().insert_axis(Axis(0));
        let d = pairwise_differences(&xn, xtrain);

        // correlation r
        //let theta = theta.to_owned().insert_axis(Axis(0));
        let wd = d.mapv(|v| v.abs()).dot(&weights.mapv(|v| v.abs()));
        let theta_wd = theta.to_owned() * &wd;
        let a = theta_wd
            .to_owned()
            .mapv(|v| F::one() + F::cast(3).sqrt() * v)
            .map_axis(Axis(1), |row| row.product());
        let b = theta_wd
            .sum_axis(Axis(1))
            .mapv(|v| F::exp(-F::cast(3).sqrt() * v));

        // correlation dr/dx(xnorm)
        // (x - mean).weights
        // (1, nx).(nx, ncomp) -> shape(1 x ncomp)   (ncomp=nx when no PLS)
        let sign_wd = (d.dot(weights)).mapv(|v| v.signum());

        let mut db = Array2::<F>::zeros((xtrain.nrows(), xtrain.ncols()));
        let abs_d = d.mapv(|v| v.abs());
        let abs_w = weights.mapv(|v| v.abs());
        Zip::from(db.rows_mut())
            .and(&a)
            .and(&b)
            .and(sign_wd.rows())
            .for_each(|mut db_i, ai, bi, si| {
                Zip::from(&mut db_i)
                    .and(abs_w.rows())
                    .and(&si)
                    .for_each(|db_ij, abs_wi, sij| {
                        let coef = (theta.to_owned() * abs_wi)
                            .mapv(|v| -F::cast(3).sqrt() * v)
                            .sum();
                        *db_ij = *ai * coef * *sij * *bi;
                    });
            });
        let mut da = Array2::<F>::zeros((xtrain.nrows(), xtrain.ncols()));
        Zip::from(da.rows_mut())
            .and(abs_d.rows())
            .and(sign_wd.rows())
            .for_each(|mut da_p, abs_d_p, sign_p| {
                Zip::indexed(&mut da_p)
                    .and(abs_w.rows())
                    .and(&sign_p)
                    .for_each(|i, da_pi, abs_w_i, sign_pi| {
                        let mut coef = F::zero();
                        Zip::indexed(&abs_w_i).for_each(|k, abs_w_ik| {
                            let mut ter = F::one();
                            let dev = F::cast(3).sqrt() * theta[k] * *abs_w_ik * *sign_pi;
                            Zip::indexed(abs_w.rows()).and(abs_d_p).for_each(
                                |j, abs_w_j, abs_d_pj| {
                                    Zip::indexed(abs_w_j).and(theta).for_each(
                                        |l, abs_w_jl, theta_l| {
                                            if l != k || j != i {
                                                ter *= F::one()
                                                    + F::cast(3).sqrt()
                                                        * *theta_l
                                                        * *abs_w_jl
                                                        * *abs_d_pj;
                                            }
                                        },
                                    );
                                },
                            );
                            coef += dev * ter;
                        });
                        *da_pi = coef;
                    });
            });

        let da = einsum("i,ij->ij", &[&b, &da])
            .unwrap()
            .into_shape((xtrain.nrows(), xtrain.ncols()))
            .unwrap();
        db.to_owned() + da
    }
}

impl fmt::Display for Matern32Corr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matern32")
    }
}

/// Matern 5/2 correlation model
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(into = "String"),
    serde(try_from = "String")
)]
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

    fn jac(
        &self,
        _xnorm: &ArrayBase<impl Data<Elem = F>, Ix1>,
        _xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        _theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        _weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        todo!()
    }
}

impl fmt::Display for Matern52Corr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matern52")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{DistanceMatrix, NormalizedMatrix};
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, array};
    use paste::paste;

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

    macro_rules! test_correlation {
        ($corr:ident) => {
            paste! {
                #[test]
                fn [<test_corr_ $corr:lower _derivatives>]() {
                    let x = array![3., 5.];
                    let xt = array![
                        [-9.375, -5.625],
                        [-5.625, -4.375],
                        [9.375, 1.875],
                        [8.125, 5.625],
                        [-4.375, -0.625],
                        [6.875, -3.125],
                        [4.375, 9.375],
                        [3.125, 4.375],
                        [5.625, -8.125],
                        [-8.125, 3.125],
                        [1.875, -6.875],
                        [-0.625, 8.125],
                        [-1.875, -1.875],
                        [0.625, 0.625],
                        [-6.875, -9.375],
                        [-3.125, 6.875]
                    ];
                    let xtrain = NormalizedMatrix::new(&xt);
                    let xnorm = (x.to_owned() - &xtrain.mean) / &xtrain.std;

                    let theta = array![0.34599115925909146, 0.32083374253611624];
                    let weights = array![[1., 0.], [0., 1.]];

                    let corr = [< $corr Corr >]::default();
                    let jac = corr.jac(&xnorm, &xtrain.data, &theta, &weights);

                    let xa: f64 = x[0];
                    let xb: f64 = x[1];
                    let e = 1e-5;
                    let x = array![
                        [xa, xb],
                        [xa + e, xb],
                        [xa - e, xb],
                        [xa, xb + e],
                        [xa, xb - e]
                    ];

                    let mut rxx = Array2::zeros((xtrain.data.nrows(), x.nrows()));
                    Zip::from(rxx.columns_mut())
                        .and(x.rows())
                        .for_each(|mut rxxi, xi| {
                            let xnorm = (xi.to_owned() - &xtrain.mean) / &xtrain.std;
                            let d = pairwise_differences(&xnorm.insert_axis(Axis(0)), &xtrain.data);
                            rxxi.assign(&(corr.apply(&theta, &d, &weights).column(0)));
                        });
                    let fdiffa = (rxx.column(1).to_owned() - rxx.column(2)).mapv(|v| v * xtrain.std[0] / (2. * e));
                    assert_abs_diff_eq!(fdiffa, jac.column(0), epsilon=1e-6);
                    let fdiffb = (rxx.column(3).to_owned() - rxx.column(4)).mapv(|v| v * xtrain.std[1] / (2. * e));
                    assert_abs_diff_eq!(fdiffb, jac.column(1), epsilon=1e-6);
                }
            }
        };
    }

    test_correlation!(SquaredExponential);
    test_correlation!(AbsoluteExponential);
    test_correlation!(Matern32);

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
