//! A module for correlation models with PLS weighting to model the error term of the GP model.
//!
//! The following correlation models are implemented:
//! * squared exponential,
//! * absolute exponential,
//! * matern 3/2,
//! * matern 5/2.

use crate::utils::differences;
use linfa::Float;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip};
use ndarray_einsum_beta::einsum;
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::fmt;

/// A trait for using a correlation model in GP regression
pub trait CorrelationModel<F: Float>: Clone + Copy + Default + fmt::Display {
    /// Compute correlation function matrix r(x, x') given distances `d` between x and x',
    /// `theta` parameters, and PLS `weights`, where:
    /// `theta`   : hyperparameters (1xd)
    /// `d`     : distances (nxd)
    /// `weight`: PLS weights (dxh)
    /// where d is the initial dimension and h (<d) is the reduced dimension when PLS is used (kpls_dim)
    fn apply(
        &self,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F>;

    /// Compute jacobian matrix of `r(x, x')` at given `x` given a set of `xtrain` training samples,
    /// `theta` parameters, and PLS `weights`.
    fn jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
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
    ///   h    d
    /// prod prod exp( - theta_l * |d_i . weight_i|^2 )
    ///  l=1  i=1
    fn apply(
        &self,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let mut r = Array2::zeros((d.nrows(), 1));
        Zip::from(r.rows_mut())
            .and(d.rows())
            .for_each(|mut r_i, d_i| {
                let mut coef = F::zero();
                Zip::indexed(&d_i).for_each(|j, d_ij| {
                    Zip::indexed(weights.columns())
                        .for_each(|l, w_l| coef += theta[l] * (w_l[j] * *d_ij).powf(F::cast(2.)))
                });
                r_i[0] = F::exp(-coef)
            });
        r
    }

    fn jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let d = differences(x, xtrain);

        // correlation r
        let mut r = Array2::zeros((d.nrows(), 1));
        Zip::from(r.rows_mut())
            .and(d.rows())
            .for_each(|mut r_i, d_i| {
                let mut coef = F::zero();
                Zip::indexed(&d_i).for_each(|j, d_ij| {
                    Zip::indexed(weights.columns())
                        .for_each(|l, w_l| coef += theta[l] * (w_l[j] * *d_ij).powf(F::cast(2.)))
                });
                r_i[0] = F::exp(-coef)
            });

        println!("d={}", d);
        println!("theta={}", theta);
        println!("weights={}", weights);

        println!("r={}", r);
        let mut dr = Array2::zeros((d.nrows(), d.ncols()));
        Zip::from(dr.rows_mut())
            .and(d.rows())
            .and(r.rows())
            .for_each(|mut dr_i, d_i, r_i| {
                Zip::indexed(&mut dr_i)
                    .and(&d_i)
                    .for_each(|j, dr_ij, d_ij| {
                        let mut coef = F::zero();
                        Zip::indexed(weights.columns())
                            .for_each(|l, w_l| coef += theta[l] * (w_l[j]).powf(F::cast(2.)));
                        coef *= F::cast(-2.);
                        *dr_ij = coef * *d_ij * r_i[0]
                    });
            });
        println!("dr={}", dr);
        // correlation dr/dx(xnorm)
        dr
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
    ///   h    d
    /// prod prod exp( - theta_l * |d_i . weight_i| )
    ///  l=1  i=1
    fn apply(
        &self,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let wd = d.mapv(|v| v.abs()).dot(weights).mapv(|v| v.abs());
        let theta_r = theta.to_owned().insert_axis(Axis(0));
        let r = (theta_r * wd).sum_axis(Axis(1)).mapv(|v| F::exp(-v));
        r.into_shape((d.nrows(), 1)).unwrap()
    }

    fn jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let d = differences(x, xtrain);

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
            .into_shape((xtrain.nrows(), weights.ncols()))
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
    ///   h    d         
    /// prod prod (1 + sqrt(3) * theta_l * |d_i . weight_i|) exp( - sqrt(3) * theta_l * |d_i . weight_i| )
    ///  l=1  i=1
    fn apply(
        &self,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let mut a = Array1::zeros(d.nrows());
        Zip::from(&mut a).and(d.rows()).for_each(|a_i, d_i| {
            let mut coef = F::one();
            Zip::indexed(weights.rows()).for_each(|j, w_j| {
                Zip::indexed(theta).for_each(|l, theta_l| {
                    let v = *theta_l * w_j[l].abs() * d_i[j].abs();
                    coef *= F::one() + F::cast(3.).sqrt() * v;
                })
            });
            *a_i = coef;
        });
        let wd = d.mapv(|v| v.abs()).dot(&weights.mapv(|v| v.abs()));
        let theta_wd = theta.to_owned() * &wd;
        let b = theta_wd
            .sum_axis(Axis(1))
            .mapv(|v| F::exp(-F::cast(3).sqrt() * v));
        let r = a * b;
        r.into_shape((d.nrows(), 1)).unwrap()
    }

    fn jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let d = differences(x, xtrain);

        // correlation r
        let mut a = Array1::zeros(d.nrows());
        Zip::from(&mut a).and(d.rows()).for_each(|a_i, d_i| {
            let mut coef = F::one();
            Zip::indexed(weights.rows()).for_each(|j, w_j| {
                Zip::indexed(theta).for_each(|l, theta_l| {
                    let v = *theta_l * w_j[l].abs() * d_i[j].abs();
                    coef *= F::one() + F::cast(3.).sqrt() * v;
                })
            });
            *a_i = coef;
        });
        let wd = d.mapv(|v| v.abs()).dot(&weights.mapv(|v| v.abs()));
        let theta_wd = theta.to_owned() * &wd;
        let b = theta_wd
            .sum_axis(Axis(1))
            .mapv(|v| F::exp(-F::cast(3).sqrt() * v));

        let sign_d = d.mapv(|v| v.signum());
        println!("sign = {}", sign_d);
        let mut db = Array2::<F>::zeros((xtrain.nrows(), xtrain.ncols()));
        let abs_d = d.mapv(|v| v.abs());
        let abs_w = weights.mapv(|v| v.abs());
        Zip::from(db.rows_mut())
            .and(&a)
            .and(&b)
            .and(sign_d.rows())
            .for_each(|mut db_i, ai, bi, si| {
                Zip::from(&mut db_i)
                    .and(&si)
                    .and(abs_w.rows())
                    .for_each(|db_ij, sij, abs_wj| {
                        let coef = -theta.to_owned().dot(&abs_wj) * F::cast(3.).sqrt();
                        *db_ij = *ai * coef * *sij * *bi;
                    });
            });
        println!("db={}", db);

        let mut da = Array2::<F>::zeros((xtrain.nrows(), xtrain.ncols()));
        Zip::from(da.rows_mut())
            .and(abs_d.rows())
            .and(sign_d.rows())
            .for_each(|mut da_p, abs_d_p, sign_p| {
                Zip::indexed(&mut da_p)
                    .and(&sign_p)
                    .for_each(|i, da_pi, sign_pi| {
                        let mut coef = F::zero();
                        Zip::indexed(abs_w.columns()).for_each(|k, abs_w_k| {
                            let mut ter = F::one();
                            let dev = F::cast(3.).sqrt() * theta[k] * abs_w_k[i] * *sign_pi;
                            println!("dev={}", dev);
                            Zip::indexed(abs_w.rows()).and(abs_d_p).for_each(
                                |j, abs_w_j, abs_d_pj| {
                                    Zip::indexed(abs_w_j).and(theta).for_each(
                                        |l, abs_w_jl, theta_l| {
                                            if l != k || j != i {
                                                let v = *theta_l * *abs_w_jl * *abs_d_pj;
                                                ter *= F::one() + F::cast(3).sqrt() * v
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
        println!("da={}", da);

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
    ///   h    d         
    /// prod prod (1 + sqrt(5) * theta_l * |d_i . weight_i| + (5./3.) * theta_l^2 * |d_i . weight_i|^2) exp( - sqrt(5) * theta_l * |d_i . weight_i| )
    ///  l=1  i=1
    fn apply(
        &self,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let mut a = Array1::zeros(d.nrows());
        Zip::from(&mut a).and(d.rows()).for_each(|a_i, d_i| {
            let mut coef = F::one();
            Zip::indexed(weights.rows()).for_each(|j, w_j| {
                Zip::indexed(theta).for_each(|l, theta_l| {
                    let v = *theta_l * w_j[l].abs() * d_i[j].abs();
                    coef *= F::one() + F::cast(5.).sqrt() * v + F::cast(5. / 3.) * v * v;
                })
            });
            *a_i = coef;
        });
        println!("A={}", a);
        let wd = d.mapv(|v| v.abs()).dot(&weights.mapv(|v| v.abs()));
        let theta_wd = theta.to_owned() * &wd;
        let b = theta_wd
            .sum_axis(Axis(1))
            .mapv(|v| F::exp(-F::cast(5).sqrt() * v));
        let r = a * b;
        r.into_shape((d.nrows(), 1)).unwrap()
    }

    fn jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let d = differences(x, xtrain);

        println!("d={}", d);
        println!("theta={}", theta);
        println!("weights={}", weights);

        // correlation
        let wd = d.mapv(|v| v.abs()).dot(&weights.mapv(|v| v.abs()));

        let theta_wd = theta.to_owned() * &wd;
        let mut a = Array1::zeros(xtrain.nrows());
        Zip::from(&mut a).and(d.rows()).for_each(|a_i, d_i| {
            let mut coef = F::one();
            Zip::indexed(weights.rows()).for_each(|j, w_j| {
                Zip::indexed(theta).for_each(|l, theta_l| {
                    let v = *theta_l * w_j[l].abs() * d_i[j].abs();
                    coef *= F::one() + F::cast(5.).sqrt() * v + F::cast(5. / 3.) * v * v;
                })
            });
            *a_i = coef;
        });
        let b = theta_wd
            .sum_axis(Axis(1))
            .mapv(|v| F::exp(-F::cast(5).sqrt() * v));
        println!("A={}", a);
        println!("B={}", b);
        // correlation dr/dx(xnorm)
        // (x - mean).weights
        // (1, nx).(nx, ncomp) -> shape(1 x ncomp)   (ncomp=nx when no PLS)
        let sign_d = d.mapv(|v| v.signum());
        println!("sign = {}", sign_d);
        let mut db = Array2::<F>::zeros((xtrain.nrows(), xtrain.ncols()));
        let abs_d = d.mapv(|v| v.abs());
        let abs_w = weights.mapv(|v| v.abs());
        Zip::from(db.rows_mut())
            .and(&a)
            .and(&b)
            .and(sign_d.rows())
            .for_each(|mut db_i, ai, bi, si| {
                Zip::from(&mut db_i)
                    .and(&si)
                    .and(abs_w.rows())
                    .for_each(|db_ij, sij, abs_wj| {
                        let coef = -theta.to_owned().dot(&abs_wj) * F::cast(5.).sqrt();
                        *db_ij = *ai * coef * *sij * *bi;
                    });
            });
        println!("db={}", db);

        let mut da = Array2::<F>::zeros((xtrain.nrows(), xtrain.ncols()));
        Zip::from(da.rows_mut())
            .and(abs_d.rows())
            .and(sign_d.rows())
            .for_each(|mut da_p, abs_d_p, sign_p| {
                Zip::indexed(&mut da_p).and(&abs_d_p).and(&sign_p).for_each(
                    |i, da_pi, abs_d_pi, sign_pi| {
                        let mut coef = F::zero();
                        Zip::indexed(abs_w.columns()).for_each(|k, abs_w_k| {
                            let mut ter = F::one();
                            let dev = F::cast(5.).sqrt() * theta[k] * abs_w_k[i] * *sign_pi
                                + F::cast((5. / 3.) * 2.)
                                    * theta[k].powf(F::cast(2.))
                                    * abs_w_k[i].powf(F::cast(2.))
                                    * *sign_pi
                                    * *abs_d_pi;
                            println!("dev={}", dev);
                            Zip::indexed(abs_w.rows()).and(abs_d_p).for_each(
                                |j, abs_w_j, abs_d_pj| {
                                    Zip::indexed(abs_w_j).and(theta).for_each(
                                        |l, abs_w_jl, theta_l| {
                                            if l != k || j != i {
                                                let v = *theta_l * *abs_w_jl * *abs_d_pj;
                                                ter *= F::one()
                                                    + F::cast(5).sqrt() * v
                                                    + F::cast(5. / 3.) * v * v;
                                            }
                                        },
                                    );
                                },
                            );
                            coef += dev * ter;
                        });
                        *da_pi = coef;
                    },
                );
            });
        println!("da={}", da);
        let da = einsum("i,ij->ij", &[&b, &da])
            .unwrap()
            .into_shape((xtrain.nrows(), xtrain.ncols()))
            .unwrap();
        println!("da={}", db.to_owned() + &da);
        db.to_owned() + da
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
        let res = SquaredExponentialCorr::default().apply(&dm.d, &arr1(&[0.1]), &array![[1.]]);
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
            &dm.d,
            &arr1(&[1., 2.]),
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
            Matern32Corr::default().apply(&dm.d, &arr1(&[1., 2.]), &array![[1., 0.], [0., 1.]]);
        let expected = array![[1.08539595e-03], [1.10776401e-07], [1.08539595e-03]];
        assert_abs_diff_eq!(res, expected, epsilon = 1e-6);
    }

    macro_rules! test_correlation {
        ($corr:ident, $kpls:expr) => {
            paste! {
                #[test]
                fn [<test_corr_ $corr:lower _kpls_ $kpls _derivatives>]() {
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
                    println!("xt mean={}", xtrain.mean);
                    println!("xt std={}", xtrain.std);
                    println!("xt ={}", xtrain.data);
                    println!("xnorm={}", xnorm);
                    let (theta, weights) = if $kpls {
                        (array![1.43301257],
                            array![[-0.02701716],
                            [-0.99963497]])
                    } else {
                        (array![0.34599115925909146, 0.32083374253611624],
                         array![[1., 0.], [0., 1.]])
                    };

                    let corr = [< $corr Corr >]::default();
                    let jac = corr.jac(&xnorm, &xtrain.data, &theta, &weights) / &xtrain.std;

                    let xa: f64 = x[0];
                    let xb: f64 = x[1];
                    let e = 1e-5;
                    let x = array![
                        [xa, xb],
                        // [xa + e, xb],
                        // [xa - e, xb],
                        // [xa, xb + e],
                        // [xa, xb - e]
                    ];

                    let mut rxx = Array2::zeros((xtrain.data.nrows(), x.nrows()));
                    Zip::from(rxx.columns_mut())
                        .and(x.rows())
                        .for_each(|mut rxxi, xi| {
                            let xnorm = (xi.to_owned() - &xtrain.mean) / &xtrain.std;
                            let d = differences(&xnorm, &xtrain.data);
                            rxxi.assign(&(corr.apply( &d, &theta, &weights).column(0)));
                        });
                    // let fdiffa = (rxx.column(1).to_owned() - rxx.column(2)).mapv(|v| v / (2. * e));
                    // assert_abs_diff_eq!(fdiffa, jac.column(0), epsilon=1e-6);
                    // let fdiffb = (rxx.column(3).to_owned() - rxx.column(4)).mapv(|v| v / (2. * e));
                    // assert_abs_diff_eq!(fdiffb, jac.column(1), epsilon=1e-6);
                }
            }
        };
    }

    test_correlation!(SquaredExponential, false);
    test_correlation!(AbsoluteExponential, false);
    test_correlation!(Matern32, false);
    test_correlation!(Matern52, false);
    test_correlation!(SquaredExponential, true);
    test_correlation!(AbsoluteExponential, true);
    test_correlation!(Matern32, true);
    test_correlation!(Matern52, true);

    #[test]
    fn test_matern52_2d() {
        let xt = array![[0., 1.], [2., 3.], [4., 5.]];
        let dm = DistanceMatrix::new(&xt);
        let res =
            Matern52Corr::default().apply(&dm.d, &arr1(&[1., 2.]), &array![[1., 0.], [0., 1.]]);
        let expected = array![[6.62391590e-04], [1.02117882e-08], [6.62391590e-04]];
        assert_abs_diff_eq!(res, expected, epsilon = 1e-6);
    }
}
