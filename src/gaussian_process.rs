use crate::errors::{GpError, Result};
use crate::utils::{constant, squared_exponential, DistanceMatrix, NormalizedMatrix};
use ndarray::{arr1, s, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use ndarray_einsum_beta::*;
use ndarray_linalg::cholesky::*;
use ndarray_linalg::qr::*;
use ndarray_linalg::svd::*;
use ndarray_linalg::triangular::*;
use nlopt::*;

#[derive(Clone)]
pub struct GpHyperParams {
    initial_theta: f64,
}

impl GpHyperParams {
    pub fn new() -> GpHyperParams {
        GpHyperParams {
            initial_theta: 1e-2,
        }
    }

    /// Set initial value for theta hyper parameter.
    ///
    /// During training process, the internal optimization
    /// is started from `initial_theta`.
    pub fn initial_theta(mut self, initial_theta: f64) -> Self {
        self.initial_theta = initial_theta;
        self
    }
}

impl GpHyperParams {
    pub fn fit(
        self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> GaussianProcess {
        let xtrain = NormalizedMatrix::new(x);
        let ytrain = NormalizedMatrix::new(y);

        let theta0 = Array1::from_elem(xtrain.ncols(), self.initial_theta);
        let x_distances = DistanceMatrix::new(&xtrain.data);
        let y_train = ytrain.clone();
        let base: f64 = 10.;
        let objfn = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            let theta =
                Array1::from_shape_vec((x.len(),), x.iter().map(|v| base.powf(*v)).collect())
                    .unwrap();
            match reduced_likelihood(&theta, &x_distances, &y_train) {
                Ok(r) => {
                    // println!("GP lkh OK: {}", -r.value);
                    -r.value
                }
                Err(err) => {
                    // println!("GP lkh ERROR: {:?}", err);
                    f64::INFINITY
                }
            }
        };
        let mut optimizer = Nlopt::new(
            Algorithm::Cobyla,
            x_distances.n_features,
            objfn,
            Target::Minimize,
            (),
        );
        let mut index = 0;
        for i in 0..theta0.len() {
            index = i; // cannot use i in closure directly: it is undefined in closures when compiling in release mode.
            let cstr_low = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
                // -(x[i] - f64::log10(1e-6))
                -x[index] - 6.
            };
            let cstr_up = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
                // -(f64::log10(100.) - x[i])
                x[index] - 2.
            };

            optimizer
                .add_inequality_constraint(cstr_low, (), 1e-2)
                .unwrap();
            optimizer
                .add_inequality_constraint(cstr_up, (), 1e-2)
                .unwrap();
        }
        let mut theta_vec = theta0.mapv(|t| f64::log10(t)).into_raw_vec();
        optimizer.set_initial_step1(0.5).unwrap();
        optimizer
            .set_maxeval(10 * x_distances.n_features as u32)
            .unwrap();
        let res = optimizer.optimize(&mut theta_vec);
        if let Err(e) = res {
            println!("ERROR OPTIM in GP {:?}", e);
        }
        let opt_theta = arr1(&theta_vec).mapv(|v| base.powf(v));
        let likelihood = reduced_likelihood(&opt_theta, &x_distances, &ytrain).unwrap();
        GaussianProcess {
            theta: opt_theta,
            likelihood,
            xtrain,
            ytrain,
        }
    }
}

pub struct GaussianProcess {
    theta: Array1<f64>,
    likelihood: Likelihood,
    xtrain: NormalizedMatrix,
    ytrain: NormalizedMatrix,
}

impl GaussianProcess {
    pub fn params() -> GpHyperParams {
        GpHyperParams::new()
    }

    pub fn predict_values(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        let r = self._compute_correlation(&x);
        // Compute the regression function
        let f = constant(x);
        // Scaled predictor
        let y_ = &f.dot(&self.likelihood.beta) + &r.dot(&self.likelihood.gamma);
        // Predictor
        &y_ * &self.ytrain.std + &self.ytrain.mean
    }

    pub fn predict_variances(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        let r = self._compute_correlation(&x);
        let lh = &self.likelihood;

        let tr = r.t().to_owned();
        let rt = lh
            .c_mx
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &tr)
            .unwrap();
        let lhs = lh.ft_mx.t().dot(&rt) - constant(x).t();
        let u = lh
            .g_mx
            .t()
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &lhs)
            .unwrap();

        let a = &lh.sigma2;
        let b = 1.0 - rt.mapv(|v| v * v).sum_axis(Axis(0)) + u.mapv(|v| v * v).sum_axis(Axis(0));
        let mse = einsum("i,j->ji", &[a, &b])
            .unwrap()
            .into_shape((x.shape()[0], 1))
            .unwrap();

        // Mean Squared Error might be slightly negative depending on
        // machine precision: set to zero in that case
        mse.mapv(|v| if v < 0. { 0. } else { v })
    }
    fn _compute_correlation(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        let n_obs = x.nrows();
        let n_features = x.ncols();

        let xnorm = (x - &self.xtrain.mean) / &self.xtrain.std;
        let nt = self.xtrain.data.nrows();
        // Get pairwise componentwise L1-distances to the input training set
        let mut dx: Array2<f64> = Array2::zeros((nt * n_obs, n_features));
        for (i, xrow) in xnorm.genrows().into_iter().enumerate() {
            let dxrows = &self.xtrain.data - &xrow.into_shape((1, n_features)).unwrap();
            let a = i * nt;
            let b = (i + 1) * nt;
            dx.slice_mut(s![a..b, ..]).assign(&dxrows);
        }
        // Compute the correlation function
        let r = squared_exponential(&self.theta, &dx);
        r.into_shape((n_obs, nt)).unwrap().to_owned()
    }
}

#[derive(Debug)]
pub struct Likelihood {
    value: f64,
    sigma2: Array1<f64>,
    beta: Array2<f64>,
    gamma: Array2<f64>,
    c_mx: Array2<f64>,
    ft_mx: Array2<f64>,
    g_mx: Array2<f64>,
}

pub fn reduced_likelihood(
    theta: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    x_distances: &DistanceMatrix,
    ytrain: &NormalizedMatrix,
) -> Result<Likelihood> {
    let nugget = 10.0 * f64::EPSILON;
    let r = squared_exponential(theta, &x_distances.d);
    let mut r_mx: Array2<f64> = Array2::<f64>::eye(x_distances.n_obs).mapv(|v| (v + v * nugget));
    for (i, ij) in x_distances.d_indices.outer_iter().enumerate() {
        r_mx[[ij[0], ij[1]]] = r[[i, 0]];
        r_mx[[ij[1], ij[0]]] = r[[i, 0]];
    }
    // println!("r_mx = {:?}", r_mx);
    let c_mx = r_mx.cholesky(UPLO::Lower)?;
    let ft_mx = c_mx
        .solve_triangular(UPLO::Lower, Diag::NonUnit, &x_distances.f)
        .unwrap();
    let (q_mx, g_mx) = ft_mx.qr().unwrap();
    let (_, sv_g, _) = g_mx.svd(false, false).unwrap();

    let cond_g_mx = sv_g[sv_g.len() - 1] / sv_g[0];
    if cond_g_mx < 1e-10 {
        let (_, sv_f, _) = x_distances.f.svd(false, false).unwrap();
        let cond_f_mx = sv_f[0] / sv_f[sv_f.len() - 1];
        if cond_f_mx > 1e15 {
            return Err(GpError::LikelihoodError(
                "F is too ill conditioned. Poor combination \
                   of regression model and observations."
                    .to_string(),
            ));
        } else {
            // ft_mx is too ill conditioned, get out (try different theta)
            return Err(GpError::LikelihoodError(
                "ft_mx is too ill conditioned, try another theta again".to_string(),
            ));
        }
    }

    let yt = c_mx
        .solve_triangular(UPLO::Lower, Diag::NonUnit, &ytrain.data)
        .unwrap();
    let beta = g_mx
        .solve_triangular(UPLO::Upper, Diag::NonUnit, &q_mx.t().dot(&yt))
        .unwrap();
    let rho = yt - ft_mx.dot(&beta);

    // The determinant of r_mx is equal to the squared product of the diagonal
    // elements of its Cholesky decomposition c_mx
    let exp = 2.0 / x_distances.n_obs as f64;
    let mut det_r = 1.0;
    for v in c_mx.diag().mapv(|v| v.powf(exp)).iter() {
        det_r *= v;
    }
    let rho_sqr = rho.map(|v| v.powf(2.));
    let sigma2 = rho_sqr.sum_axis(Axis(0)) / x_distances.n_obs as f64;
    let reduced_likelihood = Likelihood {
        value: -sigma2.sum() * det_r,
        sigma2: sigma2 * &ytrain.std.mapv(|v| v.powf(2.0)),
        beta,
        gamma: c_mx
            .t()
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &rho)
            .unwrap(),
        c_mx,
        ft_mx,
        g_mx,
    };
    Ok(reduced_likelihood)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr2, array};

    #[test]
    fn test_gp_fit_and_predict() {
        let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
        let yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];
        let gp = GaussianProcess::params().fit(&xt, &yt);
        let expected = 5.62341325;
        assert_abs_diff_eq!(expected, gp.theta[0], epsilon = 1e-6);
        let yvals = gp.predict_values(&arr2(&[[1.0], [2.1]]));
        let expected_y = arr2(&[[0.6856779931432053], [1.4484644169993859]]);
        assert_abs_diff_eq!(expected_y, yvals, epsilon = 1e-6);
    }
    #[test]
    fn test_train_and_predict_variances() {
        let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
        let yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];
        let gp = GaussianProcess::params().fit(&xt, &yt);
        let expected = 5.62341325;
        assert_abs_diff_eq!(expected, gp.theta[0], epsilon = 1e-6);
        let yvars = gp.predict_variances(&arr2(&[[1.0], [2.1]]));
        let expected_vars = arr2(&[[0.03422835527498675], [0.014105203477142668]]);
        assert_abs_diff_eq!(expected_vars, yvars, epsilon = 1e-6);
    }
}
