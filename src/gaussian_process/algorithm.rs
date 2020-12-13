use crate::errors::{EgoboxError, Result};
use crate::gaussian_process::{CorrelationModel, GpHyperParams, RegressionModel};
use crate::utils::{DistanceMatrix, NormalizedMatrix};
use ndarray::{arr1, s, Array1, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_einsum_beta::*;
use ndarray_linalg::cholesky::*;
use ndarray_linalg::qr::*;
use ndarray_linalg::svd::*;
use ndarray_linalg::triangular::*;
use nlopt::*;

pub struct GpInnerParams {
    /// Gaussian process variance
    sigma2: Array1<f64>,
    /// Generalized least-squares regression weights for Universal Kriging or given beta0 for Ordinary Kriging
    beta: Array2<f64>,
    /// Gaussian Process weights
    gamma: Array2<f64>,
    /// Cholesky decomposition of the correlation matrix [R]
    r_chol: Array2<f64>,
    /// Solution of the linear equation system : [R] x Ft = y
    ft: Array2<f64>,
    /// R upper triangle matrix of QR decomposition of the matrix Ft
    ft_qr_r: Array2<f64>,
}

impl Default for GpInnerParams {
    fn default() -> Self {
        Self {
            sigma2: Array1::zeros(1),
            beta: Array2::zeros((1, 1)),
            gamma: Array2::zeros((1, 1)),
            r_chol: Array2::zeros((1, 1)),
            ft: Array2::zeros((1, 1)),
            ft_qr_r: Array2::zeros((1, 1)),
        }
    }
}

/// Gaussian
pub struct GaussianProcess<Mean: RegressionModel, Kernel: CorrelationModel> {
    /// Parameter of the autocorrelation model
    theta: Array1<f64>,
    /// Regression model
    mean: Mean,
    /// Correlation kernel
    kernel: Kernel,
    /// Gaussian process internal fitted params
    inner_params: GpInnerParams,
    /// Training inputs
    xtrain: NormalizedMatrix<f64>,
    /// Training outputs
    ytrain: NormalizedMatrix<f64>,
}

impl<Mean: RegressionModel, Kernel: CorrelationModel> GaussianProcess<Mean, Kernel> {
    pub fn params<NewMean: RegressionModel, NewKernel: CorrelationModel>(
        mean: NewMean,
        kernel: NewKernel,
    ) -> GpHyperParams<NewMean, NewKernel> {
        GpHyperParams::new(mean, kernel)
    }

    pub fn predict_values(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Result<Array2<f64>> {
        let corr = self._compute_correlation(&x);
        // Compute the mean at x
        let f = self.mean.eval(x);
        // Scaled predictor
        let y_ = &f.dot(&self.inner_params.beta) + &corr.dot(&self.inner_params.gamma);
        // Predictor
        Ok(&y_ * &self.ytrain.std + &self.ytrain.mean)
    }

    pub fn predict_variances(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let corr = self._compute_correlation(&x);
        let inners = &self.inner_params;

        let corr_t = corr.t().to_owned();
        let rt = inners
            .r_chol
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &corr_t)
            .unwrap();
        let lhs = inners.ft.t().dot(&rt) - self.mean.eval(x).t();
        let u = inners
            .ft_qr_r
            .t()
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &lhs)
            .unwrap();

        let a = &inners.sigma2;
        let b = 1.0 - rt.mapv(|v| v * v).sum_axis(Axis(0)) + u.mapv(|v| v * v).sum_axis(Axis(0));
        let mse = einsum("i,j->ji", &[a, &b])
            .unwrap()
            .into_shape((x.shape()[0], 1))
            .unwrap();

        // Mean Squared Error might be slightly negative depending on
        // machine precision: set to zero in that case
        Ok(mse.mapv(|v| if v < 0. { 0. } else { v }))
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
        let r = self.kernel.eval(&self.theta, &dx);
        r.into_shape((n_obs, nt)).unwrap().to_owned()
    }
}

impl<Mean: RegressionModel, Kernel: CorrelationModel> GpHyperParams<Mean, Kernel> {
    pub fn fit(
        self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<GaussianProcess<Mean, Kernel>> {
        let xtrain = NormalizedMatrix::new(x);
        let ytrain = NormalizedMatrix::new(y);

        let theta0 = Array1::from_elem(xtrain.ncols(), self.initial_theta());
        let x_distances = DistanceMatrix::new(&xtrain.data);
        let fx = self.mean().eval(x);
        let y_train = ytrain.clone();
        let base: f64 = 10.;
        let objfn = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            let theta =
                Array1::from_shape_vec((x.len(),), x.iter().map(|v| base.powf(*v)).collect())
                    .unwrap();
            let rxx = self.kernel().eval(&theta, &x_distances.d);
            match reduced_likelihood(&fx, &rxx, &x_distances, &y_train) {
                Ok(r) => {
                    // println!("GP lkh OK: {}", -r.value);
                    -r.0
                }
                Err(_) => {
                    // println!("GP lkh ERROR: {:?}", err);
                    f64::INFINITY
                }
            }
        };

        let opt_theta;
        {
            // block to drop optimizer and allow self.kernel borrowing after
            let mut optimizer = Nlopt::new(
                Algorithm::Cobyla,
                x_distances.n_features,
                objfn,
                Target::Minimize,
                (),
            );
            let mut index;
            for i in 0..theta0.len() {
                index = i; // cannot use i in closure directly: it is undefined in closures when compiling in release mode.
                let cstr_low =
                    |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
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
            let mut theta_vec = theta0.mapv(f64::log10).into_raw_vec();
            optimizer.set_initial_step1(0.5).unwrap();
            optimizer
                .set_maxeval(10 * x_distances.n_features as u32)
                .unwrap();
            let res = optimizer.optimize(&mut theta_vec);
            if let Err(e) = res {
                println!("ERROR OPTIM in GP {:?}", e);
            }
            opt_theta = arr1(&theta_vec).mapv(|v| base.powf(v));
        }

        let rxx = self.kernel().eval(&opt_theta, &x_distances.d);
        let (_, inner_params) = reduced_likelihood(&fx, &rxx, &x_distances, &ytrain)?;
        Ok(GaussianProcess {
            theta: opt_theta,
            mean: *self.mean(),
            kernel: *self.kernel(),
            inner_params,
            xtrain,
            ytrain,
        })
    }
}

pub fn reduced_likelihood(
    fx: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    rxx: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    x_distances: &DistanceMatrix<f64>,
    ytrain: &NormalizedMatrix<f64>,
) -> Result<(f64, GpInnerParams)> {
    let nugget = 10.0 * f64::EPSILON;

    // Set up R
    let mut r_mx: Array2<f64> = Array2::<f64>::eye(x_distances.n_obs).mapv(|v| (v + v * nugget));
    for (i, ij) in x_distances.d_indices.outer_iter().enumerate() {
        r_mx[[ij[0], ij[1]]] = rxx[[i, 0]];
        r_mx[[ij[1], ij[0]]] = rxx[[i, 0]];
    }

    // R cholesky decomposition
    let r_chol = r_mx.cholesky(UPLO::Lower)?;

    // Solve generalized least squared problem
    let ft = r_chol
        .solve_triangular(UPLO::Lower, Diag::NonUnit, &fx.to_owned())
        .unwrap();
    let (ft_qr_q, ft_qr_r) = ft.qr().unwrap();

    // Check whether we have an ill-conditionned problem
    let (_, sv_qr_r, _) = ft_qr_r.svd(false, false).unwrap();
    let cond_ft = sv_qr_r[sv_qr_r.len() - 1] / sv_qr_r[0];
    if cond_ft < 1e-10 {
        let (_, sv_f, _) = &fx.svd(false, false).unwrap();
        let cond_fx = sv_f[0] / sv_f[sv_f.len() - 1];
        if cond_fx > 1e15 {
            return Err(EgoboxError::LikelihoodComputationError(
                "F is too ill conditioned. Poor combination \
                of regression model and observations."
                    .to_string(),
            ));
        } else {
            // ft is too ill conditioned, get out (try different theta)
            return Err(EgoboxError::LikelihoodComputationError(
                "ft is too ill conditioned, try another theta again".to_string(),
            ));
        }
    }

    let yt = r_chol.solve_triangular(UPLO::Lower, Diag::NonUnit, &ytrain.data)?;
    let beta = ft_qr_r.solve_triangular(UPLO::Upper, Diag::NonUnit, &ft_qr_q.t().dot(&yt))?;

    let rho = yt - ft.dot(&beta);
    let gamma = r_chol
        .t()
        .solve_triangular(UPLO::Upper, Diag::NonUnit, &rho)?;

    // The determinant of R is equal to the squared product of
    // the diagonal elements of its Cholesky decomposition r_chol
    let exp = 2.0 / x_distances.n_obs as f64;
    let mut det_r = 1.0;
    for v in r_chol.diag().mapv(|v| v.powf(exp)).iter() {
        det_r *= v;
    }

    // Reduced likelihood
    let rho_sqr = rho.map(|v| v * v);
    let sigma2 = rho_sqr.sum_axis(Axis(0)) / x_distances.n_obs as f64;
    Ok((
        -sigma2.sum() * det_r,
        GpInnerParams {
            sigma2: sigma2 * &ytrain.std.mapv(|v| v * v),
            beta,
            gamma,
            r_chol,
            ft,
            ft_qr_r,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian_process::{
        correlation_models::SquaredExponentialKernel, mean_models::ConstantMean,
    };
    use approx::assert_abs_diff_eq;
    use ndarray::{arr2, array};

    #[test]
    fn test_gp_fit_and_predict() {
        let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
        let yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];
        let gp = GaussianProcess::<ConstantMean, SquaredExponentialKernel>::params(
            ConstantMean::new(),
            SquaredExponentialKernel::new(),
        )
        .fit(&xt, &yt)
        .expect("GP fit error");
        let expected = 5.62341325;
        assert_abs_diff_eq!(expected, gp.theta[0], epsilon = 1e-6);
        let yvals = gp
            .predict_values(&arr2(&[[1.0], [2.1]]))
            .expect("prediction error");
        let expected_y = arr2(&[[0.6856779931432053], [1.4484644169993859]]);
        assert_abs_diff_eq!(expected_y, yvals, epsilon = 1e-6);
    }
    #[test]
    fn test_train_and_predict_variances() {
        let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
        let yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];
        let gp = GaussianProcess::<ConstantMean, SquaredExponentialKernel>::params(
            ConstantMean::new(),
            SquaredExponentialKernel::new(),
        )
        .fit(&xt, &yt)
        .expect("GP fit error");
        let expected = 5.62341325;
        assert_abs_diff_eq!(expected, gp.theta[0], epsilon = 1e-6);
        let yvars = gp
            .predict_variances(&arr2(&[[1.0], [2.1]]))
            .expect("prediction error");
        let expected_vars = arr2(&[[0.03422835527498675], [0.014105203477142668]]);
        assert_abs_diff_eq!(expected_vars, yvars, epsilon = 1e-6);
    }
}
