use crate::correlation_models::CorrelationModel;
use crate::errors::{GpError, Result};
use crate::mean_models::RegressionModel;
use crate::parameters::GpParams;
use crate::utils::{DistanceMatrix, NormalizedMatrix};
use doe::{SamplingMethod, LHS};
use linfa::{traits::Fit, Dataset};
use linfa_pls::PlsRegression;
use ndarray::{arr1, s, Array1, Array2, ArrayBase, Axis, Data, Ix2, Zip};
use ndarray_einsum_beta::*;
use ndarray_linalg::cholesky::*;
use ndarray_linalg::qr::*;
use ndarray_linalg::svd::*;
use ndarray_linalg::triangular::*;
use ndarray_stats::QuantileExt;
use nlopt::*;

const LOG10_20: f64 = 1.301_029_995_663_981_3; //f64::log10(20.);

#[derive(Default)]
pub struct GpInnerParams {
    /// Gaussian process variance
    sigma2: Array1<f64>,
    /// Generalized least-squares regression weights for Universal Kriging or given beta0 for Ordinary Kriging
    beta: Array2<f64>,
    /// Gaussian Process weights
    gamma: Array2<f64>,
    /// Cholesky decomposition of the correlation matrix \[R\]
    r_chol: Array2<f64>,
    /// Solution of the linear equation system : \[R\] x Ft = y
    ft: Array2<f64>,
    /// R upper triangle matrix of QR decomposition of the matrix Ft
    ft_qr_r: Array2<f64>,
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
    /// Weights in case of KPLS dimension reduction coming from PLS regression (orig_dim, kpls_dim)
    w_star: Array2<f64>,
    /// Training inputs
    xtrain: NormalizedMatrix<f64>,
    /// Training outputs
    ytrain: NormalizedMatrix<f64>,
}

impl<Mean: RegressionModel, Kernel: CorrelationModel> GaussianProcess<Mean, Kernel> {
    pub fn params<NewMean: RegressionModel, NewKernel: CorrelationModel>(
        mean: NewMean,
        kernel: NewKernel,
    ) -> GpParams<NewMean, NewKernel> {
        GpParams::new(mean, kernel)
    }

    pub fn predict_values(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Result<Array2<f64>> {
        let corr = self._compute_correlation(&x);
        // Compute the mean at x
        let f = self.mean.apply(x);
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
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &corr_t)?;
        let lhs = inners.ft.t().dot(&rt) - self.mean.apply(x).t();
        let u = inners
            .ft_qr_r
            .t()
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &lhs)?;

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
        let r = self.kernel.apply(&self.theta, &dx, &self.w_star);
        r.into_shape((n_obs, nt)).unwrap().to_owned()
    }
}

impl<Mean: RegressionModel, Kernel: CorrelationModel> GpParams<Mean, Kernel> {
    pub fn fit(
        self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<GaussianProcess<Mean, Kernel>> {
        self.validate(x.ncols())?;

        let xtrain = NormalizedMatrix::new(x);
        let ytrain = NormalizedMatrix::new(y);

        let mut w_star = Array2::eye(x.ncols());
        if let Some(n_components) = self.kpls_dim() {
            let ds = Dataset::new(x.to_owned(), y.to_owned());
            let pls = PlsRegression::params(*n_components).fit(&ds)?;
            let (x_rotations, _) = pls.rotations();
            w_star = x_rotations.to_owned();
        };

        let x_distances = DistanceMatrix::new(&xtrain.data);
        let theta0 = Array1::from_elem(w_star.ncols(), self.initial_theta());
        let fx = self.mean().apply(x);
        let y_t = ytrain.clone();
        let base: f64 = 10.;
        let objfn = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            let theta =
                Array1::from_shape_vec((x.len(),), x.iter().map(|v| base.powf(*v)).collect())
                    .unwrap();
            let rxx = self.kernel().apply(&theta, &x_distances.d, &w_star);
            match reduced_likelihood(&fx, rxx, &x_distances, &y_t, self.nugget()) {
                Ok(r) => {
                    // println!("theta={} lkh={}", theta, -r.0);
                    -r.0
                }
                Err(_) => {
                    // println!("GP lkh ERROR: {:?}", err);
                    f64::INFINITY
                }
            }
        };

        // Multistart: user theta0 + 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10.
        let mut theta0s = Array2::zeros((8, theta0.len()));
        theta0s.row_mut(0).assign(&theta0);
        let mut xlimits = Array2::zeros((theta0.len(), 2));
        for mut row in xlimits.genrows_mut() {
            row.assign(&arr1(&[1e-6, 20.]));
        }
        // xlimits.column_mut(0).assign(&Array1::from_elem(theta0.len(), 1e-6));
        // xlimits.column_mut(1).assign(&Array1::from_elem(theta0.len(), 20.));
        let seeds = LHS::new(&xlimits).sample(7);
        Zip::from(theta0s.slice_mut(s![1.., ..]).genrows_mut())
            .and(seeds.genrows())
            .par_apply(|mut theta, row| theta.assign(&row));
        // println!("theta0s = {:?}", theta0s);
        let opt_thetas =
            theta0s.map_axis(Axis(1), |theta| optimize_theta(objfn, &theta.to_owned()));
        // println!("opt_theta = {:?}", opt_thetas);

        let opt_index = opt_thetas.map(|(_, opt_f)| opt_f).argmin().unwrap();
        let opt_theta = &(opt_thetas[opt_index]).0;

        let rxx = self.kernel().apply(opt_theta, &x_distances.d, &w_star);
        let (_, inner_params) = reduced_likelihood(&fx, rxx, &x_distances, &ytrain, self.nugget())?;
        Ok(GaussianProcess {
            theta: opt_theta.to_owned(),
            mean: *self.mean(),
            kernel: *self.kernel(),
            inner_params,
            w_star,
            xtrain,
            ytrain,
        })
    }
}

pub fn optimize_theta<F>(objfn: F, theta0: &Array1<f64>) -> (Array1<f64>, f64)
where
    F: Fn(&[f64], Option<&mut [f64]>, &mut ()) -> f64,
{
    let base: f64 = 10.;
    // block to drop optimizer and allow self.kernel borrowing after
    let mut optimizer = Nlopt::new(Algorithm::Cobyla, theta0.len(), objfn, Target::Minimize, ());
    let mut index;
    for i in 0..theta0.len() {
        index = i; // cannot use i in closure directly: it is undefined in closures when compiling in release mode.
        let cstr_low = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            // -(x[i] - f64::log10(1e-6))
            -x[index] - 6.
        };
        let cstr_up = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            // -(f64::log10(20.) - x[i])
            x[index] - LOG10_20
        };

        optimizer
            .add_inequality_constraint(cstr_low, (), 2e-4)
            .unwrap();
        optimizer
            .add_inequality_constraint(cstr_up, (), 2e-4)
            .unwrap();
    }
    let mut theta_vec = theta0.mapv(f64::log10).into_raw_vec();
    optimizer.set_initial_step1(0.5).unwrap();
    optimizer.set_maxeval(10 * theta0.len() as u32).unwrap();
    optimizer.set_ftol_rel(1e-4).unwrap();
    match optimizer.optimize(&mut theta_vec) {
        Ok((_, fmin)) => (arr1(&theta_vec).mapv(|v| base.powf(v)), fmin),
        Err(_e) => {
            // println!("ERROR OPTIM in GP {:?}", e);
            (arr1(&theta_vec).mapv(|v| base.powf(v)), f64::INFINITY)
        }
    }
}

pub fn reduced_likelihood(
    fx: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    rxx: ArrayBase<impl Data<Elem = f64>, Ix2>,
    x_distances: &DistanceMatrix<f64>,
    ytrain: &NormalizedMatrix<f64>,
    nugget: f64,
) -> Result<(f64, GpInnerParams)> {
    // Set up R
    let mut r_mx: Array2<f64> = Array2::<f64>::eye(x_distances.n_obs).mapv(|v| (v + v * nugget));
    for (i, ij) in x_distances.d_indices.outer_iter().enumerate() {
        r_mx[[ij[0], ij[1]]] = rxx[[i, 0]];
        r_mx[[ij[1], ij[0]]] = rxx[[i, 0]];
    }

    // R cholesky decomposition
    let r_chol = r_mx.cholesky_into(UPLO::Lower)?;

    // Solve generalized least squared problem
    let ft = r_chol
        .solve_triangular_into(UPLO::Lower, Diag::NonUnit, fx.to_owned())
        .unwrap();
    let (ft_qr_q, ft_qr_r) = ft.qr().unwrap();

    // Check whether we have an ill-conditionned problem
    let (_, sv_qr_r, _) = ft_qr_r.svd(false, false).unwrap();
    let cond_ft = sv_qr_r[sv_qr_r.len() - 1] / sv_qr_r[0];
    if cond_ft < 1e-10 {
        let (_, sv_f, _) = &fx.svd(false, false).unwrap();
        let cond_fx = sv_f[0] / sv_f[sv_f.len() - 1];
        if cond_fx > 1e15 {
            return Err(GpError::LikelihoodComputationError(
                "F is too ill conditioned. Poor combination \
                of regression model and observations."
                    .to_string(),
            ));
        } else {
            // ft is too ill conditioned, get out (try different theta)
            return Err(GpError::LikelihoodComputationError(
                "ft is too ill conditioned, try another theta again".to_string(),
            ));
        }
    }
    let yt = r_chol.solve_triangular(UPLO::Lower, Diag::NonUnit, &ytrain.data)?;
    let beta = ft_qr_r.solve_triangular_into(UPLO::Upper, Diag::NonUnit, ft_qr_q.t().dot(&yt))?;

    let rho = yt - ft.dot(&beta);
    let rho_sqr = rho.mapv(|v| v * v).sum_axis(Axis(0));
    let gamma = r_chol
        .t()
        .solve_triangular_into(UPLO::Upper, Diag::NonUnit, rho)?;

    // The determinant of R is equal to the squared product of
    // the diagonal elements of its Cholesky decomposition r_chol
    let n_obs: f64 = x_distances.n_obs as f64;
    let logdet = r_chol.diag().mapv(|v| v.log10()).sum() * 2. / n_obs;
    // Reduced likelihood
    let sigma2 = rho_sqr / n_obs;
    let reduced_likelihood = -n_obs * (sigma2.sum().log10() + logdet);
    Ok((
        reduced_likelihood,
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
    use crate::{correlation_models::*, mean_models::ConstantMean};
    use approx::assert_abs_diff_eq;
    use doe::{SamplingMethod, LHS};
    use ndarray::{arr2, array, Zip};
    use ndarray_npy::{read_npy, write_npy};
    use ndarray_rand::rand::SeedableRng;
    use paste::paste;
    use rand_isaac::Isaac64Rng;

    macro_rules! test_gp {
        ($corr:ident, $expected:expr) => {
            paste! {

                #[test]
                fn [<test_gp_ $corr:lower >]() {
                    let xt = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
                    let yt = array![[0.0], [1.0], [1.5], [0.9], [1.0]];
                    let gp = GaussianProcess::<ConstantMean, [<$corr Kernel>]>::params(
                        ConstantMean::default(),
                        [<$corr Kernel>]::default(),
                    )
                    .set_initial_theta(0.1)
                    .fit(&xt, &yt)
                    .expect("GP fit error");
                    assert_abs_diff_eq!($expected, gp.theta[0], epsilon = 1e-2);

                    let yvals = gp
                        .predict_values(&arr2(&[[1.0], [3.5]]))
                        .expect("prediction error");
                    let expected_y = arr2(&[[1.0], [0.9]]);
                    assert_abs_diff_eq!(expected_y, yvals, epsilon = 1e-1);

                    let yvars = gp
                        .predict_variances(&arr2(&[[1.0], [3.5]]))
                        .expect("prediction error");
                    let expected_vars = arr2(&[[0.], [0.0105914]]);
                    assert_abs_diff_eq!(expected_vars, yvars, epsilon = 0.5);
                }
            }
        };
    }

    test_gp!(SquaredExponential, 1.66);
    test_gp!(AbsoluteExponential, 22.35);
    test_gp!(Matern32, 21.68);
    test_gp!(Matern52, 21.68);

    #[test]
    fn test_kpls() {
        let dims = vec![5, 10, 20, 60];
        let nts = vec![100, 300, 400, 800];

        // for i in 3..dims.len() {
        // for i in 0..dims.len() {
        for i in 0..1 {
            let dim = dims[i];
            let nt = nts[i];

            let griewank = |x: &Array1<f64>| -> f64 {
                let d = Array1::linspace(1., dim as f64, dim).mapv(|v| v.sqrt());
                x.mapv(|v| v * v).sum() / 4000.
                    - (x / &d).mapv(|v| v.cos()).fold(1., |acc, x| acc * x)
                    + 1.0
            };
            let prefix = "gp";
            let xfilename = format!("{}_xt_{}x{}.npy", prefix, nt, dim);
            let yfilename = format!("{}_yt_{}x{}.npy", prefix, nt, 1);

            let xt = match read_npy(&xfilename) {
                Ok(xt) => xt,
                Err(_) => {
                    let lim = array![[-600., 600.]];
                    let xlimits = lim.broadcast((dim, 2)).unwrap();
                    let rng = Isaac64Rng::seed_from_u64(42);
                    let xt = LHS::new(&xlimits).with_rng(rng).sample(nt);
                    write_npy(&xfilename, xt.to_owned()).expect("cannot save xt");
                    xt
                }
            };

            let yt = match read_npy(&yfilename) {
                Ok(yt) => yt,
                Err(_) => {
                    let mut yv: Array1<f64> = Array1::zeros(xt.nrows());
                    Zip::from(&mut yv).and(xt.genrows()).par_apply(|y, x| {
                        *y = griewank(&x.to_owned());
                    });
                    let yt = yv.into_shape((xt.nrows(), 1)).unwrap();
                    write_npy(&yfilename, yt.to_owned()).expect("cannot save yt");
                    yt
                }
            };

            let gp = GaussianProcess::<ConstantMean, SquaredExponentialKernel>::params(
                ConstantMean::default(),
                SquaredExponentialKernel::default(),
            )
            //.with_kpls_dim(1)
            //.with_initial_theta(1.0)
            .fit(&xt, &yt)
            .expect("GP fit error");

            let xtest = Array2::zeros((1, dim));
            let ytest = gp.predict_values(&xtest).expect("prediction error");
            println!("ytest = {}", ytest);
        }
    }
}
