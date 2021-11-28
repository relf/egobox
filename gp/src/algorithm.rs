use crate::correlation_models::CorrelationModel;
use crate::errors::{GpError, Result};
use crate::mean_models::RegressionModel;
use crate::parameters::{Float, GpParams};
use crate::utils::{DistanceMatrix, NormalizedMatrix};
use doe::{SamplingMethod, LHS};
use linfa::traits::Fit;
use linfa::Dataset;
use linfa_pls::PlsRegression;
use ndarray::{arr1, s, Array, Array1, Array2, ArrayBase, Axis, Data, Ix2, Zip};
use ndarray_einsum_beta::*;
use ndarray_linalg::cholesky::*;
use ndarray_linalg::qr::*;
use ndarray_linalg::svd::*;
use ndarray_linalg::triangular::*;
use ndarray_rand::rand::SeedableRng;
use ndarray_stats::QuantileExt;
use nlopt::*;
use num_traits;
use rand_isaac::Isaac64Rng;

const LOG10_20: f64 = 1.301_029_995_663_981_3; //f64::log10(20.);

#[derive(Default, Debug)]
pub struct GpInnerParams<F: Float> {
    /// Gaussian process variance
    sigma2: Array1<F>,
    /// Generalized least-squares regression weights for Universal Kriging or given beta0 for Ordinary Kriging
    beta: Array2<F>,
    /// Gaussian Process weights
    gamma: Array2<F>,
    /// Cholesky decomposition of the correlation matrix \[R\]
    r_chol: Array2<F>,
    /// Solution of the linear equation system : \[R\] x Ft = y
    ft: Array2<F>,
    /// R upper triangle matrix of QR decomposition of the matrix Ft
    ft_qr_r: Array2<F>,
}

impl<F: Float> Clone for GpInnerParams<F> {
    fn clone(&self) -> Self {
        Self {
            sigma2: self.sigma2.to_owned(),
            beta: self.beta.to_owned(),
            gamma: self.gamma.to_owned(),
            r_chol: self.r_chol.to_owned(),
            ft: self.ft.to_owned(),
            ft_qr_r: self.ft_qr_r.to_owned(),
        }
    }
}

/// Gaussian
pub struct GaussianProcess<F: Float, Mean: RegressionModel<F>, Kernel: CorrelationModel<F>> {
    /// Parameter of the autocorrelation model
    theta: Array1<F>,
    /// Regression model
    mean: Mean,
    /// Correlation kernel
    kernel: Kernel,
    /// Gaussian process internal fitted params
    inner_params: GpInnerParams<F>,
    /// Weights in case of KPLS dimension reduction coming from PLS regression (orig_dim, kpls_dim)
    w_star: Array2<F>,
    /// Training inputs
    xtrain: NormalizedMatrix<F>,
    /// Training outputs
    ytrain: NormalizedMatrix<F>,
}

impl<F: Float, Mean: RegressionModel<F>, Kernel: CorrelationModel<F>> Clone
    for GaussianProcess<F, Mean, Kernel>
{
    fn clone(&self) -> Self {
        Self {
            theta: self.theta.to_owned(),
            mean: self.mean.clone(),
            kernel: self.kernel.clone(),
            inner_params: self.inner_params.clone(),
            w_star: self.w_star.to_owned(),
            xtrain: self.xtrain.clone(),
            ytrain: self.xtrain.clone(),
        }
    }
}

impl<F: Float, Mean: RegressionModel<F>, Kernel: CorrelationModel<F>>
    GaussianProcess<F, Mean, Kernel>
{
    pub fn params<NewMean: RegressionModel<F>, NewKernel: CorrelationModel<F>>(
        mean: NewMean,
        kernel: NewKernel,
    ) -> GpParams<F, NewMean, NewKernel> {
        GpParams::new(mean, kernel)
    }

    pub fn predict_values(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Result<Array2<F>> {
        let corr = self._compute_correlation(&x);
        // Compute the mean at x
        let f = self.mean.apply(x);
        // Scaled predictor
        let y_ = &f.dot(&self.inner_params.beta) + &corr.dot(&self.inner_params.gamma);
        // Predictor
        Ok(&y_ * &self.ytrain.std + &self.ytrain.mean)
    }

    pub fn predict_variances(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Result<Array2<F>> {
        let corr = self._compute_correlation(&x);
        let inners = &self.inner_params;

        let corr_t = corr.t().to_owned();
        let rt = inners
            .r_chol
            .to_owned()
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &corr_t)?;
        let lhs = inners.ft.t().dot(&rt) - self.mean.apply(x).t();
        let u = inners
            .ft_qr_r
            .to_owned()
            .t()
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &lhs)?;

        let a = &inners.sigma2;
        let b = Array::ones(rt.ncols()) - rt.mapv(|v| v * v).sum_axis(Axis(0))
            + u.mapv(|v: F| v * v).sum_axis(Axis(0));
        let mse = einsum("i,j->ji", &[a, &b])
            .unwrap()
            .into_shape((x.shape()[0], 1))
            .unwrap();

        // Mean Squared Error might be slightly negative depending on
        // machine precision: set to zero in that case
        Ok(mse.mapv(|v| if v < F::zero() { F::zero() } else { F::cast(v) }))
    }

    fn _compute_correlation(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        let n_obs = x.nrows();
        let n_features = x.ncols();

        let xnorm = (x - &self.xtrain.mean) / &self.xtrain.std;
        let nt = self.xtrain.data.nrows();
        // Get pairwise componentwise L1-distances to the input training set
        let mut dx: Array2<F> = Array2::zeros((nt * n_obs, n_features));
        for (i, xrow) in xnorm.rows().into_iter().enumerate() {
            let dxrows = &self.xtrain.data - &xrow.into_shape((1, n_features)).unwrap();
            let a = i * nt;
            let b = (i + 1) * nt;
            dx.slice_mut(s![a..b, ..]).assign(&dxrows);
        }
        // Compute the correlation function
        let r = self.kernel.apply(&self.theta, &dx, &self.w_star);
        r.into_shape((n_obs, nt)).unwrap().to_owned()
    }

    pub fn kpls_dim(&self) -> Option<usize> {
        if self.w_star.ncols() < self.xtrain.ncols() {
            Some(self.w_star.ncols())
        } else {
            None
        }
    }
}

impl<F: Float, Mean: RegressionModel<F>, Kernel: CorrelationModel<F>> GpParams<F, Mean, Kernel> {
    pub fn fit(
        self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
        y: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Result<GaussianProcess<F, Mean, Kernel>> {
        self.validate(x.ncols())?;

        let xtrain = NormalizedMatrix::new(x);
        let ytrain = NormalizedMatrix::new(y);

        let mut w_star = Array2::eye(x.ncols());
        if let Some(n_components) = self.kpls_dim() {
            let ds = Dataset::new(x.to_owned(), y.to_owned());
            let pls = PlsRegression::<F>::params(*n_components).fit(&ds)?;
            let (x_rotations, _) = pls.rotations();
            w_star = x_rotations.to_owned();
        };
        let x_distances = DistanceMatrix::new(&xtrain.data);
        let sums = x_distances
            .d
            .mapv(|v| num_traits::float::Float::abs(v))
            .sum_axis(Axis(1));
        if *sums.min().unwrap() == F::zero() {
            println!(
                "Warning: multiple x input features have the same value (at least same row twice)."
            );
        }

        let theta0 = Array1::from_elem(w_star.ncols(), self.initial_theta());
        let fx = self.mean().apply(x);
        let y_t = ytrain.clone();
        let base: f64 = 10.;
        let objfn = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            let theta =
                Array1::from_shape_vec((x.len(),), x.iter().map(|v| base.powf(*v)).collect())
                    .unwrap();
            let theta = theta.mapv(F::cast);
            let rxx = self.kernel().apply(&theta, &x_distances.d, &w_star);
            match reduced_likelihood(&fx, rxx, &x_distances, &y_t, self.nugget()) {
                Ok(r) => unsafe { -(*(&r.0 as *const F as *const f64)) },
                Err(_) => {
                    // println!("GP lkh ERROR: {:?}", err);
                    f64::INFINITY
                }
            }
        };

        // Multistart: user theta0 + 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10.
        let mut theta0s = Array2::zeros((8, theta0.len()));
        theta0s.row_mut(0).assign(&theta0);
        let mut xlimits: Array2<F> = Array2::zeros((theta0.len(), 2));
        for mut row in xlimits.rows_mut() {
            row.assign(&arr1(&[F::cast(1e-6), F::cast(20.)]));
        }
        // Use a seed here for reproducibility. Do we need to make it truly random
        // Probably no, as it is just to get init values spread over
        // [1e-6, 20] for multistart thanks to LHS method.
        let seeds = LHS::new(&xlimits)
            .with_rng(Isaac64Rng::seed_from_u64(42))
            .sample(7);
        Zip::from(theta0s.slice_mut(s![1.., ..]).rows_mut())
            .and(seeds.rows())
            .par_for_each(|mut theta, row| theta.assign(&row));
        // println!("theta0s = {:?}", theta0s);
        let opt_thetas =
            theta0s.map_axis(Axis(1), |theta| optimize_theta(objfn, &theta.to_owned()));

        let opt_index = opt_thetas.map(|(_, opt_f)| opt_f).argmin().unwrap();
        let opt_theta = &(opt_thetas[opt_index]).0.mapv(F::cast);

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

pub fn optimize_theta<ObjF, F>(objfn: ObjF, theta0: &Array1<F>) -> (Array1<f64>, f64)
where
    ObjF: Fn(&[f64], Option<&mut [f64]>, &mut ()) -> f64,
    F: Float,
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
    let mut theta_vec = theta0
        .mapv(F::log10)
        .map(|v| unsafe { *(v as *const F as *const f64) })
        .into_raw_vec();
    optimizer.set_initial_step1(0.5).unwrap();
    optimizer.set_maxeval(10 * theta0.len() as u32).unwrap();
    optimizer.set_ftol_rel(1e-4).unwrap();

    match optimizer.optimize(&mut theta_vec) {
        Ok((_, fmin)) => {
            let thetas_opt = arr1(&theta_vec).mapv(|v| base.powf(v));
            let fval = if f64::is_nan(fmin) {
                f64::INFINITY
            } else {
                fmin
            };
            (thetas_opt, fval)
        }
        Err(_e) => {
            // println!("ERROR OPTIM in GP {:?}", e);
            (arr1(&theta_vec).mapv(|v| base.powf(v)), f64::INFINITY)
        }
    }
}

pub fn reduced_likelihood<F: Float>(
    fx: &ArrayBase<impl Data<Elem = F>, Ix2>,
    rxx: ArrayBase<impl Data<Elem = F>, Ix2>,
    x_distances: &DistanceMatrix<F>,
    ytrain: &NormalizedMatrix<F>,
    nugget: F,
) -> Result<(F, GpInnerParams<F>)> {
    // Set up R
    let mut r_mx: Array2<F> = Array2::<F>::eye(x_distances.n_obs).mapv(|v| (v + v * nugget));
    for (i, ij) in x_distances.d_indices.outer_iter().enumerate() {
        r_mx[[ij[0], ij[1]]] = rxx[[i, 0]];
        r_mx[[ij[1], ij[0]]] = rxx[[i, 0]];
    }

    let fxl = fx.to_owned();

    // R cholesky decomposition
    let r_chol = r_mx.cholesky(UPLO::Lower)?;

    // Solve generalized least squared problem
    let ft = r_chol.solve_triangular(UPLO::Lower, Diag::NonUnit, &fxl)?;
    let (ft_qr_q, ft_qr_r) = ft.qr().unwrap();

    // Check whether we have an ill-conditionned problem
    let (_, sv_qr_r, _) = ft_qr_r.svd(false, false).unwrap();
    let cond_ft = sv_qr_r[sv_qr_r.len() - 1] / sv_qr_r[0];
    if F::cast(cond_ft) < F::cast(1e-10) {
        let (_, sv_f, _) = &fxl.svd(false, false).unwrap();
        let cond_fx = sv_f[0] / sv_f[sv_f.len() - 1];
        if F::cast(cond_fx) > F::cast(1e15) {
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
    let yt = r_chol.solve_triangular(UPLO::Lower, Diag::NonUnit, &ytrain.data.to_owned())?;
    let beta = ft_qr_r.solve_triangular_into(UPLO::Upper, Diag::NonUnit, ft_qr_q.t().dot(&yt))?;

    let rho = yt - ft.dot(&beta);
    let rho_sqr: Array1<F> = rho.mapv(|v| v * v).sum_axis(Axis(0));
    let gamma = r_chol
        .t()
        .solve_triangular_into(UPLO::Upper, Diag::NonUnit, rho)?;

    // The determinant of R is equal to the squared product of
    // the diagonal elements of its Cholesky decomposition r_chol
    let n_obs: F = F::cast(x_distances.n_obs);
    let logdet = r_chol.to_owned().diag().mapv(|v: F| v.log10()).sum() * F::cast(2.) / n_obs;
    // Reduced likelihood
    let sigma2 = rho_sqr / n_obs;
    let reduced_likelihood = -n_obs * (sigma2.sum().log10() + logdet);
    Ok((
        reduced_likelihood,
        GpInnerParams {
            sigma2: sigma2 * &ytrain.std.mapv(|v| v * v),
            beta: beta,
            gamma,
            r_chol: r_chol,
            ft: ft,
            ft_qr_r: ft_qr_r,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{correlation_models::*, mean_models::*};
    use approx::assert_abs_diff_eq;
    use argmin_testfunctions::rosenbrock;
    use doe::{SamplingMethod, LHS};
    use ndarray::{arr2, array, Array, Zip};
    use ndarray_linalg::Norm;
    use ndarray_npy::{read_npy, write_npy};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_stats::DeviationExt;
    use paste::paste;
    use rand_isaac::Isaac64Rng;

    macro_rules! test_gp {
        ($regr:ident, $corr:ident, $expected:expr) => {
            paste! {

                #[test]
                fn [<test_gp_ $regr:snake _ $corr:snake >]() {
                    let xt = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
                    let xplot = Array::linspace(0., 4., 100).insert_axis(Axis(1));
                    let yt = array![[0.0], [1.0], [1.5], [0.9], [1.0]];
                    let gp = GaussianProcess::<f64, [<$regr Mean>], [<$corr Kernel>] >::params(
                        [<$regr Mean>]::default(),
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
                    assert_abs_diff_eq!(expected_y, yvals, epsilon = 0.5);

                    let gpr_vals = gp.predict_values(&xplot).unwrap();

                    let yvars = gp
                        .predict_variances(&arr2(&[[1.0], [3.5]]))
                        .expect("prediction error");
                    let expected_vars = arr2(&[[0.], [0.1]]);
                    assert_abs_diff_eq!(expected_vars, yvars, epsilon = 0.5);

                    let gpr_vars = gp.predict_variances(&xplot).unwrap();

                    let xplot_file = stringify!([<gp_x_ $regr:snake _ $corr:snake >]);
                    write_npy(format!("{}.npy", xplot_file), &xplot).expect("x saved");

                    let gp_vals_file = stringify!([<gp_vals_ $regr:snake _ $corr:snake >]);
                    write_npy(format!("{}.npy", gp_vals_file), &gpr_vals).expect("gp vals saved");

                    let gp_vars_file = stringify!([<gp_vars_ $regr:snake _ $corr:snake >]);
                    write_npy(format!("{}.npy", gp_vars_file), &gpr_vars).expect("gp vars saved");
                }
            }
        };
    }

    test_gp!(Constant, SquaredExponential, 1.66);
    test_gp!(Constant, AbsoluteExponential, 22.35);
    test_gp!(Constant, Matern32, 21.68);
    test_gp!(Constant, Matern52, 21.68);

    test_gp!(Linear, SquaredExponential, 1.55);
    test_gp!(Linear, AbsoluteExponential, 21.68);
    test_gp!(Linear, Matern32, 21.68);
    test_gp!(Linear, Matern52, 21.68);

    test_gp!(Quadratic, SquaredExponential, 21.14);
    test_gp!(Quadratic, AbsoluteExponential, 24.98);
    test_gp!(Quadratic, Matern32, 22.35);
    test_gp!(Quadratic, Matern52, 21.68);

    #[test]
    fn test_kpls() {
        let dims = vec![5, 10, 20]; //, 60];
        let nts = vec![100, 300, 400]; //, 800];

        for i in 0..2 {
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
                    write_npy(&xfilename, &xt).expect("cannot save xt");
                    xt
                }
            };

            let yt = match read_npy(&yfilename) {
                Ok(yt) => yt,
                Err(_) => {
                    let mut yv: Array1<f64> = Array1::zeros(xt.nrows());
                    Zip::from(&mut yv).and(xt.rows()).par_for_each(|y, x| {
                        *y = griewank(&x.to_owned());
                    });
                    let yt = yv.into_shape((xt.nrows(), 1)).unwrap();
                    write_npy(&yfilename, &yt).expect("cannot save yt");
                    yt
                }
            };

            let gp = GaussianProcess::<f64, ConstantMean, SquaredExponentialKernel>::params(
                ConstantMean::default(),
                SquaredExponentialKernel::default(),
            )
            .set_kpls_dim(Some(3))
            .fit(&xt, &yt)
            .expect("GP fit error");

            let xtest = Array2::zeros((1, dim));
            let ytest = gp.predict_values(&xtest).expect("prediction error");
            let ytrue = griewank(&xtest.row(0).to_owned());
            assert_abs_diff_eq!(Array::from_elem((1, 1), ytrue), ytest, epsilon = 1.1);
        }
    }

    fn tensor_product_exp(x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        x.mapv(|v| v.exp())
            .map_axis(Axis(1), |row| row.product())
            .insert_axis(Axis(1))
    }

    #[test]
    fn test_kpls_tp_exp() {
        let dim = 3;
        let nt = 300;
        let lim = array![[-1., 1.]];
        let xlimits = lim.broadcast((dim, 2)).unwrap();
        let rng = Isaac64Rng::seed_from_u64(42);
        let xt = LHS::new(&xlimits).with_rng(rng).sample(nt);
        let yt = tensor_product_exp(&xt);

        let gp = GaussianProcess::<f64, ConstantMean, SquaredExponentialKernel>::params(
            ConstantMean::default(),
            SquaredExponentialKernel::default(),
        )
        .set_kpls_dim(Some(1))
        .fit(&xt, &yt)
        .expect("GP training");

        let xv = LHS::new(&xlimits).sample(100);
        let yv = tensor_product_exp(&xv);

        let ytest = gp.predict_values(&xv).unwrap();
        let err = ytest.l2_dist(&yv).unwrap() / yv.norm_l2();
        assert_abs_diff_eq!(err, 0., epsilon = 2e-2);
    }

    fn rosenb(x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
        Zip::from(y.rows_mut())
            .and(x.rows())
            .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec(), 1., 100.)]));
        y
    }

    #[test]
    fn test_kpls_rosenb() {
        let dim = 20;
        let nt = 200;
        let lim = array![[-1., 1.]];
        let xlimits = lim.broadcast((dim, 2)).unwrap();
        let rng = Isaac64Rng::seed_from_u64(42);
        let xt = LHS::new(&xlimits).with_rng(rng).sample(nt);
        let yt = rosenb(&xt);

        let gp = GaussianProcess::<f64, ConstantMean, Matern32Kernel>::params(
            ConstantMean::default(),
            Matern52Kernel::default(),
        )
        .set_kpls_dim(Some(1))
        .fit(&xt, &yt)
        .expect("GP training");

        let xv = LHS::new(&xlimits).sample(500);
        let yv = rosenb(&xv);

        let ytest = gp.predict_values(&xv).unwrap();
        let err = ytest.l2_dist(&yv).unwrap() / yv.norm_l2();
        assert_abs_diff_eq!(err, 0., epsilon = 2e-1);
    }
}
