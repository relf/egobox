use crate::correlation_models::*;
use crate::errors::{GpError, Result};
use crate::mean_models::*;
use crate::parameters::{GpParams, GpValidParams};
use crate::utils::{pairwise_differences, DistanceMatrix, NormalizedMatrix};
use egobox_doe::{Lhs, SamplingMethod};
use linfa::dataset::{WithLapack, WithoutLapack};
use linfa::prelude::{Dataset, DatasetBase, Fit, Float, PredictInplace};
#[cfg(not(feature = "blas"))]
use linfa_linalg::{cholesky::*, eigh::*, qr::*, svd::*, triangular::*};
use linfa_pls::PlsRegression;
#[cfg(feature = "blas")]
use log::warn;
use ndarray::{arr1, s, Array, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip};
use ndarray_einsum_beta::*;
#[cfg(feature = "blas")]
use ndarray_linalg::{cholesky::*, eigh::*, qr::*, svd::*, triangular::*};
use ndarray_rand::rand::SeedableRng;
use ndarray_stats::QuantileExt;

use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};
use std::fmt;

use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

// const LOG10_20: f64 = 1.301_029_995_663_981_3; //f64::log10(20.);
const N_START: usize = 10; // number of optimization restart (aka multistart)

/// Internal parameters computed Gp during training
/// used later on in prediction computations
#[derive(Default, Debug)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(bound(deserialize = "F: Deserialize<'de>"))
)]
pub(crate) struct GpInnerParams<F: Float> {
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

/// A GP regression is an interpolation method where the
/// interpolated values are modeled by a Gaussian process with a mean  
/// governed by a prior covariance kernel, which depends on some
/// parameters to be determined.
///
/// The interpolated output is modeled as stochastic process as follows:
///
/// `Y(x) = mu(x) + Z(x)`
///
/// where:
/// * `mu(x)` is the trend i.e. the mean of the gaussian process
/// * `Z(x)` the realization of stochastic gaussian process ~ `Normal(0, sigma^2)`
///
/// which in turn is written as:
///
/// `Y(x) = betas.regr(x) + sigma^2*corr(x, x')`
///
/// where:
/// * `betas` is a vector of linear regression parameters to be determined
/// * `regr(x)` a vector of polynomial basis functions
/// * `sigma^2` is the process variance
/// * `corr(x, x')` is a correlation function which depends on `distance(x, x')`
/// and a set of unknown parameters `thetas` to be determined.
///
/// # Implementation
///
/// * Based on [ndarray](https://github.com/rust-ndarray/ndarray)
/// and [linfa](https://github.com/rust-ml/linfa) and strive to follow [linfa guidelines](https://github.com/rust-ml/linfa/blob/master/CONTRIBUTE.md)
/// * GP mean model can be constant, linear or quadratic
/// * GP correlation model can be build the following kernels: squared exponential, absolute exponential, matern 3/2, matern 5/2    
/// cf. [SMT Kriging](https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/krg.html)
/// * For high dimensional problems, the classic GP algorithm does not perform well as
/// it depends on the inversion of a correlation (n, n) matrix which is an O(n3) operation.
/// To work around this problem the library implements dimension reduction using
/// Partial Least Squares method upon Kriging method also known as KPLS algorithm (see Reference)
/// * GP models can be saved and loaded using [serde](https://serde.rs/).
/// See `serializable` feature section below.
///
/// # Features
///
/// ## serializable
///
/// The `serializable` feature enables the serialization of GP models using the [`serde crate`](https://serde.rs/).
///
/// ## blas
///
/// The `blas` feature enables the use of BLAS/LAPACK linear algebra backend available with [`ndarray-linalg`](https://github.com/rust-ndarray/ndarray-linalg).
///
/// # Example
///
/// ```no_run
/// use egobox_gp::{correlation_models::*, mean_models::*, GaussianProcess};
/// use linfa::prelude::*;
/// use ndarray::{arr2, concatenate, Array, Array2, Axis};
///
/// // one-dimensional test function to approximate
/// fn xsinx(x: &Array2<f64>) -> Array2<f64> {
///     (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
/// }
///
/// // training data
/// let xt = arr2(&[[0.0], [5.0], [10.0], [15.0], [18.0], [20.0], [25.0]]);
/// let yt = xsinx(&xt);
///
/// // GP with constant mean model and squared exponential correlation model
/// // i.e. Oridinary Kriging model
/// let kriging = GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
///                 ConstantMean::default(),
///                 SquaredExponentialCorr::default())
///                 .fit(&Dataset::new(xt, yt))
///                 .expect("Kriging trained");
///
/// // Use trained model for making predictions
/// let xtest = Array::linspace(0., 25., 26).insert_axis(Axis(1));
/// let ytest = xsinx(&xtest);
///
/// let ypred = kriging.predict_values(&xtest).expect("Kriging prediction");
/// let yvariances = kriging.predict_variances(&xtest).expect("Kriging prediction");  
///```
///
/// # Reference:
///
/// Bouhlel, Mohamed Amine, et al. [Improving kriging surrogates of high-dimensional design
/// models by Partial Least Squares dimension reduction](https://hal.archives-ouvertes.fr/hal-01232938/document)
/// Structural and Multidisciplinary Optimization 53.5 (2016): 935-952.
///
#[derive(Debug)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))
)]
pub struct GaussianProcess<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> {
    /// Parameter of the autocorrelation model
    theta: Array1<F>,
    /// Regression model
    #[cfg_attr(
        feature = "serializable",
        serde(bound(serialize = "Mean: Serialize", deserialize = "Mean: Deserialize<'de>"))
    )]
    mean: Mean,
    /// Correlation kernel
    #[cfg_attr(
        feature = "serializable",
        serde(bound(serialize = "Corr: Serialize", deserialize = "Corr: Deserialize<'de>"))
    )]
    corr: Corr,
    /// Gaussian process internal fitted params
    inner_params: GpInnerParams<F>,
    /// Weights in case of KPLS dimension reduction coming from PLS regression (orig_dim, kpls_dim)
    w_star: Array2<F>,
    /// Training inputs
    xtrain: NormalizedMatrix<F>,
    /// Training outputs
    ytrain: NormalizedMatrix<F>,
}

enum GpSamplingMethod {
    Cholesky,
    EigenValues,
}

/// Kriging as GP special case when using constant mean and squared exponential correlation
pub type Kriging<F> = GpParams<F, ConstantMean, SquaredExponentialCorr>;

impl<F: Float> Kriging<F> {
    pub fn params() -> GpParams<F, ConstantMean, SquaredExponentialCorr> {
        GpParams::new(ConstantMean(), SquaredExponentialCorr())
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> Clone
    for GaussianProcess<F, Mean, Corr>
{
    fn clone(&self) -> Self {
        Self {
            theta: self.theta.to_owned(),
            mean: self.mean,
            corr: self.corr,
            inner_params: self.inner_params.clone(),
            w_star: self.w_star.to_owned(),
            xtrain: self.xtrain.clone(),
            ytrain: self.xtrain.clone(),
        }
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> fmt::Display
    for GaussianProcess<F, Mean, Corr>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GP({}, {})", self.mean, self.corr)
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> GaussianProcess<F, Mean, Corr> {
    /// Gp parameters contructor
    pub fn params<NewMean: RegressionModel<F>, NewCorr: CorrelationModel<F>>(
        mean: NewMean,
        corr: NewCorr,
    ) -> GpParams<F, NewMean, NewCorr> {
        GpParams::new(mean, corr)
    }

    /// Predict output values at n given `x` points of nx components specified as a (n, nx) matrix.
    /// Returns n scalar output values as (n, 1) column vector.
    pub fn predict_values(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Result<Array2<F>> {
        let xnorm = (x - &self.xtrain.mean) / &self.xtrain.std;
        // Compute the mean term at x
        let f = self.mean.value(&xnorm);
        // Compute the correlation term at x
        let corr = self._compute_correlation(&xnorm);
        // Scaled predictor
        let y_ = &f.dot(&self.inner_params.beta) + &corr.dot(&self.inner_params.gamma);
        // Predictor
        Ok(&y_ * &self.ytrain.std + &self.ytrain.mean)
    }

    /// Predict variance values at n given `x` points of nx components specified as a (n, nx) matrix.
    /// Returns n variance values as (n, 1) column vector.
    pub fn predict_variances(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Result<Array2<F>> {
        let (rt, u, _) = self._compute_rt_u(x);

        let a = &self.inner_params.sigma2;
        let b = Array::ones(rt.ncols()) - rt.mapv(|v| v * v).sum_axis(Axis(0))
            + u.mapv(|v: F| v * v).sum_axis(Axis(0));
        let mse = einsum("i,j->ji", &[a, &b])
            .unwrap()
            .into_shape((x.nrows(), 1))
            .unwrap();

        // Mean Squared Error might be slightly negative depending on
        // machine precision: set to zero in that case
        Ok(mse.mapv(|v| if v < F::zero() { F::zero() } else { F::cast(v) }))
    }

    /// Compute covariance matrix given x points specified as a (n, nx) matrix
    fn _compute_covariance(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        let (rt, u, xnorm) = self._compute_rt_u(x);

        let cross_dx = pairwise_differences(&xnorm, &xnorm);
        let k = self.corr.value(&cross_dx, &self.theta, &self.w_star);
        let k = k.into_shape((xnorm.nrows(), xnorm.nrows())).unwrap();

        let cov_matrix =
            &self.inner_params.sigma2.to_owned() * (k - rt.t().to_owned().dot(&rt) + u.t().dot(&u));
        cov_matrix
    }

    /// Compute `rt` and `u` matrices and return normalized x as well
    /// This method factorizes computations done to get variances and covariance matrix
    fn _compute_rt_u(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> (Array2<F>, Array2<F>, Array2<F>) {
        let xnorm = (x - &self.xtrain.mean) / &self.xtrain.std;
        let corr = self._compute_correlation(&xnorm);
        let inners = &self.inner_params;

        let corr_t = corr.t().to_owned();
        #[cfg(feature = "blas")]
        let rt = inners
            .r_chol
            .to_owned()
            .with_lapack()
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &corr_t.with_lapack())
            .unwrap()
            .without_lapack();
        #[cfg(not(feature = "blas"))]
        let rt = inners
            .r_chol
            .solve_triangular(&corr_t, UPLO::Lower)
            .unwrap();

        let rhs = inners.ft.t().dot(&rt) - self.mean.value(&xnorm).t();
        #[cfg(feature = "blas")]
        let u = inners
            .ft_qr_r
            .to_owned()
            .t()
            .with_lapack()
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &rhs.with_lapack())
            .unwrap()
            .without_lapack();
        #[cfg(not(feature = "blas"))]
        let u = inners
            .ft_qr_r
            .t()
            .solve_triangular(&rhs, UPLO::Lower)
            .unwrap();
        (rt, u, xnorm)
    }

    /// Compute correlation matrix given x points specified as a (n, nx) matrix
    fn _compute_correlation(&self, xnorm: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        // Get pairwise componentwise L1-distances to the input training set
        let dx = pairwise_differences(xnorm, &self.xtrain.data);
        // Compute the correlation function
        let r = self.corr.value(&dx, &self.theta, &self.w_star);
        let n_obs = xnorm.nrows();
        let nt = self.xtrain.data.nrows();
        r.into_shape((n_obs, nt)).unwrap().to_owned()
    }

    /// Sample the gaussian process for `n_traj` trajectories using cholesky decomposition
    pub fn sample_chol(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>, n_traj: usize) -> Array2<F> {
        self._sample(x, n_traj, GpSamplingMethod::Cholesky)
    }

    /// Sample the gaussian process for `n_traj` trajectories using eigenvalues decomposition
    pub fn sample_eig(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>, n_traj: usize) -> Array2<F> {
        self._sample(x, n_traj, GpSamplingMethod::EigenValues)
    }

    /// Sample the gaussian process for `n_traj` trajectories using eigenvalues decomposition (alias of `sample_eig`)
    pub fn sample(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>, n_traj: usize) -> Array2<F> {
        self.sample_eig(x, n_traj)
    }

    /// Sample the gaussian process for `n_traj` trajectories using either
    /// cholesky or eigenvalues decomposition to compute the decomposition of the conditioned covariance matrix.
    /// The later one is recommended as cholesky decomposition suffer from occurence of ill-conditioned matrices
    /// when the number of x locations increase.
    fn _sample(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
        n_traj: usize,
        method: GpSamplingMethod,
    ) -> Array2<F> {
        let n_eval = x.nrows();
        let cov = self._compute_covariance(x);
        let c = match method {
            GpSamplingMethod::Cholesky => {
                #[cfg(not(feature = "blas"))]
                let c = cov.with_lapack().cholesky().unwrap();
                #[cfg(feature = "blas")]
                let c = cov.with_lapack().cholesky(UPLO::Lower).unwrap();
                c
            }
            GpSamplingMethod::EigenValues => {
                #[cfg(feature = "blas")]
                let (v, w) = cov.with_lapack().eigh(UPLO::Lower).unwrap();
                #[cfg(not(feature = "blas"))]
                let (v, w) = cov.with_lapack().eigh_into().unwrap();
                let v = v.mapv(F::cast);
                let v = v.mapv(|x| {
                    // We lower bound the float value at 1e-9
                    if x < F::cast(1e-9) {
                        return F::zero();
                    }
                    x.sqrt()
                });
                let d = Array2::from_diag(&v).with_lapack();
                #[cfg(feature = "blas")]
                let c = w.dot(&d);
                #[cfg(not(feature = "blas"))]
                let c = w.dot(&d);
                c
            }
        }
        .without_lapack();
        let mean = self.predict_values(x).unwrap();
        let normal = Normal::new(0., 1.).unwrap();
        let ary = Array::random((n_eval, n_traj), normal).mapv(|v| F::cast(v));
        mean + c.dot(&ary)
    }

    /// Retrieve number of PLS components 1 <= n <= x dimension
    pub fn kpls_dim(&self) -> Option<usize> {
        if self.w_star.ncols() < self.xtrain.ncols() {
            Some(self.w_star.ncols())
        } else {
            None
        }
    }

    /// Retrieve input dimension before kpls dimension reduction if any
    pub fn input_dim(&self) -> usize {
        self.xtrain.ncols()
    }

    /// Retrieve output dimension
    pub fn output_dim(&self) -> usize {
        self.ytrain.ncols()
    }

    /// Predict derivatives of the output prediction
    /// wrt the kxth component at a set of n points `x` specified as a (n, nx) matrix where x has nx components.
    pub fn predict_kth_derivatives(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
        kx: usize,
    ) -> Array1<F> {
        let xnorm = (x - &self.xtrain.mean) / &self.xtrain.std;
        let corr = self._compute_correlation(&xnorm);

        let beta = &self.inner_params.beta;
        let gamma = &self.inner_params.gamma;

        let df_dx_kx = if self.inner_params.beta.nrows() <= 1 + self.xtrain.data.ncols() {
            // for constant or linear: df/dx = cst ([0] or [1]) for all x, so takes use x[0] to get the constant
            let df = self.mean.jacobian(&x.row(0));
            let df_dx = df.t().row(kx).dot(beta);
            df_dx.broadcast((x.nrows(), 1)).unwrap().to_owned()
        } else {
            // for quadratic df/dx really depends on x
            let mut dfdx = Array2::zeros((x.nrows(), 1));
            Zip::from(dfdx.rows_mut())
                .and(xnorm.rows())
                .for_each(|mut dfxi, xi| {
                    let df = self.mean.jacobian(&xi);
                    let df_dx = (df.t().row(kx)).dot(beta);
                    dfxi.assign(&df_dx);
                });
            dfdx
        };

        let nr = x.nrows();
        let nc = self.xtrain.data.nrows();
        let d_dx_1 = &xnorm
            .column(kx)
            .to_owned()
            .into_shape((nr, 1))
            .unwrap()
            .broadcast((nr, nc))
            .unwrap()
            .to_owned();

        let d_dx_2 = self
            .xtrain
            .data
            .column(kx)
            .to_owned()
            .as_standard_layout()
            .into_shape((1, nc))
            .unwrap()
            .to_owned();

        let d_dx = d_dx_1 - d_dx_2;

        // Get pairwise componentwise L1-distances to the input training set
        let theta = &self.theta.to_owned();
        let d_dx_corr = d_dx * corr;

        // (df(xnew)/dx).beta + (dr(xnew)/dx).R^-1(ytrain - f.beta)
        // gamma = R^-1(ytrain - f.beta)
        // Warning: squared exponential only
        let res = (df_dx_kx - d_dx_corr.dot(gamma).map(|v| F::cast(2.) * theta[kx] * *v))
            * self.ytrain.std[0]
            / self.xtrain.std[kx];
        res.column(0).to_owned()
    }

    /// Predict derivatives at a set of point `x` specified as a (n, nx) matrix where x has nx components.
    /// Returns a (n, nx) matrix containing output derivatives at x wrt each nx components
    pub fn predict_derivatives(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        let mut drv = Array2::<F>::zeros((x.nrows(), self.xtrain.data.ncols()));
        Zip::from(drv.rows_mut())
            .and(x.rows())
            .for_each(|mut row, xi| {
                let pred = self.predict_jacobian(&xi);
                row.assign(&pred.column(0));
            });
        drv
    }

    pub fn predict_jacobian(&self, x: &ArrayBase<impl Data<Elem = F>, Ix1>) -> Array2<F> {
        let xx = x.to_owned().insert_axis(Axis(0));
        let mut jac = Array2::zeros((xx.ncols(), 1));

        let xnorm = (xx - &self.xtrain.mean) / &self.xtrain.std;

        let beta = &self.inner_params.beta;
        let gamma = &self.inner_params.gamma;

        let df = self.mean.jacobian(&xnorm.row(0));
        let df_dx = df.t().dot(beta);

        let dr = self
            .corr
            .jacobian(&xnorm.row(0), &self.xtrain.data, &self.theta, &self.w_star);

        let dr_dx = df_dx + dr.t().dot(gamma);
        Zip::from(jac.rows_mut())
            .and(dr_dx.rows())
            .and(&self.xtrain.std)
            .for_each(|mut jc, dr_i, std_i| {
                let jc_i = dr_i.map(|v| *v * self.ytrain.std[0] / *std_i);
                jc.assign(&jc_i)
            });

        jac
    }

    /// Predict variance derivatives at a point `x` specified as a (nx,) vector where x has nx components.
    /// Returns a (nx,) vector containing variance derivatives at `x` wrt each nx components
    #[cfg(not(feature = "blas"))]
    pub fn predict_variance_derivatives_single(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
    ) -> Array1<F> {
        let x = &(x.to_owned().insert_axis(Axis(0)));
        let xnorm = (x - &self.xtrain.mean) / &self.xtrain.std;
        let dx = pairwise_differences(&xnorm, &self.xtrain.data);

        let sigma2 = &self.inner_params.sigma2;
        let r_chol = &self.inner_params.r_chol;

        let r = self.corr.value(&dx, &self.theta, &self.w_star);
        let dr = self
            .corr
            .jacobian(&xnorm.row(0), &self.xtrain.data, &self.theta, &self.w_star)
            / &self.xtrain.std.to_owned().insert_axis(Axis(0));

        // rho1 = Rc^-1 . r(x, X)
        let rho1 = r_chol.solve_triangular(&r, UPLO::Lower).unwrap();
        // inv_kr = Rc^t^-1 . Rc^-1 . r(x, X) = R^-1 . r(x, X)
        let inv_kr = r_chol.t().solve_triangular(&rho1, UPLO::Upper).unwrap();

        // p1 = ((dr(x, X)/dx)^t . R^-1 . r(x, X))^t = ((R^-1 . r(x, X))^t . dr(x, X)/dx) = r(x, X)^t . R^-1 . dr(x, X)/dx = p2
        // let p1 = dr.t().dot(&inv_kr).t().to_owned();

        // p2 = ((R^-1 . r(x, X))^t . dr(x, X)/dx)^t = dr(x, X)/dx)^t . R^-1 . r(x, X) = p1
        let p2 = inv_kr.t().dot(&dr);

        let f_x = self.mean.value(&xnorm).t().to_owned();
        let f_mean = self.mean.value(&self.xtrain.data);

        // rho2 = Rc^-1 . F(X)
        let rho2 = r_chol.solve_triangular(&f_mean, UPLO::Lower).unwrap();
        // inv_kf = Rc^-1^t . Rc^-1 . F(X) = R^-1 . F(X)
        let inv_kf = r_chol.t().solve_triangular(&rho2, UPLO::Upper).unwrap();

        // A = f(x)^t - r(x, X)^t . R^-1 . F(X)   -> (1 x m)
        let a_mat = f_x.t().to_owned() - r.t().dot(&inv_kf);

        // B = F(X)^t . R^-1 . F(X)
        let b_mat = f_mean.t().dot(&inv_kf);
        // rho3 = Bc
        let rho3 = b_mat.cholesky().unwrap();
        // inv_bat = Bc^-1 . A^t
        let inv_bat = rho3.solve_triangular(&a_mat.t(), UPLO::Lower).unwrap();
        // D = Bc^t-1 . Bc^-1 . A^t = B^-1 . A^t
        let d_mat = rho3.t().solve_triangular(&inv_bat, UPLO::Upper).unwrap();

        let df = self.mean.jacobian(&xnorm.row(0));

        // dA/dx = df(x)/dx^t - dr(x, X)/dx^t . R^-1 . F
        let d_a = df.t().to_owned() - dr.t().dot(&inv_kf);

        // p3 = (dA/dx . B^-1 . A^t)^t = A . B^-1 . dA/dx^t
        // let p3 = d_a.dot(&d_mat).t().to_owned();

        // p4 = (B^-1 . A)^t . dA/dx^t = A^t . B^-1 . dA/dx^t = p3
        let p4 = d_mat.t().dot(&d_a.t());

        let two = F::cast(2.);
        let prime_t = (-p2 + p4).mapv(|v| two * v).t().to_owned();

        let x_std = &self.xtrain.std;
        let dvar = sigma2 * prime_t / x_std;
        dvar.row(0).to_owned()
    }

    /// See non blas version
    #[cfg(feature = "blas")]
    pub fn predict_variance_derivatives_single(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
    ) -> Array1<F> {
        let x = &(x.to_owned().insert_axis(Axis(0)));
        let xnorm = (x - &self.xtrain.mean) / &self.xtrain.std;

        let dx = pairwise_differences(&xnorm, &self.xtrain.data);

        let sigma2 = &self.inner_params.sigma2;
        let r_chol = &self.inner_params.r_chol.to_owned().with_lapack();

        let r = self
            .corr
            .value(&dx, &self.theta, &self.w_star)
            .with_lapack();
        let dr = self
            .corr
            .jacobian(&xnorm.row(0), &self.xtrain.data, &self.theta, &self.w_star)
            .with_lapack();

        let rho1 = r_chol
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &r)
            .unwrap();
        let inv_kr = r_chol
            .t()
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &rho1)
            .unwrap();

        // let p1 = dr.t().dot(&inv_kr).t().to_owned();

        let p2 = inv_kr.t().dot(&dr);

        let f_x = self.mean.value(x).t().to_owned();
        let f_mean = self.mean.value(&self.xtrain.data).with_lapack();

        let rho2 = r_chol
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &f_mean)
            .unwrap();
        let inv_kf = r_chol
            .t()
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &rho2)
            .unwrap();

        let a_mat = f_x.t().to_owned().with_lapack() - r.t().dot(&inv_kf);

        let b_mat = f_mean.t().dot(&inv_kf);

        let d_mat = match b_mat.cholesky(UPLO::Lower) {
            Ok(rho3) => {
                let inv_bat = rho3
                    .solve_triangular(UPLO::Upper, Diag::NonUnit, &a_mat.t().to_owned())
                    .unwrap();
                rho3.t()
                    .solve_triangular(UPLO::Upper, Diag::NonUnit, &inv_bat)
                    .unwrap()
            }
            Err(_) => {
                warn!("Cholesky decomposition error during variance dervivatives computation");
                Array2::zeros((b_mat.nrows(), b_mat.ncols()))
            }
        };

        let df = self.mean.jacobian(&xnorm.row(0)).with_lapack();

        let d_a = df.t().to_owned() - dr.t().dot(&inv_kf);
        // let p3 = d_a.dot(&d_mat).t();
        let p4 = d_mat.t().dot(&d_a.t());

        let two = F::cast(2.);
        let prime_t = (-p2 + p4).without_lapack().mapv(|v| two * v).t().to_owned();

        let x_std = &self.xtrain.std;
        let dvar = sigma2 * prime_t / x_std;
        dvar.row(0).to_owned()
    }

    /// Predict variance derivatives at a set of points `x` specified as a (n, nx) matrix where x has nx components.
    /// Returns a (n, nx) matrix containing variance derivatives at `x` wrt each nx components
    pub fn predict_variance_derivatives(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let mut derivs = Array::zeros((x.nrows(), x.ncols()));
        Zip::from(derivs.rows_mut())
            .and(x.rows())
            .for_each(|mut der, x| der.assign(&self.predict_variance_derivatives_single(&x)));
        derivs
    }
}

impl<F, D, Mean, Corr> PredictInplace<ArrayBase<D, Ix2>, Array2<F>>
    for GaussianProcess<F, Mean, Corr>
where
    F: Float,
    D: Data<Elem = F>,
    Mean: RegressionModel<F>,
    Corr: CorrelationModel<F>,
{
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array2<F>) {
        assert_eq!(
            x.nrows(),
            y.nrows(),
            "The number of data points must match the number of output targets."
        );

        let values = self.predict_values(x).expect("GP Prediction");
        *y = values;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array2<F> {
        Array2::zeros((x.nrows(), self.output_dim()))
    }
}

/// Gausssian Process adaptator to implement `linfa::Predict` trait for variance prediction.
struct GpVariancePredictor<'a, F, Mean, Corr>(&'a GaussianProcess<F, Mean, Corr>)
where
    F: Float,
    Mean: RegressionModel<F>,
    Corr: CorrelationModel<F>;

impl<'a, F, D, Mean, Corr> PredictInplace<ArrayBase<D, Ix2>, Array2<F>>
    for GpVariancePredictor<'a, F, Mean, Corr>
where
    F: Float,
    D: Data<Elem = F>,
    Mean: RegressionModel<F>,
    Corr: CorrelationModel<F>,
{
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array2<F>) {
        assert_eq!(
            x.nrows(),
            y.nrows(),
            "The number of data points must match the number of output targets."
        );

        let values = self.0.predict_variances(x).expect("GP Prediction");
        *y = values;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array2<F> {
        Array2::zeros((x.nrows(), self.0.output_dim()))
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>, D: Data<Elem = F>>
    Fit<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>, GpError> for GpValidParams<F, Mean, Corr>
{
    type Object = GaussianProcess<F, Mean, Corr>;

    /// Fit GP parameters using maximum likelihood
    fn fit(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
    ) -> Result<Self::Object> {
        let x = dataset.records();
        let y = dataset.targets();
        if let Some(d) = self.kpls_dim() {
            if *d > x.ncols() {
                return Err(GpError::InvalidValue(format!(
                    "Dimension reduction {} should be smaller than actual \
                    training input dimensions {}",
                    d,
                    x.ncols()
                )));
            };
        }

        let xtrain = NormalizedMatrix::new(x);
        let ytrain = NormalizedMatrix::new(y);

        let mut w_star = Array2::eye(x.ncols());
        if let Some(n_components) = self.kpls_dim() {
            let ds = Dataset::new(x.to_owned(), y.to_owned());
            w_star = PlsRegression::params(*n_components).fit(&ds).map_or_else(
                |e| match e {
                    linfa_pls::PlsError::PowerMethodConstantResidualError() => {
                        Ok(Array2::zeros((x.ncols(), *n_components)))
                    }
                    err => Err(err),
                },
                |v| Ok(v.rotations().0.to_owned()),
            )?;
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
        let theta0 = self
            .initial_theta()
            .clone()
            .map_or(Array1::from_elem(w_star.ncols(), F::cast(1e-2)), |v| {
                Array::from_vec(v)
            });
        let fx = self.mean().value(&xtrain.data);
        let y_t = ytrain.clone();
        let base: f64 = 10.;
        let objfn = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            let theta =
                Array1::from_shape_vec((x.len(),), x.iter().map(|v| base.powf(*v)).collect())
                    .unwrap();
            for v in theta.iter() {
                // check theta as optimizer may return nan values
                if v.is_nan() {
                    // shortcut return worst value wrt to rlf minimization
                    return f64::INFINITY;
                }
            }
            let theta = theta.mapv(F::cast);
            let rxx = self.corr().value(&x_distances.d, &theta, &w_star);
            match reduced_likelihood(&fx, rxx, &x_distances, &y_t, self.nugget()) {
                Ok(r) => unsafe { -(*(&r.0 as *const F as *const f64)) },
                Err(_) => f64::INFINITY,
            }
        };

        // Multistart: user theta0 + 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10.
        let mut theta0s = Array2::zeros((N_START + 1, theta0.len()));
        theta0s.row_mut(0).assign(&theta0.mapv(|v| F::log10(v)));
        let mut xlimits: Array2<F> = Array2::zeros((theta0.len(), 2));
        for mut row in xlimits.rows_mut() {
            row.assign(&arr1(&[F::cast(-6), F::cast(2)]));
        }
        // Use a seed here for reproducibility. Do we need to make it truly random
        // Probably no, as it is just to get init values spread over
        // [1e-6, 20] for multistart thanks to LHS method.
        let seeds = Lhs::new(&xlimits)
            .kind(egobox_doe::LhsKind::Maximin)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(N_START);
        Zip::from(theta0s.slice_mut(s![1.., ..]).rows_mut())
            .and(seeds.rows())
            .par_for_each(|mut theta, row| theta.assign(&row));

        let opt_thetas =
            theta0s.map_axis(Axis(1), |theta| optimize_theta(objfn, &theta.to_owned()));
        let opt_index = opt_thetas.map(|(_, opt_f)| opt_f).argmin().unwrap();
        let opt_theta = &(opt_thetas[opt_index]).0.mapv(F::cast);
        // println!("opt_theta={}", opt_theta);
        let rxx = self.corr().value(&x_distances.d, opt_theta, &w_star);
        let (_, inner_params) = reduced_likelihood(&fx, rxx, &x_distances, &ytrain, self.nugget())?;
        Ok(GaussianProcess {
            theta: opt_theta.to_owned(),
            mean: *self.mean(),
            corr: *self.corr(),
            inner_params,
            w_star,
            xtrain,
            ytrain,
        })
    }
}

/// Optimize gp hyper parameter theta given an initial guess `theta0`
#[cfg(feature = "nlopt")]
fn optimize_theta<ObjF, F>(objfn: ObjF, theta0: &Array1<F>) -> (Array1<f64>, f64)
where
    ObjF: Fn(&[f64], Option<&mut [f64]>, &mut ()) -> f64,
    F: Float,
{
    use nlopt::*;

    let base: f64 = 10.;
    // block to drop optimizer and allow self.corr borrowing after
    let mut optimizer = Nlopt::new(Algorithm::Cobyla, theta0.len(), objfn, Target::Minimize, ());
    let mut theta_vec = theta0
        .map(|v| unsafe { *(v as *const F as *const f64) })
        .into_raw_vec();
    optimizer.set_lower_bound(-6.).unwrap();
    optimizer.set_upper_bound(2.).unwrap();
    optimizer.set_initial_step1(0.5).unwrap();
    optimizer.set_maxeval(15 * theta0.len() as u32).unwrap();
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
            // println!("ERROR OPTIM in GP err={:?}", e);
            (arr1(&theta_vec).mapv(|v| base.powf(v)), f64::INFINITY)
        }
    }
}

/// Optimize gp hyper parameter theta given an initial guess `theta0`
#[cfg(not(feature = "nlopt"))]
fn optimize_theta<ObjF, F>(objfn: ObjF, theta0: &Array1<F>) -> (Array1<f64>, f64)
where
    ObjF: Fn(&[f64], Option<&mut [f64]>, &mut ()) -> f64,
    F: Float,
{
    use cobyla::{minimize, Func, StopTols};

    let base: f64 = 10.;
    let cons: Vec<&dyn Func<()>> = vec![];
    let theta_init = theta0
        .map(|v| unsafe { *(v as *const F as *const f64) })
        .into_raw_vec();

    let initial_step = 0.5;
    let ftol_rel = 1e-4;
    let maxeval = 15 * theta0.len();
    let bounds = vec![(-6., 2.); theta0.len()];

    match minimize(
        |x, u| objfn(x, None, u),
        &theta_init,
        &bounds,
        &cons,
        (),
        maxeval,
        cobyla::RhoBeg::All(initial_step),
        Some(StopTols {
            ftol_rel,
            ..StopTols::default()
        }),
    ) {
        Ok((_, x_opt, fval)) => {
            let thetas_opt = arr1(&x_opt).mapv(|v| base.powf(v));
            let fval = if f64::is_nan(fval) {
                f64::INFINITY
            } else {
                fval
            };
            (thetas_opt, fval)
        }
        Err((status, x_opt, _)) => {
            println!("ERROR Cobyla optimizer in GP status={:?}", status);
            (arr1(&x_opt).mapv(|v| base.powf(v)), f64::INFINITY)
        }
    }
}

/// Compute reduced likelihood function
/// fx: mean factors term at x samples,
/// rxx: correlation factors at x samples,
/// x_distances: pairwise distances between x samples
/// ytrain: normalized output training values
/// nugget: factor to improve numerical stability  
#[cfg(not(feature = "blas"))]
fn reduced_likelihood<F: Float>(
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
    let fxl = fx;
    // R cholesky decomposition
    let r_chol = r_mx.cholesky()?;
    // Solve generalized least squared problem
    let ft = r_chol.solve_triangular(fxl, UPLO::Lower)?;
    let (ft_qr_q, ft_qr_r) = ft.qr().unwrap().into_decomp();

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
    let yt = r_chol.solve_triangular(&ytrain.data, UPLO::Lower)?;

    let beta = ft_qr_r.solve_triangular_into(ft_qr_q.t().dot(&yt), UPLO::Upper)?;
    let rho = yt - ft.dot(&beta);
    let rho_sqr = rho.mapv(|v| v * v).sum_axis(Axis(0));

    let gamma = r_chol.t().solve_triangular_into(rho, UPLO::Upper)?;
    // The determinant of R is equal to the squared product of
    // the diagonal elements of its Cholesky decomposition r_chol
    let n_obs: F = F::cast(x_distances.n_obs);

    let logdet = r_chol.diag().mapv(|v: F| v.log10()).sum() * F::cast(2.) / n_obs;

    // Reduced likelihood
    let sigma2: Array1<F> = rho_sqr / n_obs;
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

/// See non blas version
#[cfg(feature = "blas")]
fn reduced_likelihood<F: Float>(
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

    let fxl = fx.to_owned().with_lapack();

    // R cholesky decomposition
    let r_chol = r_mx.with_lapack().cholesky(UPLO::Lower)?;

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

    let yt = r_chol.solve_triangular(
        UPLO::Lower,
        Diag::NonUnit,
        &ytrain.data.to_owned().with_lapack(),
    )?;

    let beta = ft_qr_r.solve_triangular_into(UPLO::Upper, Diag::NonUnit, ft_qr_q.t().dot(&yt))?;

    let rho = yt - ft.dot(&beta);
    let rho_sqr = rho.mapv(|v| v * v).sum_axis(Axis(0));
    let rho_sqr = rho_sqr.without_lapack();

    let gamma = r_chol
        .t()
        .solve_triangular_into(UPLO::Upper, Diag::NonUnit, rho)?;

    // The determinant of R is equal to the squared product of
    // the diagonal elements of its Cholesky decomposition r_chol
    let n_obs: F = F::cast(x_distances.n_obs);

    let logdet = r_chol
        .to_owned()
        .without_lapack()
        .diag()
        .mapv(|v: F| v.log10())
        .sum()
        * F::cast(2.)
        / n_obs;

    // Reduced likelihood
    let sigma2: Array1<F> = rho_sqr / n_obs;
    let reduced_likelihood = -n_obs * (sigma2.sum().log10() + logdet);
    Ok((
        reduced_likelihood,
        GpInnerParams {
            sigma2: sigma2 * &ytrain.std.mapv(|v| v * v),
            beta: beta.without_lapack(),
            gamma: gamma.without_lapack(),
            r_chol: r_chol.without_lapack(),
            ft: ft.without_lapack(),
            ft_qr_r: ft_qr_r.without_lapack(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use argmin_testfunctions::rosenbrock;
    use egobox_doe::{Lhs, SamplingMethod};
    use linfa::prelude::Predict;
    #[cfg(not(feature = "blas"))]
    use linfa_linalg::norm::*;
    use ndarray::{arr2, array, Array, Zip};
    #[cfg(feature = "blas")]
    use ndarray_linalg::Norm;
    use ndarray_npy::{read_npy, write_npy};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use ndarray_stats::DeviationExt;
    use paste::paste;
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    fn test_constant_function() {
        let dim = 3;
        let lim = array![[0., 1.]];
        let xlimits = lim.broadcast((dim, 2)).unwrap();
        let rng = Xoshiro256Plus::seed_from_u64(42);
        let nt = 5;
        let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
        let yt = Array::from_vec(vec![3.1; nt]).insert_axis(Axis(1));
        let gp = GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
            ConstantMean::default(),
            SquaredExponentialCorr::default(),
        )
        .initial_theta(Some(vec![0.1]))
        .kpls_dim(Some(1))
        .fit(&Dataset::new(xt, yt))
        .expect("GP fit error");
        let rng = Xoshiro256Plus::seed_from_u64(43);
        let xtest = Lhs::new(&xlimits).with_rng(rng).sample(nt);
        let ytest = gp.predict_values(&xtest).expect("prediction error");
        assert_abs_diff_eq!(Array::from_elem((nt, 1), 3.1), ytest, epsilon = 1e-6);
    }

    macro_rules! test_gp {
        ($regr:ident, $corr:ident) => {
            paste! {

                #[test]
                fn [<test_gp_ $regr:snake _ $corr:snake >]() {
                    let xt = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
                    let xplot = Array::linspace(0., 4., 100).insert_axis(Axis(1));
                    let yt = array![[0.0], [1.0], [1.5], [0.9], [1.0]];
                    let gp = GaussianProcess::<f64, [<$regr Mean>], [<$corr Corr>] >::params(
                        [<$regr Mean>]::default(),
                        [<$corr Corr>]::default(),
                    )
                    .initial_theta(Some(vec![0.1]))
                    .fit(&Dataset::new(xt, yt))
                    .expect("GP fit error");
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

                    let test_dir = "target/tests";
                    std::fs::create_dir_all(test_dir).ok();

                    let xplot_file = stringify!([<gp_x_ $regr:snake _ $corr:snake >]);
                    let file_path = format!("{}/{}.npy", test_dir, xplot_file);
                    write_npy(file_path, &xplot).expect("x saved");

                    let gp_vals_file = stringify!([<gp_vals_ $regr:snake _ $corr:snake >]);
                    let file_path = format!("{}/{}.npy", test_dir, gp_vals_file);
                    write_npy(file_path, &gpr_vals).expect("gp vals saved");

                    let gp_vars_file = stringify!([<gp_vars_ $regr:snake _ $corr:snake >]);
                    let file_path = format!("{}/{}.npy", test_dir, gp_vars_file);
                    write_npy(file_path, &gpr_vars).expect("gp vars saved");
                }
            }
        };
    }

    test_gp!(Constant, SquaredExponential);
    test_gp!(Constant, AbsoluteExponential);
    test_gp!(Constant, Matern32);
    test_gp!(Constant, Matern52);

    test_gp!(Linear, SquaredExponential);
    test_gp!(Linear, AbsoluteExponential);
    test_gp!(Linear, Matern32);
    test_gp!(Linear, Matern52);

    test_gp!(Quadratic, SquaredExponential);
    test_gp!(Quadratic, AbsoluteExponential);
    test_gp!(Quadratic, Matern32);
    test_gp!(Quadratic, Matern52);

    #[test]
    fn test_kpls_griewank() {
        let dims = [5, 10, 20]; //, 60];
        let nts = [100, 300, 400]; //, 800];

        let test_dir = "target/tests";
        std::fs::create_dir_all(test_dir).ok();

        (0..2).for_each(|i| {
            let dim = dims[i];
            let nt = nts[i];

            let griewank = |x: &Array1<f64>| -> f64 {
                let d = Array1::linspace(1., dim as f64, dim).mapv(|v| v.sqrt());
                x.mapv(|v| v * v).sum() / 4000.
                    - (x / &d).mapv(|v| v.cos()).fold(1., |acc, x| acc * x)
                    + 1.0
            };
            let prefix = "gp";
            let xfilename = format!("{test_dir}/{prefix}_xt_{nt}x{dim}.npy");
            let yfilename = format!("{}/{}_yt_{}x{}.npy", test_dir, prefix, nt, 1);

            let xt = match read_npy(&xfilename) {
                Ok(xt) => xt,
                Err(_) => {
                    let lim = array![[-600., 600.]];
                    let xlimits = lim.broadcast((dim, 2)).unwrap();
                    let rng = Xoshiro256Plus::seed_from_u64(42);
                    let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
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

            let gp = GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
                ConstantMean::default(),
                SquaredExponentialCorr::default(),
            )
            .kpls_dim(Some(3))
            .fit(&Dataset::new(xt, yt))
            .expect("GP fit error");

            let xtest = Array2::ones((1, dim));
            let ytest = gp.predict_values(&xtest).expect("prediction error");
            let ytrue = griewank(&xtest.row(0).to_owned());
            assert_abs_diff_eq!(Array::from_elem((1, 1), ytrue), ytest, epsilon = 1.1);
        });
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
        let rng = Xoshiro256Plus::seed_from_u64(42);
        let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
        let yt = tensor_product_exp(&xt);

        let gp = GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
            ConstantMean::default(),
            SquaredExponentialCorr::default(),
        )
        .kpls_dim(Some(1))
        .fit(&Dataset::new(xt, yt))
        .expect("GP training");

        let xv = Lhs::new(&xlimits).sample(100);
        let yv = tensor_product_exp(&xv);

        let ytest = gp.predict_values(&xv).unwrap();
        let err = ytest.l2_dist(&yv).unwrap() / yv.norm_l2();
        assert_abs_diff_eq!(err, 0., epsilon = 2e-2);
    }

    fn rosenb(x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
        Zip::from(y.rows_mut())
            .and(x.rows())
            .par_for_each(|mut yi, xi| {
                yi.assign(&array![rosenbrock(&xi.to_vec(), 1., 100.)]);
            });
        y
    }

    #[test]
    fn test_kpls_rosenb() {
        let dim = 20;
        let nt = 30;
        let lim = array![[-1., 1.]];
        let xlimits = lim.broadcast((dim, 2)).unwrap();
        let rng = Xoshiro256Plus::seed_from_u64(42);
        let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
        let yt = rosenb(&xt);

        let gp = GaussianProcess::<f64, ConstantMean, Matern32Corr>::params(
            ConstantMean::default(),
            Matern52Corr::default(),
        )
        .kpls_dim(Some(1))
        .fit(&Dataset::new(xt.to_owned(), yt))
        .expect("GP training");

        let rng2 = Xoshiro256Plus::seed_from_u64(41);
        let xv = Lhs::new(&xlimits).with_rng(rng2).sample(300);
        let yv = rosenb(&xv);

        let ytest = gp.predict(&xv);
        let err = ytest.l2_dist(&yv).unwrap() / yv.norm_l2();
        assert_abs_diff_eq!(err, 0., epsilon = 4e-1);

        let var = GpVariancePredictor(&gp).predict(&xt);
        assert_abs_diff_eq!(var, Array2::zeros((nt, 1)), epsilon = 2e-1);
    }

    fn sphere(x: &Array2<f64>) -> Array2<f64> {
        let s = (x * x).sum_axis(Axis(1));
        s.insert_axis(Axis(1))
    }

    fn dsphere(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| 2. * v)
    }

    fn norm1(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.abs())
            .sum_axis(Axis(1))
            .insert_axis(Axis(1))
            .to_owned()
    }

    fn dnorm1(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0. { 1. } else { -1. })
    }

    macro_rules! test_gp_derivatives {
        ($regr:ident, $corr:ident, $func:ident, $limit:expr, $nt:expr) => {
            paste! {

                #[test]
                fn [<test_gp_derivatives_ $regr:snake _ $corr:snake>]() {
                    let mut rng = Xoshiro256Plus::seed_from_u64(42);
                    let xt = egobox_doe::Lhs::new(&array![[-$limit, $limit], [-$limit, $limit]])
                    .kind(egobox_doe::LhsKind::CenteredMaximin)
                    .with_rng(rng.clone())
                    .sample($nt);

                    let yt = [<$func>](&xt);
                    let gp = GaussianProcess::<f64, [<$regr Mean>], [<$corr Corr>] >::params(
                        [<$regr Mean>]::default(),
                        [<$corr Corr>]::default(),
                    )
                    .fit(&Dataset::new(xt, yt))
                    .expect("GP fitting");

                    let x = Array::random_using((2,), Uniform::new(-$limit, $limit), &mut rng);
                    //let x = array![3., 5.];
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

                    let y_pred = gp.predict_values(&x).unwrap();
                    println!("value at [{},{}] = {}", xa, xb, y_pred);
                    let y_deriv = gp.predict_derivatives(&x);
                    println!("deriv at [{},{}] = {}", xa, xb, y_deriv);
                    let true_deriv = [<d $func>](&array![[xa, xb]]);
                    println!("true deriv at [{},{}] = {}", xa, xb, true_deriv);
                    println!("jacob = at [{},{}] = {}", xa, xb, gp.predict_jacobian(&array![xa, xb]));

                    let diff_g = (y_pred[[1, 0]] - y_pred[[2, 0]]) / (2. * e);
                    let diff_d = (y_pred[[3, 0]] - y_pred[[4, 0]]) / (2. * e);

                    // test only if fdiff is not largely wrong
                    if (diff_g-true_deriv[[0, 0]]).abs() < 10. {
                        assert_rel_or_abs_error(y_deriv[[0, 0]], diff_g);
                    }
                    if (diff_d-true_deriv[[0, 1]]).abs() < 10. {
                        assert_rel_or_abs_error(y_deriv[[0, 1]], diff_d);
                    }
                }
            }
        };
    }

    test_gp_derivatives!(Constant, SquaredExponential, sphere, 10., 10);
    test_gp_derivatives!(Linear, SquaredExponential, sphere, 10., 10);
    test_gp_derivatives!(Quadratic, SquaredExponential, sphere, 10., 10);
    test_gp_derivatives!(Constant, AbsoluteExponential, sphere, 10., 10);
    test_gp_derivatives!(Linear, AbsoluteExponential, norm1, 10., 16);
    test_gp_derivatives!(Quadratic, AbsoluteExponential, norm1, 10., 16);
    test_gp_derivatives!(Constant, Matern32, norm1, 10., 16);
    test_gp_derivatives!(Linear, Matern32, norm1, 10., 16);
    test_gp_derivatives!(Quadratic, Matern32, sphere, 10., 16);
    test_gp_derivatives!(Constant, Matern52, norm1, 10., 16);
    test_gp_derivatives!(Linear, Matern52, norm1, 10., 16);
    test_gp_derivatives!(Quadratic, Matern52, sphere, 10., 10);

    #[allow(unused_macros)]
    macro_rules! test_gp_variance_derivatives {
        ($regr:ident, $corr:ident, $func:ident, $limit:expr, $nt:expr) => {
            paste! {

                #[test]
                fn [<test_gp_variance_derivatives_ $regr:snake _ $corr:snake>]() {
                    let mut rng = Xoshiro256Plus::seed_from_u64(42);
                    let xt = egobox_doe::Lhs::new(&array![[-$limit, $limit], [-$limit, $limit]]).with_rng(rng.clone()).sample($nt);
                    let yt = [<$func>](&xt);

                    let gp = GaussianProcess::<f64, [<$regr Mean>], [<$corr Corr>] >::params(
                        [<$regr Mean>]::default(),
                        [<$corr Corr>]::default(),
                    )
                    .fit(&Dataset::new(xt, yt))
                    .expect("GP fitting");

                    for _ in 0..10 {
                        let x = Array::random_using((2,), Uniform::new(-$limit, $limit), &mut rng);
                        let xa: f64 = x[0];
                        let xb: f64 = x[1];
                        let e = 1e-4;

                        let x = array![
                            [xa, xb],
                            [xa + e, xb],
                            [xa - e, xb],
                            [xa, xb + e],
                            [xa, xb - e]
                        ];
                        println!("****************************************");
                        let y_pred = gp.predict_values(&x).unwrap();
                        println!("value at [{},{}] = {}", xa, xb, y_pred);
                        let y_deriv = gp.predict_derivatives(&x);
                        println!("deriv at [{},{}] = {}", xa, xb, y_deriv);
                        let y_pred = gp.predict_variances(&x).unwrap();
                        println!("variance at [{},{}] = {}", xa, xb, y_pred);
                        let y_deriv = gp.predict_variance_derivatives(&x);
                        println!("variance deriv at [{},{}] = {}", xa, xb, y_deriv);

                        let diff_g = (y_pred[[1, 0]] - y_pred[[2, 0]]) / (2. * e);
                        let diff_d = (y_pred[[3, 0]] - y_pred[[4, 0]]) / (2. * e);

                        if "[<$corr>]" == "SquaredExponential" {
                            assert_rel_or_abs_error(y_deriv[[0, 0]], diff_g);
                            assert_rel_or_abs_error(y_deriv[[0, 1]], diff_d);
                        } else {
                            assert_abs_diff_eq!(y_deriv[[0, 0]], diff_g, epsilon=5e-1);
                            assert_abs_diff_eq!(y_deriv[[1, 0]], diff_d, epsilon=5e-1);
                        }
                    }
                }
            }
        };
    }

    test_gp_variance_derivatives!(Constant, SquaredExponential, sphere, 10., 100);
    test_gp_variance_derivatives!(Linear, SquaredExponential, sphere, 10., 100);
    test_gp_variance_derivatives!(Quadratic, SquaredExponential, sphere, 10., 100);
    test_gp_variance_derivatives!(Constant, AbsoluteExponential, norm1, 10., 100);
    test_gp_variance_derivatives!(Linear, AbsoluteExponential, norm1, 1., 50);
    test_gp_variance_derivatives!(Quadratic, AbsoluteExponential, sphere, 10., 100);
    test_gp_variance_derivatives!(Constant, Matern32, sphere, 10., 100);
    test_gp_variance_derivatives!(Linear, Matern32, norm1, 1., 50);
    test_gp_variance_derivatives!(Quadratic, Matern32, sphere, 10., 100);
    test_gp_variance_derivatives!(Constant, Matern52, sphere, 10., 100);
    test_gp_variance_derivatives!(Linear, Matern52, norm1, 1., 50);
    test_gp_variance_derivatives!(Quadratic, Matern52, sphere, 10., 100);

    #[test]
    fn test_variance_derivatives() {
        let xt = egobox_doe::FullFactorial::new(&array![[-10., 10.], [-10., 10.]]).sample(10);
        let yt = sphere(&xt);

        let gp = GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
            ConstantMean::default(),
            SquaredExponentialCorr::default(),
        )
        .fit(&Dataset::new(xt, yt))
        .expect("GP fitting");

        for _ in 0..20 {
            let mut rng = Xoshiro256Plus::seed_from_u64(42);
            let x = Array::random_using((2,), Uniform::new(-10., 10.), &mut rng);
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
            let y_pred = gp.predict_variances(&x).unwrap();
            println!("variance at [{xa},{xb}] = {y_pred}");
            let y_deriv = gp.predict_variance_derivatives(&x);
            println!("variance deriv at [{xa},{xb}] = {y_deriv}");

            let diff_g = (y_pred[[1, 0]] - y_pred[[2, 0]]) / (2. * e);
            let diff_d = (y_pred[[3, 0]] - y_pred[[4, 0]]) / (2. * e);

            // TODO: still brittle, to be reworked
            if y_pred[[0, 0]].abs() > 1e-1 && y_pred[[0, 0]].abs() > 1e-1 {
                // do not test with fdiff when variance or deriv is too small
                assert_rel_or_abs_error(y_deriv[[0, 0]], diff_g);
            }
            if y_pred[[0, 0]].abs() > 1e-1 && y_pred[[0, 0]].abs() > 1e-1 {
                // do not test with fdiff when variance or deriv  is too small
                assert_rel_or_abs_error(y_deriv[[0, 1]], diff_d);
            }
        }
    }

    fn x2sinx(x: &Array2<f64>) -> Array2<f64> {
        (x * x) * (x).mapv(|v| v.sin())
    }

    #[test]
    fn test_sampling() {
        let xdoe = array![[-8.5], [-4.0], [-3.0], [-1.0], [4.0], [7.5]];
        let ydoe = x2sinx(&xdoe);
        let krg = Kriging::<f64>::params()
            .fit(&Dataset::new(xdoe, ydoe))
            .expect("Kriging training");
        let n_plot = 35;
        let n_traj = 10;
        let (x_min, x_max) = (-10., 10.);
        let x = Array::linspace(x_min, x_max, n_plot)
            .into_shape((n_plot, 1))
            .unwrap();
        let trajs = krg.sample(&x, n_traj);
        assert_eq!(&[n_plot, n_traj], trajs.shape())
    }

    #[test]
    fn test_sampling_eigen() {
        let xdoe = array![[-8.5], [-4.0], [-3.0], [-1.0], [4.0], [7.5]];
        let ydoe = x2sinx(&xdoe);
        let krg = Kriging::<f64>::params()
            .fit(&Dataset::new(xdoe, ydoe))
            .expect("Kriging training");
        let n_plot = 500;
        let n_traj = 10;
        let (x_min, x_max) = (-10., 10.);
        let x = Array::linspace(x_min, x_max, n_plot)
            .into_shape((n_plot, 1))
            .unwrap();
        let trajs = krg.sample_eig(&x, n_traj);
        assert_eq!(&[n_plot, n_traj], trajs.shape());
        assert!(!trajs.fold(false, |acc, v| acc || v.is_nan())); // check no nans
    }

    fn assert_rel_or_abs_error(y_deriv: f64, fdiff: f64) {
        println!("analytic deriv = {y_deriv}, fdiff = {fdiff}");
        if fdiff.abs() < 2e-1 {
            let atol = 2e-1;
            println!("Check absolute error: should be < {atol}");
            assert_abs_diff_eq!(y_deriv, 0.0, epsilon = atol); // check absolute when close to zero
        } else {
            let rtol = 2e-1;
            let rel_error = (y_deriv - fdiff).abs() / fdiff; // check relative
            println!("Check relative error: should be < {rtol}");
            assert_abs_diff_eq!(rel_error, 0.0, epsilon = rtol);
        }
    }
}
