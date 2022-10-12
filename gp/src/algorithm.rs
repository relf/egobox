use crate::correlation_models::*;
use crate::errors::{GpError, Result};
use crate::mean_models::*;
use crate::parameters::{GpParams, GpValidParams};
use crate::utils::{pairwise_differences, DistanceMatrix, NormalizedMatrix};
use egobox_doe::{Lhs, SamplingMethod};
#[cfg(feature = "blas")]
use linfa::dataset::{WithLapack, WithoutLapack};
use linfa::prelude::{Dataset, DatasetBase, Fit, Float, PredictInplace};
#[cfg(not(feature = "blas"))]
use linfa_linalg::{cholesky::*, qr::*, svd::*, triangular::*};
use linfa_pls::PlsRegression;
use ndarray::{arr1, s, Array, Array1, Array2, ArrayBase, Axis, Data, Ix2, Zip};
use ndarray_einsum_beta::*;
#[cfg(feature = "blas")]
use ndarray_linalg::{cholesky::*, qr::*, svd::*, triangular::*};
use ndarray_rand::rand::SeedableRng;
use ndarray_stats::QuantileExt;
use nlopt::*;
use rand_isaac::Isaac64Rng;
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

const LOG10_20: f64 = 1.301_029_995_663_981_3; //f64::log10(20.);

/// Internal parameters computed Gp during training
/// used later on in prediction computations
#[derive(Default, Debug)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(bound(deserialize = "F: Deserialize<'de>"))
)]
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

/// Gaussian Process
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

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> GaussianProcess<F, Mean, Corr> {
    /// Gp parameters contructor
    pub fn params<NewMean: RegressionModel<F>, NewCorr: CorrelationModel<F>>(
        mean: NewMean,
        corr: NewCorr,
    ) -> GpParams<F, NewMean, NewCorr> {
        GpParams::new(mean, corr)
    }

    /// Predict output values at the n given points specified as (n, ndim) matrix.
    /// Returns output values as (n, 1) matrix.
    pub fn predict_values(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Result<Array2<F>> {
        let corr = self._compute_correlation(x);
        // Compute the mean at x
        let f = self.mean.apply(x);
        // Scaled predictor
        let y_ = &f.dot(&self.inner_params.beta) + &corr.dot(&self.inner_params.gamma);
        // Predictor
        Ok(&y_ * &self.ytrain.std + &self.ytrain.mean)
    }

    /// Predict variance values at the n given points.
    /// Returns variance values as (n, 1) matrix.
    pub fn predict_variances(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Result<Array2<F>> {
        let corr = self._compute_correlation(x);
        let inners = &self.inner_params;

        let corr_t = corr.t().to_owned();
        #[cfg(feature = "blas")]
        let rt = inners
            .r_chol
            .to_owned()
            .with_lapack()
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &corr_t.with_lapack())?
            .without_lapack();
        #[cfg(not(feature = "blas"))]
        let rt = inners.r_chol.solve_triangular(&corr_t, UPLO::Lower)?;

        let lhs = inners.ft.t().dot(&rt) - self.mean.apply(x).t();
        #[cfg(feature = "blas")]
        let u = inners
            .ft_qr_r
            .to_owned()
            .t()
            .with_lapack()
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &lhs.with_lapack())?
            .without_lapack();
        #[cfg(not(feature = "blas"))]
        let u = inners.ft_qr_r.t().solve_triangular(&lhs, UPLO::Lower)?;

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
        let xnorm = (x - &self.xtrain.mean) / &self.xtrain.std;
        // Get pairwise componentwise L1-distances to the input training set
        let dx = pairwise_differences(&xnorm, &self.xtrain.data);
        // Compute the correlation function
        let r = self.corr.apply(&self.theta, &dx, &self.w_star);
        let n_obs = x.nrows();
        let nt = self.xtrain.data.nrows();
        r.into_shape((n_obs, nt)).unwrap().to_owned()
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
        self.ytrain.ncols()
    }

    /// Retrieve output dimension
    pub fn output_dim(&self) -> usize {
        self.ytrain.ncols()
    }
    // }

    #[cfg(feature = "blas")]
    pub fn predict_derivatives(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
        kx: usize,
    ) -> Array1<F> {
        Array1::zeros((1,))
    }

    // impl<F: Float> GaussianProcess<F, ConstantMean, SquaredExponentialCorr> {
    /// Predict derivatives of the output prediction
    /// wrt the kx th components at point a set of points x \[n_samples, n_components\].
    #[cfg(not(feature = "blas"))]
    pub fn predict_derivatives(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
        kx: usize,
    ) -> Array1<F> {
        let corr = self._compute_correlation(x);
        // println!("r = {}", corr);
        let x = (x - &self.xtrain.mean) / &self.xtrain.std;

        let df = Array2::<F>::zeros((1, x.ncols()));
        let beta = &self.inner_params.beta;
        let gamma = &self.inner_params.gamma;
        let df_dx = &df.t().dot(beta);
        // println!("x = {}", x);
        // println!("df_dx = {}", df_dx);
        // println!(
        //     "shapes df={:?}, beta={:?}, df_dx={:?}",
        //     df.shape(),
        //     beta.shape(),
        //     df_dx.shape()
        // );

        let nr = x.nrows();
        let nc = self.xtrain.data.nrows();
        let d_dx_1 = &x
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
        // println!("d_dx = {}", d_dx);
        // println!("theta = {}", self.theta);
        // Get pairwise componentwise L1-distances to the input training set
        let theta = &self.theta.to_owned();
        let d_dx_corr = d_dx * corr;
        let res = (df_dx.row(kx).to_owned()
            - F::cast(2.) * theta[kx] * d_dx_corr.dot(gamma)[[0, 0]])
            * self.ytrain.std[0]
            / self.xtrain.std[kx];
        res
    }

    #[cfg(feature = "blas")]
    pub fn predict_jacobian(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        return Array2::<F>::zeros((1, 1));
    }

    /// Predict jacobian at one point x
    #[cfg(not(feature = "blas"))]
    pub fn predict_jacobian(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        let mut jac = Array2::zeros((self.xtrain.data.ncols(), 1));
        Zip::indexed(jac.rows_mut()).for_each(|i, mut r| {
            let pred = self.predict_derivatives(x, i);
            // println!("df/dx{}={}", i, pred);
            r.assign(&pred);
        });
        jac
    }

    #[cfg(feature = "blas")]
    pub fn predict_variance_jacobian(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        return Array2::<F>::zeros((1, 1));
    }

    /// Predict derivatives of the output prediction variance
    /// wrt the kx th components at point one input.
    #[cfg(not(feature = "blas"))]
    pub fn predict_variance_jacobian(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        // Initialization
        let xnorm = (x - &self.xtrain.mean) / &self.xtrain.std;
        let theta = &self.theta;
        // println!("x={:?}", xnorm);
        // Get pairwise componentwise L1-distances to the input training set
        // dx = differences(x, Y=self.X_norma.copy())
        let dx = pairwise_differences(&xnorm, &self.xtrain.data);
        // println!("dx={:?}", dx);
        // d = self._componentwise_distance(dx)
        // dd = self._componentwise_distance(
        //     dx, theta=self.optimal_theta, return_derivative=True
        // )
        let dd = einsum("j,ij->ij", &[&theta.t(), &dx])
            .unwrap()
            .mapv(|v| F::cast(2) * v);
        // println!("dd={:?}", dd);
        let sigma2 = &self.inner_params.sigma2;
        let cholesky_k = &self.inner_params.r_chol;

        // derivative_dic = {"dx": dx, "dd": dd}
        // r, dr = self._correlation_types[self.options["corr"]](
        //     theta, d, derivative_params=derivative_dic
        // )
        let r = self.corr.apply(&self.theta, &dx, &self.w_star);
        let dr = -einsum("i,ij->ij", &[&r.t().row(0), &dd])
            .unwrap()
            .into_shape((dd.shape()[0], dd.shape()[1]))
            .unwrap();
        // println!("r={:?}", r);
        // println!("dr={:?}", dr);

        let rho1 = cholesky_k.solve_triangular(&r, UPLO::Lower).unwrap();
        let inv_kr = cholesky_k.t().solve_triangular(&rho1, UPLO::Upper).unwrap();

        let p1 = dr.t().dot(&inv_kr).t().to_owned();

        let p2 = inv_kr.t().dot(&dr);

        let f_x = self.mean.apply(x).t().to_owned(); //(x).T
        let f_mean = self.mean.apply(&self.xtrain.data);

        let rho2 = cholesky_k.solve_triangular(&f_mean, UPLO::Lower).unwrap();
        let inv_kf = cholesky_k.t().solve_triangular(&rho2, UPLO::Upper).unwrap();

        let a_mat = f_x.t().to_owned() - r.t().dot(&inv_kf);

        let b_mat = f_mean.t().dot(&inv_kf);

        let rho3 = b_mat.cholesky().unwrap();
        let inv_bat = rho3.solve_triangular(&a_mat.t(), UPLO::Lower).unwrap();
        let d_mat = rho3.t().solve_triangular(&inv_bat, UPLO::Upper).unwrap();

        // mean = constant
        let df = Array2::zeros((1, x.ncols()));
        // if self.options["poly"] == "constant":
        //     df = np.zeros((1, self.nx))
        // elif self.options["poly"] == "linear":
        //     df = np.zeros((self.nx + 1, self.nx))
        //     df[1:, :] = np.eye(self.nx)
        // else:
        //     raise ValueError(
        //         "The derivative is only available for ordinary kriging or "
        //         + "universal kriging using a linear trend"
        //     )

        let d_a = df.t().to_owned() - dr.t().dot(&inv_kf);
        let p3 = d_a.dot(&d_mat).t().to_owned();
        let p4 = d_mat.t().dot(&d_a.t());
        let prime_t = (-p1 - p2 + p3 + p4).t().to_owned();

        // derived_variance = []
        let x_std = &self.xtrain.std;
        let mut dvar = Array2::<F>::zeros((x_std.len(), x_std.len()));
        // for i in range(len(x_std)):
        //     derived_variance.append(sigma2 * prime.T[i] / x_std[i])
        Zip::from(dvar.rows_mut())
            .and(prime_t.rows())
            .for_each(|mut dv, p| {
                let dv_val = (sigma2.to_owned() * p) / x_std;
                // println!("sigma {}", sigma2);
                // println!("p {}", p);
                // println!("x_std {}", x_std);
                // println!("dv {}", dv);
                // println!("dvval {}", dv_val);
                dv.assign(&dv_val);
                // println!("dv {}", dv);
            });

        dvar.t().to_owned()
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
        let fx = self.mean().apply(x);
        let y_t = ytrain.clone();
        let base: f64 = 10.;
        let objfn = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            let theta =
                Array1::from_shape_vec((x.len(),), x.iter().map(|v| base.powf(*v)).collect())
                    .unwrap();
            let theta = theta.mapv(F::cast);
            let rxx = self.corr().apply(&theta, &x_distances.d, &w_star);
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
        let seeds = Lhs::new(&xlimits)
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

        let rxx = self.corr().apply(opt_theta, &x_distances.d, &w_star);
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

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> GpValidParams<F, Mean, Corr> {
    /// Constructor of valid params from values
    #[doc(hidden)]
    pub fn from(
        mean: Mean,
        corr: Corr,
        theta: Array1<F>,
        inner_params: GpInnerParams<F>,
        w_star: Array2<F>,
        xtrain: NormalizedMatrix<F>,
        ytrain: NormalizedMatrix<F>,
    ) -> Result<GaussianProcess<F, Mean, Corr>> {
        //TODO: add some consistency checks
        Ok(GaussianProcess {
            mean,
            corr,
            theta,
            inner_params,
            w_star,
            xtrain,
            ytrain,
        })
    }
}

fn optimize_theta<ObjF, F>(objfn: ObjF, theta0: &Array1<F>) -> (Array1<f64>, f64)
where
    ObjF: Fn(&[f64], Option<&mut [f64]>, &mut ()) -> f64,
    F: Float,
{
    let base: f64 = 10.;
    // block to drop optimizer and allow self.corr borrowing after
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
            // println!("ERROR OPTIM in GP {:?}", e);
            (arr1(&theta_vec).mapv(|v| base.powf(v)), f64::INFINITY)
        }
    }
}

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
    use ndarray_rand::rand;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_stats::DeviationExt;
    use paste::paste;
    use rand_isaac::Isaac64Rng;

    #[test]
    fn test_constant_function() {
        let dim = 3;
        let lim = array![[0., 1.]];
        let xlimits = lim.broadcast((dim, 2)).unwrap();
        let rng = Isaac64Rng::seed_from_u64(42);
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
        let rng = Isaac64Rng::seed_from_u64(43);
        let xtest = Lhs::new(&xlimits).with_rng(rng).sample(nt);
        let ytest = gp.predict_values(&xtest).expect("prediction error");
        assert_abs_diff_eq!(Array::from_elem((nt, 1), 3.1), ytest, epsilon = 1e-6);
    }

    macro_rules! test_gp {
        ($regr:ident, $corr:ident, $expected:expr) => {
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

    test_gp!(Constant, SquaredExponential, 1.67);
    test_gp!(Constant, AbsoluteExponential, 22.35);
    test_gp!(Constant, Matern32, 21.68);
    test_gp!(Constant, Matern52, 21.68);

    test_gp!(Linear, SquaredExponential, 1.56);
    test_gp!(Linear, AbsoluteExponential, 21.68);
    test_gp!(Linear, Matern32, 21.68);
    test_gp!(Linear, Matern52, 21.68);

    test_gp!(Quadratic, SquaredExponential, 21.14);
    test_gp!(Quadratic, AbsoluteExponential, 24.98);
    test_gp!(Quadratic, Matern32, 22.35);
    test_gp!(Quadratic, Matern52, 21.68);

    #[test]
    fn test_kpls_griewank() {
        let dims = vec![5, 10, 20]; //, 60];
        let nts = vec![100, 300, 400]; //, 800];

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
            let xfilename = format!("{}/{}_xt_{}x{}.npy", test_dir, prefix, nt, dim);
            let yfilename = format!("{}/{}_yt_{}x{}.npy", test_dir, prefix, nt, 1);

            let xt = match read_npy(&xfilename) {
                Ok(xt) => xt,
                Err(_) => {
                    let lim = array![[-600., 600.]];
                    let xlimits = lim.broadcast((dim, 2)).unwrap();
                    let rng = Isaac64Rng::seed_from_u64(42);
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
        let rng = Isaac64Rng::seed_from_u64(42);
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
        let nt = 200;
        let lim = array![[-1., 1.]];
        let xlimits = lim.broadcast((dim, 2)).unwrap();
        let rng = Isaac64Rng::seed_from_u64(42);
        let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
        let yt = rosenb(&xt);

        let gp = GaussianProcess::<f64, ConstantMean, Matern32Corr>::params(
            ConstantMean::default(),
            Matern52Corr::default(),
        )
        .kpls_dim(Some(1))
        .fit(&Dataset::new(xt.to_owned(), yt))
        .expect("GP training");

        let xv = Lhs::new(&xlimits).sample(500);
        let yv = rosenb(&xv);

        let ytest = gp.predict(&xv);
        let err = ytest.l2_dist(&yv).unwrap() / yv.norm_l2();
        assert_abs_diff_eq!(err, 0., epsilon = 2e-1);

        let var = GpVariancePredictor(&gp).predict(&xt);
        assert_abs_diff_eq!(var, Array2::zeros((nt, 1)), epsilon = 2e-1);
    }

    fn sphere(x: &Array2<f64>) -> Array2<f64> {
        let s = (x * x).sum_axis(Axis(1));
        s.insert_axis(Axis(1))
    }

    fn dsphere(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| 2. * v).t().to_owned()
    }

    #[cfg(not(feature = "blas"))]
    #[test]
    fn test_derivatives() {
        let xt = egobox_doe::Lhs::new(&array![[-10., 10.], [-10., 10.]]).sample(100);
        let yt = sphere(&xt);
        let gp = GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
            ConstantMean::default(),
            SquaredExponentialCorr::default(),
        )
        .fit(&Dataset::new(xt, yt))
        .expect("GP fitting");

        let x1: f64 = rand::random::<f64>() * 20. - 10.;
        let x2: f64 = rand::random::<f64>() * 20. - 10.;
        let xtest = array![[x1, x2]];

        let jac = gp.predict_jacobian(&xtest);
        let df = dsphere(&xtest);

        let jac_rel_err1 = (jac[[0, 0]] - df[[0, 0]]).abs() / jac[[0, 0]];
        let jac_rel_err2 = (jac[[1, 0]] - df[[1, 0]]).abs() / jac[[1, 0]];
        println!("Test sphere predicted derivatives at {}", xtest);
        assert_abs_diff_eq!(jac_rel_err1, 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(jac_rel_err2, 0.0, epsilon = 1e-3);
    }

    #[cfg(not(feature = "blas"))]
    #[test]
    fn test_variance_derivatives() {
        let xt = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let yt = array![[0.0], [1.0], [1.5], [0.9], [1.0]];
        let gp = GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
            ConstantMean::default(),
            SquaredExponentialCorr::default(),
        )
        .fit(&Dataset::new(xt, yt))
        .expect("GP fitting");

        let x = array![[0.5]];
        let dvar = gp.predict_variance_jacobian(&x);
        println!("dvar={}", dvar)
    }
}
