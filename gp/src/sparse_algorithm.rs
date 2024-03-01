use crate::algorithm::optimize_params;
use crate::errors::{GpError, Result};
use crate::sparse_parameters::{
    Inducings, ParamEstimation, SgpParams, SgpValidParams, SparseMethod,
};
use crate::{correlation_models::*, utils::pairwise_differences};
use crate::{prepare_multistart, CobylaParams};
use linfa::prelude::{Dataset, DatasetBase, Fit, Float, PredictInplace};
use linfa_linalg::{cholesky::*, triangular::*};
use linfa_pls::PlsRegression;
use ndarray::{s, Array, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix2, Zip};
use ndarray_einsum_beta::*;
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use log::debug;
use rayon::prelude::*;
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Instant;

/// Woodbury data computed during training and used for prediction
///
/// Name came from [Woodbury matrix identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))
)]
#[derive(Debug)]
pub(crate) struct WoodburyData<F: Float> {
    vec: Array2<F>,
    inv: Array2<F>,
}
impl<F: Float> Clone for WoodburyData<F> {
    fn clone(&self) -> Self {
        WoodburyData {
            vec: self.vec.to_owned(),
            inv: self.inv.clone(),
        }
    }
}

/// Sparse gaussian process considers a set of `M` inducing points either to approximate the posterior Gaussian distribution
/// with a low-rank representation (FITC - Fully Independent Training Conditional method), or to approximate the posterior
/// distribution directly (VFE - Variational Free Energy method).
///
/// These methods enable accurate modeling with large training datasets of N points while preserving
/// computational efficiency. With `M < N`, we get `O(NM^2)` complexity instead of `O(N^3)`
/// in time processing and `O(NM)` instead of `O(N^2)` in memory space.
///
/// See Reference section for more information.
///
/// # Implementation
///
/// [`SparseGaussianProcess`] inducing points definition can be either random or provided by the user through
/// the [`Inducings`] specification. The used sparse method is specified with the [`SparseMethod`].
/// Noise variance can be either specified as a known constant or estimated (see [`ParamEstimation`]).
/// Unlike [`GaussianProcess`]([crate::GaussianProcess]) implementation [`SparseGaussianProcess`]
/// does not allow choosing a trend which is supposed to be zero.
/// The correlation kernel might be selected amongst [available kernels](crate::correlation_models).
/// When targetting a squared exponential kernel, one can use the [SparseKriging] shortcut.  
///
/// # Features
///
/// ## serializable
///
/// The `serializable` feature enables the serialization of GP models using the [`serde crate`](https://serde.rs/).
///
/// # Example
///
/// ```
/// use ndarray::{Array, Array2, Axis};
/// use ndarray_rand::rand;
/// use ndarray_rand::rand::SeedableRng;
/// use ndarray_rand::RandomExt;
/// use ndarray_rand::rand_distr::{Normal, Uniform};
/// use linfa::prelude::{Dataset, DatasetBase, Fit, Float, PredictInplace};
///
/// use egobox_gp::{SparseKriging, Inducings};
///
/// const PI: f64 = std::f64::consts::PI;
///
/// // Let us define a hidden target function for our sparse GP example
/// fn f_obj(x: &Array2<f64>) -> Array2<f64> {
///   x.mapv(|v| (3. * PI * v).sin() + 0.3 * (9. * PI * v).cos() + 0.5 * (7. * PI * v).sin())
/// }
///
/// // Then we can define a utility function to generate some noisy data
/// // nt points with a gaussian noise with a variance eta2.
/// fn make_test_data(
///     nt: usize,
///     eta2: f64,
/// ) -> (Array2<f64>, Array2<f64>) {
///     let normal = Normal::new(0., eta2.sqrt()).unwrap();
///     let mut rng = rand::thread_rng();
///     let gaussian_noise = Array::<f64, _>::random_using((nt, 1), normal, &mut rng);
///     let xt = 2. * Array::<f64, _>::random_using((nt, 1), Uniform::new(0., 1.), &mut rng) - 1.;
///     let yt = f_obj(&xt) + gaussian_noise;
///     (xt, yt)
/// }
///
/// // Generate training data
/// let nt = 200;
/// // Variance of the gaussian noise on our training data
/// let eta2: f64 = 0.01;
/// let (xt, yt) = make_test_data(nt, eta2);
///
/// // Train our sparse gaussian process with n inducing points taken in the dataset
/// let n_inducings = 30;
/// let sgp = SparseKriging::params(Inducings::Randomized(n_inducings))
///     .fit(&Dataset::new(xt, yt))
///     .expect("SGP fitted");
///
/// println!("sgp theta={:?}", sgp.theta());
/// println!("sgp variance={:?}", sgp.variance());
/// println!("noise variance={:?}", sgp.noise_variance());
///
/// // Predict with our trained SGP
/// let xplot = Array::linspace(-1., 1., 100).insert_axis(Axis(1));
/// let sgp_vals = sgp.predict_values(&xplot).unwrap();
/// let sgp_vars = sgp.predict_variances(&xplot).unwrap();
/// ```
///
/// # Reference
///
/// Matthias Bauer, Mark van der Wilk, and Carl Edward Rasmussen.
/// [Understanding Probabilistic Sparse Gaussian Process Approximations](https://arxiv.org/pdf/1606.04820.pdf).
/// In: Advances in Neural Information Processing Systems. Ed. by D. Lee et al. Vol. 29. Curran Associates, Inc., 2016
///
#[derive(Debug)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))
)]
pub struct SparseGaussianProcess<F: Float, Corr: CorrelationModel<F>> {
    /// Correlation kernel
    #[cfg_attr(
        feature = "serializable",
        serde(bound(serialize = "Corr: Serialize", deserialize = "Corr: Deserialize<'de>"))
    )]
    corr: Corr,
    /// Sparse method used
    method: SparseMethod,
    /// Parameter of the autocorrelation model
    theta: Array1<F>,
    /// Estimated gaussian process variance
    sigma2: F,
    /// Gaussian noise variance
    noise: F,
    /// Reduced likelihood value (result from internal optimization)
    /// Maybe used to compare different trained models
    likelihood: F,
    /// Weights in case of KPLS dimension reduction coming from PLS regression (orig_dim, kpls_dim)
    w_star: Array2<F>,
    /// Training inputs
    xtrain: Array2<F>,
    /// Training outputs
    ytrain: Array2<F>,
    /// Inducing points
    inducings: Array2<F>,
    /// Data used for prediction
    w_data: WoodburyData<F>,
}

/// Kriging as sparse GP special case when using squared exponential correlation
pub type SparseKriging<F> = SgpParams<F, SquaredExponentialCorr>;

impl<F: Float> SparseKriging<F> {
    pub fn params(inducings: Inducings<F>) -> SgpParams<F, SquaredExponentialCorr> {
        SgpParams::new(SquaredExponentialCorr(), inducings)
    }
}

impl<F: Float, Corr: CorrelationModel<F>> Clone for SparseGaussianProcess<F, Corr> {
    fn clone(&self) -> Self {
        Self {
            corr: self.corr,
            method: self.method,
            theta: self.theta.to_owned(),
            sigma2: self.sigma2,
            noise: self.noise,
            likelihood: self.likelihood,
            w_star: self.w_star.to_owned(),
            xtrain: self.xtrain.clone(),
            ytrain: self.xtrain.clone(),
            inducings: self.inducings.clone(),
            w_data: self.w_data.clone(),
        }
    }
}

impl<F: Float, Corr: CorrelationModel<F>> fmt::Display for SparseGaussianProcess<F, Corr> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SGP(corr={}, theta={}, variance={}, noise variance={}, likelihood={})",
            self.corr, self.theta, self.sigma2, self.noise, self.likelihood
        )
    }
}

impl<F: Float, Corr: CorrelationModel<F>> SparseGaussianProcess<F, Corr> {
    /// Gp parameters contructor
    pub fn params<NewCorr: CorrelationModel<F>>(
        corr: NewCorr,
        inducings: Inducings<F>,
    ) -> SgpParams<F, NewCorr> {
        SgpParams::new(corr, inducings)
    }

    fn compute_k(
        &self,
        a: &ArrayBase<impl Data<Elem = F>, Ix2>,
        b: &ArrayBase<impl Data<Elem = F>, Ix2>,
        w_star: &Array2<F>,
        theta: &Array1<F>,
        sigma2: F,
    ) -> Array2<F> {
        // Get pairwise componentwise L1-distances to the input training set
        let dx = pairwise_differences(a, b);
        // Compute the correlation function
        let r = self.corr.value(&dx, theta, w_star);
        r.into_shape((a.nrows(), b.nrows()))
            .unwrap()
            .mapv(|v| v * sigma2)
    }

    /// Predict output values at n given `x` points of nx components specified as a (n, nx) matrix.
    /// Returns n scalar output values as (n, 1) column vector.
    pub fn predict_values(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Result<Array2<F>> {
        let kx = self.compute_k(x, &self.inducings, &self.w_star, &self.theta, self.sigma2);
        let mu = kx.dot(&self.w_data.vec);
        Ok(mu)
    }

    /// Predict variance values at n given `x` points of nx components specified as a (n, nx) matrix.
    /// Returns n variance values as (n, 1) column vector.
    pub fn predict_variances(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Result<Array2<F>> {
        let kx = self.compute_k(&self.inducings, x, &self.w_star, &self.theta, self.sigma2);
        let kxx = Array::from_elem(x.nrows(), self.sigma2);
        let var = kxx - (self.w_data.inv.t().clone().dot(&kx) * &kx).sum_axis(Axis(0));
        let var = var.mapv(|v| {
            if v < F::cast(1e-15) {
                F::cast(1e-15) + self.noise
            } else {
                v + self.noise
            }
        });
        Ok(var.insert_axis(Axis(1)))
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

    /// Optimal theta
    pub fn theta(&self) -> &Array1<F> {
        &self.theta
    }

    /// Estimated variance
    pub fn variance(&self) -> F {
        self.sigma2
    }

    /// Estimated noise variance
    pub fn noise_variance(&self) -> F {
        self.noise
    }

    /// Retrieve reduced likelihood value
    pub fn likelihood(&self) -> F {
        self.likelihood
    }

    /// Inducing points
    pub fn inducings(&self) -> &Array2<F> {
        &self.inducings
    }
}

impl<F, D, Corr> PredictInplace<ArrayBase<D, Ix2>, Array2<F>> for SparseGaussianProcess<F, Corr>
where
    F: Float,
    D: Data<Elem = F>,
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
struct GpVariancePredictor<'a, F, Corr>(&'a SparseGaussianProcess<F, Corr>)
where
    F: Float,
    Corr: CorrelationModel<F>;

impl<'a, F, D, Corr> PredictInplace<ArrayBase<D, Ix2>, Array2<F>>
    for GpVariancePredictor<'a, F, Corr>
where
    F: Float,
    D: Data<Elem = F>,
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

impl<F: Float, Corr: CorrelationModel<F>, D: Data<Elem = F> + Sync>
    Fit<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>, GpError> for SgpValidParams<F, Corr>
{
    type Object = SparseGaussianProcess<F, Corr>;

    /// Fit GP parameters using maximum likelihood
    fn fit(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
    ) -> Result<Self::Object> {
        let x = dataset.records();
        let y = dataset.targets();
        if let Some(d) = self.kpls_dim() {
            if *d > x.ncols() {
                return Err(GpError::InvalidValueError(format!(
                    "Dimension reduction {} should be smaller than actual \
                    training input dimensions {}",
                    d,
                    x.ncols()
                )));
            };
        }

        let xtrain = x;
        let ytrain = y;

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

        // Initial guess for theta
        let theta0_dim = self.theta_tuning().theta0().len();
        let theta0 = if theta0_dim == 1 {
            Array1::from_elem(w_star.ncols(), self.theta_tuning().theta0()[0])
        } else if theta0_dim == w_star.ncols() {
            Array::from_vec(self.theta_tuning().theta0().to_vec())
        } else {
            panic!("Initial guess for theta should be either 1-dim or dim of xtrain (w_star.ncols()), got {}", theta0_dim)
        };

        // Initial guess for variance
        let y_std = ytrain.std_axis(Axis(0), F::one());
        let sigma2_0 = y_std[0] * y_std[0];

        // Initial guess for noise, when noise variance constant, it is not part of optimization params
        let (is_noise_estimated, noise0) = match self.noise_variance() {
            ParamEstimation::Fixed(c) => (false, c),
            ParamEstimation::Estimated {
                initial_guess: c,
                bounds: _,
            } => (true, c),
        };

        // Params consist in [theta1, ..., thetap, sigma2, [noise]]
        // where sigma2 is the variance of the gaussian process
        // where noise is the variance of the noise when it is estimated
        let n = theta0.len() + 1 + is_noise_estimated as usize;
        let mut params_0 = Array1::zeros(n);
        params_0
            .slice_mut(s![..n - 1 - is_noise_estimated as usize])
            .assign(&theta0);
        params_0[n - 1 - is_noise_estimated as usize] = sigma2_0;
        if is_noise_estimated {
            // noise variance is estimated, noise0 is initial_guess
            params_0[n - 1] = *noise0;
        }

        let mut rng = match self.seed() {
            Some(seed) => Xoshiro256Plus::seed_from_u64(*seed),
            None => Xoshiro256Plus::from_entropy(),
        };
        let z = match self.inducings() {
            Inducings::Randomized(n) => make_inducings(*n, &xtrain.view(), &mut rng),
            Inducings::Located(z) => z.to_owned(),
        };

        // We prefer optimize variable change log10(theta)
        // as theta is used as exponent in objective function
        let base: f64 = 10.;
        let objfn = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            for v in x.iter() {
                // check theta as optimizer may give nan values
                if v.is_nan() {
                    // shortcut return worst value wrt to rlf minimization
                    return f64::INFINITY;
                }
            }
            let input = Array1::from_shape_vec(
                (x.len(),),
                x.iter().map(|v| F::cast(base.powf(*v))).collect(),
            )
            .unwrap();

            let theta = input.slice(s![..input.len() - 1 - is_noise_estimated as usize]);
            let sigma2 = input[input.len() - 1 - is_noise_estimated as usize];
            let noise = if is_noise_estimated {
                input[input.len() - 1]
            } else {
                F::cast(*noise0)
            };

            let theta = theta.mapv(F::cast);
            match self.reduced_likelihood(
                &theta,
                sigma2,
                noise,
                &w_star,
                &xtrain.view(),
                &ytrain.view(),
                &z,
                self.nugget(),
            ) {
                Ok(r) => unsafe { -(*(&r.0 as *const F as *const f64)) },
                Err(_) => f64::INFINITY,
            }
        };

        // // bounds of theta, variance and optionally noise variance
        // let mut bounds = vec![(F::cast(1e-16).log10(), F::cast(1.).log10()); params.ncols()];
        // Initial guess for theta
        let bounds_dim = self.theta_tuning().bounds().len();
        let bounds = if bounds_dim == 1 {
            vec![self.theta_tuning().bounds()[0]; params_0.len()]
        } else if theta0_dim == params_0.len() {
            self.theta_tuning().bounds().to_vec()
        } else {
            panic!(
                "Bounds for theta should be either 1-dim or dim of xtrain ({}), got {}",
                w_star.ncols(),
                theta0_dim
            )
        };

        let (params, mut bounds) = prepare_multistart(self.n_start(), &params_0, &bounds);

        // variance bounds
        bounds[params.ncols() - 1 - is_noise_estimated as usize] =
            (F::cast(1e-12).log10(), (F::cast(9.) * sigma2_0).log10());
        // optionally adjust noise variance bounds
        if let ParamEstimation::Estimated {
            initial_guess: _,
            bounds: (lo, up),
        } = self.noise_variance()
        {
            // Set bounds for noise
            if let Some(noise_bounds) = bounds.last_mut() {
                *noise_bounds = (lo.log10(), up.log10());
            }
        }
        debug!(
            "Optimize with multistart theta = {:?} and bounds = {:?}",
            params, bounds
        );
        let now = Instant::now();
        let opt_params = (0..params.nrows())
            .into_par_iter()
            .map(|i| {
                let opt_res = optimize_params(
                    objfn,
                    &params.row(i).to_owned(),
                    &bounds,
                    CobylaParams {
                        maxeval: (10 * theta0_dim).max(CobylaParams::default().maxeval),
                        ..CobylaParams::default()
                    },
                );

                opt_res
            })
            .reduce(
                || (Array::ones((params.ncols(),)), f64::INFINITY),
                |a, b| if b.1 < a.1 { b } else { a },
            );
        debug!("elapsed optim = {:?}", now.elapsed().as_millis());
        let opt_params = opt_params.0.mapv(|v| F::cast(base.powf(v)));
        let opt_theta = opt_params
            .slice(s![..n - 1 - is_noise_estimated as usize])
            .to_owned();
        let opt_sigma2 = opt_params[n - 1 - is_noise_estimated as usize];
        let opt_noise = if is_noise_estimated {
            opt_params[n - 1]
        } else {
            *noise0
        };

        // Recompute reduced likelihood with optimized params
        let (lkh, w_data) = self.reduced_likelihood(
            &opt_theta,
            opt_sigma2,
            opt_noise,
            &w_star,
            &xtrain.view(),
            &ytrain.view(),
            &z,
            self.nugget(),
        )?;
        Ok(SparseGaussianProcess {
            corr: *self.corr(),
            method: self.method(),
            theta: opt_theta,
            sigma2: opt_sigma2,
            noise: opt_noise,
            likelihood: lkh,
            w_data,
            w_star,
            xtrain: xtrain.to_owned(),
            ytrain: ytrain.to_owned(),
            inducings: z.clone(),
        })
    }
}

impl<F: Float, Corr: CorrelationModel<F>> SgpValidParams<F, Corr> {
    /// Compute reduced likelihood function
    /// nugget: factor to improve numerical stability  
    #[allow(clippy::too_many_arguments)]
    fn reduced_likelihood(
        &self,
        theta: &Array1<F>,
        sigma2: F,
        noise: F,
        w_star: &Array2<F>,
        xtrain: &ArrayView2<F>,
        ytrain: &ArrayView2<F>,
        z: &Array2<F>,
        nugget: F,
    ) -> Result<(F, WoodburyData<F>)> {
        let (likelihood, w_data) = match self.method() {
            SparseMethod::Fitc => {
                self.fitc(theta, sigma2, noise, w_star, xtrain, ytrain, z, nugget)
            }
            SparseMethod::Vfe => self.vfe(theta, sigma2, noise, w_star, xtrain, ytrain, z, nugget),
        };

        Ok((likelihood, w_data))
    }

    /// Compute covariance matrix between a and b matrices
    fn compute_k(
        &self,
        a: &ArrayBase<impl Data<Elem = F>, Ix2>,
        b: &ArrayBase<impl Data<Elem = F>, Ix2>,
        w_star: &Array2<F>,
        theta: &Array1<F>,
        sigma2: F,
    ) -> Array2<F> {
        // Get pairwise componentwise L1-distances to the input training set
        let dx = pairwise_differences(a, b);
        // Compute the correlation function
        let r = self.corr().value(&dx, theta, w_star);
        r.into_shape((a.nrows(), b.nrows()))
            .unwrap()
            .mapv(|v| v * sigma2)
    }

    /// FITC method
    #[allow(clippy::too_many_arguments)]
    fn fitc(
        &self,
        theta: &Array1<F>,
        sigma2: F,
        noise: F,
        w_star: &Array2<F>,
        xtrain: &ArrayView2<F>,
        ytrain: &ArrayView2<F>,
        z: &Array2<F>,
        nugget: F,
    ) -> (F, WoodburyData<F>) {
        let nz = z.nrows();
        let knn = Array1::from_elem(xtrain.nrows(), sigma2);
        let kmm = self.compute_k(z, z, w_star, theta, sigma2) + Array::eye(nz) * nugget;
        let kmn = self.compute_k(z, xtrain, w_star, theta, sigma2);

        // Compute (lower) Cholesky decomposition: Kmm = U U^T
        let u = kmm.cholesky().unwrap();

        // Compute cholesky decomposition: Qnn = V^T V
        let ui = u
            .solve_triangular(&Array::eye(u.nrows()), UPLO::Lower)
            .unwrap();
        let v = ui.dot(&kmn);

        // Assumption on the gaussian noise on training outputs
        let eta2 = noise;

        // Compute diagonal correction: nu = Knn_diag - Qnn_diag + \eta^2
        let nu = knn;
        let nu = nu - (v.to_owned() * &v).sum_axis(Axis(0));
        let nu = nu + eta2;
        // Compute beta, the effective noise precision
        let beta = nu.mapv(|v| F::one() / v);

        // Compute (lower) Cholesky decomposition: A = I + V diag(beta) V^T = L L^T
        let a = Array::eye(nz) + &(v.to_owned() * beta.to_owned().insert_axis(Axis(0))).dot(&v.t());

        let l = a.cholesky().unwrap();
        let li = l
            .solve_triangular(&Array::eye(l.nrows()), UPLO::Lower)
            .unwrap();

        // Compute a and b
        let a = einsum("ij,i->ij", &[ytrain, &beta])
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        let tmp = li.dot(&v);
        let b = tmp.dot(&a);

        // Compute marginal log-likelihood
        // constant term ignored in reduced likelihood
        //let term0 = self.ytrain.nrows() * F::cast(2. * std::f64::consts::PI);
        let term1 = nu.mapv(|v| v.ln()).sum();
        let term2 = F::cast(2.) * l.diag().mapv(|v| v.ln()).sum();
        let term3 = (a.t().to_owned()).dot(ytrain)[[0, 0]];
        //let term4 = einsum("ij,ij->", &[&b, &b]).unwrap();
        let term4 = -(b.to_owned() * &b).sum();
        let likelihood = -F::cast(0.5) * (term1 + term2 + term3 + term4);

        // Store Woodbury vectors for prediction step
        let li_ui = li.dot(&ui);
        let li_ui_t = li_ui.t();
        let w_data = WoodburyData {
            vec: li_ui_t.dot(&b),
            inv: (ui.t()).dot(&ui) - li_ui_t.dot(&li_ui),
        };

        (likelihood, w_data)
    }

    /// VFE method
    #[allow(clippy::too_many_arguments)]
    fn vfe(
        &self,
        theta: &Array1<F>,
        sigma2: F,
        noise: F,
        w_star: &Array2<F>,
        xtrain: &ArrayView2<F>,
        ytrain: &ArrayView2<F>,
        z: &Array2<F>,
        nugget: F,
    ) -> (F, WoodburyData<F>) {
        // Compute: Kmm and Kmn
        let nz = z.nrows();
        let kmm = self.compute_k(z, z, w_star, theta, sigma2) + Array::eye(nz) * nugget;
        let kmn = self.compute_k(z, xtrain, w_star, theta, sigma2);

        // Compute cholesky decomposition: Kmm = U U^T
        let u = kmm.cholesky().unwrap();

        // Compute cholesky decomposition: Qnn = V^T V
        let ui = u
            .solve_triangular(&Array::eye(u.nrows()), UPLO::Lower)
            .unwrap();
        let v = ui.dot(&kmn);

        // Compute beta, the effective noise precision
        let beta = F::one() / noise.max(nugget);

        // Compute A = beta * V @ V.T
        let a = v.to_owned().dot(&v.t()).mapv(|v| v * beta);

        // Compute cholesky decomposition: B = I + A = L L^T
        let b: Array2<F> = Array::eye(nz) + &a;
        let l = b.cholesky().unwrap();
        let li = l
            .solve_triangular(&Array::eye(l.nrows()), UPLO::Lower)
            .unwrap();

        // Compute b
        let b = li.dot(&v).dot(ytrain).mapv(|v| v * beta);

        // Compute log-marginal likelihood
        // constant term ignored in reduced likelihood
        //let term0 = self.ytrain.nrows() * (F::cast(2. * std::f64::consts::PI)
        let term1 = -F::cast(ytrain.nrows()) * beta.ln();
        let term2 = F::cast(2.) * l.diag().mapv(|v| v.ln()).sum();
        let term3 = beta * (ytrain.to_owned() * ytrain).sum();
        let term4 = -b.t().dot(&b)[[0, 0]];
        let term5 = F::cast(ytrain.nrows()) * beta * sigma2;
        let term6 = -a.diag().sum();

        let likelihood = -F::cast(0.5) * (term1 + term2 + term3 + term4 + term5 + term6);

        let li_ui = li.dot(&ui);
        let bi = Array::eye(nz) + li.t().dot(&li);
        let w_data = WoodburyData {
            vec: li_ui.t().dot(&b),
            inv: ui.t().dot(&bi).dot(&ui),
        };

        (likelihood, w_data)
    }
}

fn make_inducings<F: Float>(
    n_inducing: usize,
    xt: &ArrayView2<F>,
    rng: &mut Xoshiro256Plus,
) -> Array2<F> {
    let mut indices = (0..xt.nrows()).collect::<Vec<_>>();
    indices.shuffle(rng);
    let n = n_inducing.min(xt.nrows());
    let mut z = Array2::zeros((n, xt.ncols()));
    let idx = indices[..n].to_vec();
    Zip::from(z.rows_mut())
        .and(&Array1::from_vec(idx))
        .for_each(|mut zi, i| zi.assign(&xt.row(*i)));
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use ndarray::Array;
    // use ndarray_npy::{read_npy, write_npy};
    use ndarray_npy::write_npy;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::{Normal, Uniform};
    use ndarray_rand::RandomExt;
    use rand_xoshiro::Xoshiro256Plus;

    const PI: f64 = std::f64::consts::PI;

    fn f_obj(x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        x.mapv(|v| (3. * PI * v).sin() + 0.3 * (9. * PI * v).cos() + 0.5 * (7. * PI * v).sin())
    }

    fn make_test_data(
        nt: usize,
        eta2: f64,
        rng: &mut Xoshiro256Plus,
    ) -> (Array2<f64>, Array2<f64>) {
        let normal = Normal::new(0., eta2.sqrt()).unwrap();
        let gaussian_noise = Array::<f64, _>::random_using((nt, 1), normal, rng);
        let xt = 2. * Array::<f64, _>::random_using((nt, 1), Uniform::new(0., 1.), rng) - 1.;
        let yt = f_obj(&xt) + gaussian_noise;
        (xt, yt)
    }

    fn save_data(
        xt: &Array2<f64>,
        yt: &Array2<f64>,
        z: &Array2<f64>,
        xplot: &Array2<f64>,
        sgp_vals: &Array2<f64>,
        sgp_vars: &Array2<f64>,
    ) {
        let test_dir = "target/tests";
        std::fs::create_dir_all(test_dir).ok();

        let file_path = format!("{}/sgp_xt.npy", test_dir);
        write_npy(file_path, xt).expect("xt saved");
        let file_path = format!("{}/sgp_yt.npy", test_dir);
        write_npy(file_path, yt).expect("yt saved");
        let file_path = format!("{}/sgp_z.npy", test_dir);
        write_npy(file_path, z).expect("z saved");
        let file_path = format!("{}/sgp_x.npy", test_dir);
        write_npy(file_path, xplot).expect("x saved");
        let file_path = format!("{}/sgp_vals.npy", test_dir);
        write_npy(file_path, sgp_vals).expect("sgp vals saved");
        let file_path = format!("{}/sgp_vars.npy", test_dir);
        write_npy(file_path, sgp_vars).expect("sgp vars saved");
    }

    #[test]
    fn test_sgp_default() {
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        // Generate training data
        let nt = 200;
        // Variance of the gaussian noise on our training data
        let eta2: f64 = 0.01;
        let (xt, yt) = make_test_data(nt, eta2, &mut rng);
        // let test_dir = "target/tests";
        // let file_path = format!("{}/smt_xt.npy", test_dir);
        // let xt: Array2<f64> = read_npy(file_path).expect("xt read");
        // let file_path = format!("{}/smt_yt.npy", test_dir);
        // let yt: Array2<f64> = read_npy(file_path).expect("yt read");

        let xplot = Array::linspace(-1.0, 1.0, 100).insert_axis(Axis(1));
        let n_inducings = 30;

        // let file_path = format!("{}/smt_z.npy", test_dir);
        // let z: Array2<f64> = read_npy(file_path).expect("z read");
        // println!("{:?}", z);

        let sgp = SparseKriging::params(Inducings::Randomized(n_inducings))
            //let sgp = SparseKriging::params(Inducings::Located(z))
            //.noise_variance(ParamEstimation::Constant(0.01))
            .seed(Some(42))
            .fit(&Dataset::new(xt.clone(), yt.clone()))
            .expect("GP fitted");

        println!("theta={:?}", sgp.theta());
        println!("variance={:?}", sgp.variance());
        println!("noise variance={:?}", sgp.noise_variance());
        // assert_abs_diff_eq!(eta2, sgp.noise_variance());

        let sgp_vals = sgp.predict_values(&xplot).unwrap();
        let yplot = f_obj(&xplot);
        let errvals = (yplot - &sgp_vals).mapv(|v| v.abs());
        assert_abs_diff_eq!(errvals, Array2::zeros((xplot.nrows(), 1)), epsilon = 0.5);
        let sgp_vars = sgp.predict_variances(&xplot).unwrap();
        let errvars = (&sgp_vars - Array2::from_elem((xplot.nrows(), 1), 0.01)).mapv(|v| v.abs());
        assert_abs_diff_eq!(errvars, Array2::zeros((xplot.nrows(), 1)), epsilon = 0.2);

        save_data(&xt, &yt, sgp.inducings(), &xplot, &sgp_vals, &sgp_vars);
    }

    #[test]
    fn test_sgp_vfe() {
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        // Generate training data
        let nt = 200;
        // Variance of the gaussian noise on our training data
        let eta2: f64 = 0.01;
        let (xt, yt) = make_test_data(nt, eta2, &mut rng);
        // let test_dir = "target/tests";
        // let file_path = format!("{}/smt_xt.npy", test_dir);
        // let xt: Array2<f64> = read_npy(file_path).expect("xt read");
        // let file_path = format!("{}/smt_yt.npy", test_dir);
        // let yt: Array2<f64> = read_npy(file_path).expect("yt read");

        let xplot = Array::linspace(-1., 1., 100).insert_axis(Axis(1));
        let n_inducings = 30;

        let z = make_inducings(n_inducings, &xt.view(), &mut rng);
        // let file_path = format!("{}/smt_z.npy", test_dir);
        // let z: Array2<f64> = read_npy(file_path).expect("z read");

        let sgp = SparseGaussianProcess::<f64, SquaredExponentialCorr>::params(
            SquaredExponentialCorr::default(),
            Inducings::Located(z),
        )
        .sparse_method(SparseMethod::Vfe)
        .noise_variance(ParamEstimation::Fixed(0.01))
        .fit(&Dataset::new(xt.clone(), yt.clone()))
        .expect("GP fitted");

        println!("theta={:?}", sgp.theta());
        println!("variance={:?}", sgp.variance());
        println!("noise variance={:?}", sgp.noise_variance());
        assert_abs_diff_eq!(eta2, sgp.noise_variance());

        let sgp_vals = sgp.predict_values(&xplot).unwrap();
        let sgp_vars = sgp.predict_variances(&xplot).unwrap();

        save_data(&xt, &yt, sgp.inducings(), &xplot, &sgp_vals, &sgp_vars);
    }

    #[test]
    fn test_sgp_noise_estimation() {
        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        // Generate training data
        let nt = 200;
        // Variance of the gaussian noise on our training data
        let eta2: f64 = 0.01;
        let (xt, yt) = make_test_data(nt, eta2, &mut rng);
        // let test_dir = "target/tests";
        // let file_path = format!("{}/smt_xt.npy", test_dir);
        // let xt: Array2<f64> = read_npy(file_path).expect("xt read");
        // let file_path = format!("{}/smt_yt.npy", test_dir);
        // let yt: Array2<f64> = read_npy(file_path).expect("yt read");

        let xplot = Array::linspace(-1., 1., 100).insert_axis(Axis(1));
        let n_inducings = 30;

        let z = make_inducings(n_inducings, &xt.view(), &mut rng);
        // let file_path = format!("{}/smt_z.npy", test_dir);
        // let z: Array2<f64> = read_npy(file_path).expect("z read");

        let sgp = SparseGaussianProcess::<f64, SquaredExponentialCorr>::params(
            SquaredExponentialCorr::default(),
            Inducings::Located(z.clone()),
        )
        .sparse_method(SparseMethod::Vfe)
        //.sparse_method(SparseMethod::Fitc)
        .theta_init(vec![0.1])
        .noise_variance(ParamEstimation::Estimated {
            initial_guess: 0.02,
            bounds: (1e-3, 1.),
        })
        .fit(&Dataset::new(xt.clone(), yt.clone()))
        .expect("SGP fitted");

        println!("theta={:?}", sgp.theta());
        println!("variance={:?}", sgp.variance());
        println!("noise variance={:?}", sgp.noise_variance());
        assert_abs_diff_eq!(eta2, sgp.noise_variance(), epsilon = 0.015);
        assert_abs_diff_eq!(&z, sgp.inducings(), epsilon = 0.0015);

        let sgp_vals = sgp.predict_values(&xplot).unwrap();
        let sgp_vars = sgp.predict_variances(&xplot).unwrap();

        save_data(&xt, &yt, &z, &xplot, &sgp_vals, &sgp_vars);
    }
}
