//! `egobox`, Rust toolbox for efficient global optimization
//!
//! Thanks to the [PyO3 project](https://pyo3.rs), which makes Rust well suited for building Python extensions,
//! the mixture of gaussian process surrogates is binded in Python. You can install the Python package using:
//!
//! ```bash
//! pip install egobox
//! ```
//!
//! See the [tutorial notebook](https://github.com/relf/egobox/doc/Sgp_Tutorial.ipynb) for usage.
//!
use crate::types::*;
use egobox_gp::{Inducings, ParamTuning, ThetaTuning};
use egobox_moe::{GpSurrogate, SparseGpMixture};
use linfa::{traits::Fit, Dataset};
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rand_xoshiro::Xoshiro256Plus;

/// Sparse Gaussian processes mixture builder
///
///     n_clusters (int >= 0)
///         Number of clusters used by the mixture of surrogate experts.
///         When set to 0, the number of cluster is determined automatically and refreshed every
///         10-points addition (should say 'tentative addition' because addition may fail for some points
///         but failures are counted anyway).
///
///     corr_spec (CorrelationSpec flags, an int in [1, 15]):
///         Specification of correlation models used in mixture.
///         Can be CorrelationSpec.SQUARED_EXPONENTIAL (1), CorrelationSpec.ABSOLUTE_EXPONENTIAL (2),
///         CorrelationSpec.MATERN32 (4), CorrelationSpec.MATERN52 (8) or
///         any bit-wise union of these values (e.g. CorrelationSpec.MATERN32 | CorrelationSpec.MATERN52)
///
///     recombination (Recombination.Smooth or Recombination.Hard)
///         Specify how the various experts predictions are recombined
///         * Smooth: prediction is a combination of experts prediction wrt their responsabilities,
///         the heaviside factor which controls steepness of the change between experts regions is optimized
///         to get best mixture quality.
///         * Hard: prediction is taken from the expert with highest responsability
///         resulting in a model with discontinuities.
///
///     kpls_dim (0 < int < nx where nx is the dimension of inputs x)
///         Number of components to be used when PLS projection is used (a.k.a KPLS method).
///         This is used to address high-dimensional problems typically when nx > 9.
///
///     n_start (int >= 0)
///         Number of internal GP hyperpameters optimization restart (multistart)
///
///     method (SparseMethod.FITC or SparseMethod.VFE)
///         Sparse method to be used (default is FITC)
///
///     seed (int >= 0)
///         Random generator seed to allow computation reproducibility.
///         
#[pyclass]
pub(crate) struct SparseGpMix {
    pub correlation_spec: CorrelationSpec,
    pub theta_init: Option<Vec<f64>>,
    pub theta_bounds: Option<Vec<Vec<f64>>>,
    pub kpls_dim: Option<usize>,
    pub n_start: usize,
    pub nz: Option<usize>,
    pub z: Option<Array2<f64>>,
    pub method: SparseMethod,
    pub seed: Option<u64>,
}

#[pymethods]
impl SparseGpMix {
    #[new]
    #[pyo3(signature = (
        corr_spec = CorrelationSpec::SQUARED_EXPONENTIAL,
        theta_init = None,
        theta_bounds = None,
        kpls_dim = None,
        n_start = 10,
        nz = None,
        z = None,
        method = SparseMethod::Fitc,
        seed = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        corr_spec: u8,
        theta_init: Option<Vec<f64>>,
        theta_bounds: Option<Vec<Vec<f64>>>,
        kpls_dim: Option<usize>,
        n_start: usize,
        nz: Option<usize>,
        z: Option<PyReadonlyArray2<f64>>,
        method: SparseMethod,
        seed: Option<u64>,
    ) -> Self {
        SparseGpMix {
            correlation_spec: CorrelationSpec(corr_spec),
            theta_init,
            theta_bounds,
            kpls_dim,
            n_start,
            nz,
            z: z.map(|z| z.as_array().to_owned()),
            method,
            seed,
        }
    }

    /// Fit the parameters of the model using the training dataset to build a trained model
    ///
    /// Parameters
    ///     xt (array[nsamples, nx]): input samples
    ///     yt (array[nsamples, 1]): output samples
    ///
    /// Returns Sgp object
    ///     the fitted Gaussian process mixture  
    ///
    fn fit(
        &mut self,
        py: Python,
        xt: PyReadonlyArray2<f64>,
        yt: PyReadonlyArray2<f64>,
    ) -> SparseGpx {
        let dataset = Dataset::new(xt.as_array().to_owned(), yt.as_array().to_owned());

        let rng = if let Some(seed) = self.seed {
            Xoshiro256Plus::seed_from_u64(seed)
        } else {
            Xoshiro256Plus::from_entropy()
        };

        let inducings = if let Some(z) = self.z.as_ref() {
            Inducings::Located(z.clone())
        } else if let Some(nz) = self.nz {
            Inducings::Randomized(nz)
        } else {
            panic!("You must specify inducing points")
        };

        let method = match self.method {
            SparseMethod::Fitc => egobox_gp::SparseMethod::Fitc,
            SparseMethod::Vfe => egobox_gp::SparseMethod::Vfe,
        };

        let mut theta_tuning = ThetaTuning::default();
        if let Some(init) = self.theta_init.as_ref() {
            theta_tuning = ParamTuning {
                init: init.to_vec(),
                ..theta_tuning.into()
            }
            .try_into()
            .expect("Theta tuning initial init");
        }
        if let Some(bounds) = self.theta_bounds.as_ref() {
            theta_tuning = ParamTuning {
                bounds: bounds.iter().map(|v| (v[0], v[1])).collect(),
                ..theta_tuning.into()
            }
            .try_into()
            .expect("Theta tuning bounds");
        }

        if let Err(ctrlc::Error::MultipleHandlers) = ctrlc::set_handler(|| std::process::exit(2)) {
            // ignore multiple handlers error
        };
        let sgp = py.allow_threads(|| {
            SparseGpMixture::params(inducings)
                .correlation_spec(
                    egobox_moe::CorrelationSpec::from_bits(self.correlation_spec.0).unwrap(),
                )
                .theta_tuning(theta_tuning)
                .kpls_dim(self.kpls_dim)
                .n_start(self.n_start)
                .sparse_method(method)
                .with_rng(rng)
                .fit(&dataset)
                .expect("Sgp model training")
        });
        SparseGpx(Box::new(sgp))
    }
}

/// A trained Gaussian processes mixture
#[pyclass]
pub(crate) struct SparseGpx(Box<SparseGpMixture>);

#[pymethods]
impl SparseGpx {
    /// Get Gaussian processes mixture builder aka `GpSparse`
    ///
    /// See `GpSparse` constructor
    #[staticmethod]
    #[pyo3(signature = (
        corr_spec = CorrelationSpec::SQUARED_EXPONENTIAL,
        theta_init = None,
        theta_bounds = None,
        kpls_dim = None,
        n_start = 10,
        nz = None,
        z = None,
        method = SparseMethod::Fitc,
        seed = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn builder(
        corr_spec: u8,
        theta_init: Option<Vec<f64>>,
        theta_bounds: Option<Vec<Vec<f64>>>,
        kpls_dim: Option<usize>,
        n_start: usize,
        nz: Option<usize>,
        z: Option<PyReadonlyArray2<f64>>,
        method: SparseMethod,
        seed: Option<u64>,
    ) -> SparseGpMix {
        SparseGpMix::new(
            corr_spec,
            theta_init,
            theta_bounds,
            kpls_dim,
            n_start,
            nz,
            z,
            method,
            seed,
        )
    }

    /// Returns the String representation from serde json serializer
    fn __repr__(&self) -> String {
        serde_json::to_string(&self.0).unwrap()
    }

    /// Returns a String informal representation
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    /// Save Gaussian processes mixture in a json file.
    ///
    /// Parameters
    ///     filename (string)
    ///         json file generated in the current directory
    ///
    fn save(&self, filename: String) {
        self.0.save(&filename).ok();
    }

    /// Load Gaussian processes mixture from a json file.
    ///
    /// Parameters
    ///     filename (string)
    ///         json filepath generated by saving a trained Gaussian processes mixture
    ///
    #[staticmethod]
    fn load(filename: String) -> SparseGpx {
        SparseGpx(SparseGpMixture::load(&filename).unwrap())
    }

    /// Predict output values at nsamples points.
    ///
    /// Parameters
    ///     x (array[nsamples, nx])
    ///         input values
    ///
    /// Returns
    ///     the output values at nsamples x points (array[nsamples, 1])
    ///
    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> &'py PyArray2<f64> {
        self.0
            .predict(&x.as_array().to_owned())
            .unwrap()
            .into_pyarray(py)
    }

    /// Predict variances at nsample points.
    ///
    /// Parameters
    ///     x (array[nsamples, nx])
    ///         input values
    ///
    /// Returns
    ///     the variances of the output values at nsamples input points (array[nsamples, 1])
    ///
    fn predict_var<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> &'py PyArray2<f64> {
        self.0
            .predict_var(&x.as_array().to_owned())
            .unwrap()
            .into_pyarray(py)
    }
}
