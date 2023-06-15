//! `egobox`, Rust toolbox for efficient global optimization
//!
//! Thanks to the [PyO3 project](https://pyo3.rs), which makes Rust well suited for building Python extensions,
//! the mixture of gaussian process surrogates is binded in Python. You can install the Python package using:
//!
//! ```bash
//! pip install egobox
//! ```
//!
//! See the [tutorial notebook](https://github.com/relf/egobox/doc/Gpx_Tutorial.ipynb) for usage.
//!
use crate::types::*;
use egobox_moe::{Moe, Surrogate};
use linfa::{traits::Fit, Dataset};
use ndarray_rand::rand::SeedableRng;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rand_xoshiro::Xoshiro256Plus;

/// Gaussian processes mixture builder
///
///     n_clusters (int >= 0)
///         Number of clusters used by the mixture of surrogate experts.
///         When set to 0, the number of cluster is determined automatically and refreshed every
///         10-points addition (should say 'tentative addition' because addition may fail for some points
///         but failures are counted anyway).
///
///     regr_spec (RegressionSpec flags, an int in [1, 7]):
///         Specification of regression models used in mixture.
///         Can be RegressionSpec.CONSTANT (1), RegressionSpec.LINEAR (2), RegressionSpec.QUADRATIC (4) or
///         any bit-wise union of these values (e.g. RegressionSpec.CONSTANT | RegressionSpec.LINEAR)
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
///         an optional heaviside factor might be used control steepness of the change between experts regions.
///         * Hard: prediction is taken from the expert with highest responsability
///         resulting in a model with discontinuities.
///
///     kpls_dim (0 < int < nx where nx is the dimension of inputs x)
///         Number of components to be used when PLS projection is used (a.k.a KPLS method).
///         This is used to address high-dimensional problems typically when nx > 9.
///
///     seed (int >= 0)
///         Random generator seed to allow computation reproducibility.
///         
#[pyclass]
#[pyo3(text_signature = "()")]
pub(crate) struct GpMix {
    pub n_clusters: usize,
    pub regression_spec: RegressionSpec,
    pub correlation_spec: CorrelationSpec,
    pub recombination: Recombination,
    pub kpls_dim: Option<usize>,
    pub seed: Option<u64>,
    pub training_data: Option<Dataset<f64, f64>>,
}

#[pymethods]
impl GpMix {
    #[new]
    #[pyo3(signature = (
        n_clusters = 1,
        regr_spec = RegressionSpec::CONSTANT,
        corr_spec = CorrelationSpec::SQUARED_EXPONENTIAL,
        recombination = Recombination::Smooth,
        kpls_dim = None,
        seed = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_clusters: usize,
        regr_spec: u8,
        corr_spec: u8,
        recombination: Recombination,
        kpls_dim: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        GpMix {
            n_clusters,
            regression_spec: RegressionSpec(regr_spec),
            correlation_spec: CorrelationSpec(corr_spec),
            recombination,
            kpls_dim,
            seed,
            training_data: None,
        }
    }

    /// Set the training dataset (xt, yt) used to train the Gaussian process mixture.
    ///
    /// Parameters
    ///     xt (array[nsamples, nx]): input samples
    ///     yt (array[nsamples, 1]): output samples
    ///   
    fn set_training_values(&mut self, xt: PyReadonlyArray2<f64>, yt: PyReadonlyArray2<f64>) {
        self.training_data = Some(Dataset::new(
            xt.as_array().to_owned(),
            yt.as_array().to_owned(),
        ));
    }

    /// Fit the parameters of the model using the training dataset to build a trained model
    ///
    /// Returns Gpx object
    ///     the fitted Gaussian process mixture  
    ///
    fn train(&mut self) -> Gpx {
        let recomb = match self.recombination {
            Recombination::Hard => egobox_moe::Recombination::Hard,
            Recombination::Smooth => egobox_moe::Recombination::Smooth(None),
        };
        let rng = if let Some(seed) = self.seed {
            Xoshiro256Plus::seed_from_u64(seed)
        } else {
            Xoshiro256Plus::from_entropy()
        };
        let moe = Moe::params()
            .n_clusters(self.n_clusters)
            .recombination(recomb)
            .regression_spec(egobox_moe::RegressionSpec::from_bits(self.regression_spec.0).unwrap())
            .correlation_spec(
                egobox_moe::CorrelationSpec::from_bits(self.correlation_spec.0).unwrap(),
            )
            .kpls_dim(self.kpls_dim)
            .with_rng(rng)
            .fit(self.training_data.as_ref().unwrap())
            .expect("MoE model training");
        Gpx(Box::new(moe))
    }
}

/// A trained Gaussian processes mixture
#[pyclass]
#[pyo3(text_signature = "()")]
pub(crate) struct Gpx(Box<Moe>);

#[pymethods]
impl Gpx {
    /// Get Gaussian processes mixture builder aka `GpMix`
    ///
    /// See `GpMix` constructor
    #[staticmethod]
    #[pyo3(signature = (
        n_clusters = 1,
        regr_spec = RegressionSpec::CONSTANT,
        corr_spec = CorrelationSpec::SQUARED_EXPONENTIAL,
        recombination = Recombination::Smooth,
        kpls_dim = None,
        seed = None
    ))]
    fn builder(
        n_clusters: usize,
        regr_spec: u8,
        corr_spec: u8,
        recombination: Recombination,
        kpls_dim: Option<usize>,
        seed: Option<u64>,
    ) -> GpMix {
        GpMix::new(
            n_clusters,
            regr_spec,
            corr_spec,
            recombination,
            kpls_dim,
            seed,
        )
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
    fn load(filename: String) -> Gpx {
        Gpx(Moe::load(&filename).unwrap())
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
    fn predict_values<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> &'py PyArray2<f64> {
        self.0
            .predict_values(&x.as_array().to_owned())
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
    fn predict_variances<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        self.0
            .predict_variances(&x.as_array().to_owned())
            .unwrap()
            .into_pyarray(py)
    }

    /// Sample gaussian process trajectories.
    ///
    /// Parameters
    ///     x (array[nsamples, nx])
    ///         locations of the sampled trajectories
    ///     n_traj number of trajectories to generate
    ///
    /// Returns
    ///     the trajectories as an array[nsamples, n_traj]
    ///
    fn sample<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        n_traj: usize,
    ) -> &'py PyArray2<f64> {
        self.0
            .sample(&x.as_array().to_owned(), n_traj)
            .unwrap()
            .into_pyarray(py)
    }
}
