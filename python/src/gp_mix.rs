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
use std::{cmp::Ordering, path::Path};

use crate::types::*;
use egobox_gp::metrics::CrossValScore;
use egobox_moe::{Clustered, MixtureGpSurrogate, NbClusters, ThetaTuning};
#[allow(unused_imports)] // Avoid linting problem
use egobox_moe::{GpMixture, GpSurrogate, GpSurrogateExt};
use linfa::{traits::Fit, Dataset};
use log::error;
use ndarray::{array, Array1, Array2, Axis, Ix1, Ix2, Zip};
use ndarray_rand::rand::SeedableRng;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rand_xoshiro::Xoshiro256Plus;

/// Gaussian processes mixture builder
///
///     n_clusters (int)
///         Number of clusters used by the mixture of surrogate experts (default is 1).
///         When set to 0, the number of cluster is determined automatically and refreshed every
///         10-points addition (should say 'tentative addition' because addition may fail for some points
///         but it is counted anyway).
///         When set to negative number -n, the number of clusters is determined automatically in [1, n]
///         this is used to limit the number of trials hence the execution time.
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
///     recombination (Recombination.Smooth or Recombination.Hard (default))
///         Specify how the various experts predictions are recombined
///         * Smooth: prediction is a combination of experts prediction wrt their responsabilities,
///         the heaviside factor which controls steepness of the change between experts regions is optimized
///         to get best mixture quality.
///         * Hard: prediction is taken from the expert with highest responsability
///         resulting in a model with discontinuities.
///
///     theta_init ([nx] where nx is the dimension of inputs x)
///         Initial guess for GP theta hyperparameters.
///         When None the default is 1e-2 for all components
///
///     theta_bounds ([[lower_1, upper_1], ..., [lower_nx, upper_nx]] where nx is the dimension of inputs x)
///         Space search when optimizing theta GP hyperparameters
///         When None the default is [1e-6, 1e2] for all components
///
///     kpls_dim (0 < int < nx where nx is the dimension of inputs x)
///         Number of components to be used when PLS projection is used (a.k.a KPLS method).
///         This is used to address high-dimensional problems typically when nx > 9.
///
///     n_start (int >= 0)
///         Number of internal GP hyperpameters optimization restart (multistart)
///
///     seed (int >= 0)
///         Random generator seed to allow computation reproducibility.
///         
#[pyclass]
pub(crate) struct GpMix {
    pub n_clusters: NbClusters,
    pub regression_spec: RegressionSpec,
    pub correlation_spec: CorrelationSpec,
    pub recombination: Recombination,
    pub theta_init: Option<Vec<f64>>,
    pub theta_bounds: Option<Vec<Vec<f64>>>,
    pub kpls_dim: Option<usize>,
    pub n_start: usize,
    pub seed: Option<u64>,
}

#[pymethods]
impl GpMix {
    #[new]
    #[pyo3(signature = (
        n_clusters = 1,
        regr_spec = RegressionSpec::CONSTANT,
        corr_spec = CorrelationSpec::SQUARED_EXPONENTIAL,
        recombination = Recombination::Hard,
        theta_init = None,
        theta_bounds = None,
        kpls_dim = None,
        n_start = 10,
        seed = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_clusters: isize,
        regr_spec: u8,
        corr_spec: u8,
        recombination: Recombination,
        theta_init: Option<Vec<f64>>,
        theta_bounds: Option<Vec<Vec<f64>>>,
        kpls_dim: Option<usize>,
        n_start: usize,
        seed: Option<u64>,
    ) -> Self {
        let n_clusters = match n_clusters.cmp(&0) {
            Ordering::Greater => NbClusters::fixed(n_clusters as usize),
            Ordering::Equal => NbClusters::auto(),
            Ordering::Less => NbClusters::automax(-n_clusters as usize),
        };

        GpMix {
            n_clusters,
            regression_spec: RegressionSpec(regr_spec),
            correlation_spec: CorrelationSpec(corr_spec),
            recombination,
            theta_init,
            theta_bounds,
            kpls_dim,
            n_start,
            seed,
        }
    }

    /// Fit the parameters of the model using the training dataset to build a trained model
    ///
    /// Parameters
    ///     xt (array[nsamples, nx]): input samples
    ///     yt (array[nsamples, 1]): output samples
    ///
    /// Returns Gpx object
    ///     the fitted Gaussian process mixture  
    ///
    fn fit(&mut self, py: Python, xt: PyReadonlyArrayDyn<f64>, yt: PyReadonlyArrayDyn<f64>) -> Gpx {
        let xt = xt.as_array();
        let xt = match xt.to_owned().into_dimensionality::<Ix2>() {
            Ok(xt) => xt,
            Err(_) => match xt.into_dimensionality::<Ix1>() {
                Ok(xt) => xt.insert_axis(Axis(1)).to_owned(),
                _ => {
                    error!("Training input has to be an [nsamples, nx] array");
                    panic!("Bad training input data");
                }
            },
        };

        let yt = yt.as_array();
        let yt = match yt.to_owned().into_dimensionality::<Ix1>() {
            Ok(yt) => yt,
            Err(_) => match yt.into_dimensionality::<Ix2>() {
                Ok(yt) => {
                    if yt.dim().1 == 1 {
                        yt.to_owned().remove_axis(Axis(1))
                    } else {
                        error!("Training output has to be one dimensional");
                        panic!("Bad training output data");
                    }
                }
                Err(_) => {
                    error!("Training output has to be one dimensional");
                    panic!("Bad training output data");
                }
            },
        };
        let dataset = Dataset::new(xt, yt);

        let recomb = match self.recombination {
            Recombination::Hard => egobox_moe::Recombination::Hard,
            Recombination::Smooth => egobox_moe::Recombination::Smooth(None),
        };
        let rng = if let Some(seed) = self.seed {
            Xoshiro256Plus::seed_from_u64(seed)
        } else {
            Xoshiro256Plus::from_entropy()
        };

        let mut theta_tuning = ThetaTuning::default();
        if let Some(init) = self.theta_init.as_ref() {
            theta_tuning = ThetaTuning::Full {
                init: Array1::from_vec(init.to_vec()),
                bounds: array![ThetaTuning::<f64>::DEFAULT_BOUNDS],
            }
        }
        if let Some(bounds) = self.theta_bounds.as_ref() {
            theta_tuning = ThetaTuning::Full {
                init: theta_tuning.init().to_owned(),
                bounds: bounds.iter().map(|v| (v[0], v[1])).collect(),
            }
        }
        let theta_tunings = if let NbClusters::Fixed { nb } = self.n_clusters {
            vec![theta_tuning; nb]
        } else {
            vec![theta_tuning; 1] // used as default theta tuning for all experts
        };

        if let Err(ctrlc::Error::MultipleHandlers) = ctrlc::set_handler(|| std::process::exit(2)) {
            // ignore multiple handlers error
        };
        let moe = py.allow_threads(|| {
            GpMixture::params()
                .n_clusters(self.n_clusters.clone())
                .recombination(recomb)
                .regression_spec(
                    egobox_moe::RegressionSpec::from_bits(self.regression_spec.0).unwrap(),
                )
                .correlation_spec(
                    egobox_moe::CorrelationSpec::from_bits(self.correlation_spec.0).unwrap(),
                )
                .theta_tunings(&theta_tunings)
                .kpls_dim(self.kpls_dim)
                .n_start(self.n_start)
                .with_rng(rng)
                .fit(&dataset)
                .expect("MoE model training")
        });

        Gpx(Box::new(moe))
    }
}

/// A trained Gaussian processes mixture
#[pyclass]
pub(crate) struct Gpx(Box<GpMixture>);

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
        theta_init = None,
        theta_bounds = None,
        kpls_dim = None,
        n_start = 10,
        seed = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn builder(
        n_clusters: isize,
        regr_spec: u8,
        corr_spec: u8,
        recombination: Recombination,
        theta_init: Option<Vec<f64>>,
        theta_bounds: Option<Vec<Vec<f64>>>,
        kpls_dim: Option<usize>,
        n_start: usize,
        seed: Option<u64>,
    ) -> GpMix {
        GpMix::new(
            n_clusters,
            regr_spec,
            corr_spec,
            recombination,
            theta_init,
            theta_bounds,
            kpls_dim,
            n_start,
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

    /// Save Gaussian processes mixture in a file.
    /// If the filename has .json JSON human readable format is used
    /// otherwise an optimized binary format is used.
    ///
    /// Parameters
    ///     filename with .json or .bin extension (string)
    ///         file generated in the current directory
    ///
    /// Returns True if save succeeds otherwise False
    ///
    fn save(&self, filename: String) -> bool {
        let format = match Path::new(&filename).extension().unwrap().to_str().unwrap() {
            "json" => egobox_moe::GpFileFormat::Json,
            _ => egobox_moe::GpFileFormat::Binary,
        };
        self.0.save(&filename, format).is_ok()
    }

    /// Load Gaussian processes mixture from file.
    ///
    /// Parameters
    ///     filename (string)
    ///         json filepath generated by saving a trained Gaussian processes mixture
    ///
    #[staticmethod]
    fn load(filename: String) -> Gpx {
        let format = match Path::new(&filename).extension().unwrap().to_str().unwrap() {
            "json" => egobox_moe::GpFileFormat::Json,
            _ => egobox_moe::GpFileFormat::Binary,
        };
        Gpx(GpMixture::load(&filename, format).unwrap())
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
    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> Bound<'py, PyArray2<f64>> {
        self.0
            .predict(&x.as_array())
            .unwrap()
            .insert_axis(Axis(1))
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
    fn predict_var<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        self.0.predict_var(&x.as_array()).unwrap().into_pyarray(py)
    }

    /// Predict surrogate output derivatives at nsamples points.
    ///
    /// Parameters
    ///     x (array[nsamples, nx])
    ///         input values
    ///
    /// Returns
    ///     the output derivatives at nsamples x points (array[nsamples, nx]) wrt inputs
    ///     The ith column is the partial derivative value wrt to the ith component of x at the given samples.
    ///
    fn predict_gradients<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        self.0
            .predict_gradients(&x.as_array())
            .unwrap()
            .into_pyarray(py)
    }

    /// Predict variance derivatives at nsamples points.
    ///
    /// Parameters
    ///     x (array[nsamples, nx])
    ///         input values
    ///
    /// Returns
    ///     the variance derivatives at nsamples x points (array[nsamples, nx]) wrt inputs
    ///     The ith column is the partial derivative value wrt to the ith component of x at the given samples.
    ///
    fn predict_var_gradients<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        self.0
            .predict_var_gradients(&x.as_array())
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
    ) -> Bound<'py, PyArray2<f64>> {
        self.0
            .sample(&x.as_array(), n_traj)
            .unwrap()
            .into_pyarray(py)
    }

    /// Get the input and output dimensions of the surrogate
    ///
    /// Returns
    ///     the couple (nx, ny)
    ///
    fn dims(&self) -> (usize, usize) {
        self.0.dims()
    }

    /// Get the nt training data points used to fit the surrogate
    ///
    /// Returns
    ///     the couple (ndarray[nt, nx], ndarray[nt,])
    ///
    fn training_data<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
        let (xdata, ydata) = self.0.training_data();
        (
            xdata.to_owned().into_pyarray(py),
            ydata.to_owned().into_pyarray(py),
        )
    }

    /// Get optimized thetas hyperparameters (ie once GP experts are fitted)
    ///
    /// Returns
    ///     thetas as an array[n_clusters, nx or kpls_dim]
    ///
    fn thetas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let experts = self.0.experts();
        let proto = experts.first().expect("Mixture should contain an expert");
        let mut thetas = Array2::zeros((self.0.n_clusters(), proto.theta().len()));
        Zip::from(thetas.rows_mut())
            .and(experts)
            .for_each(|mut theta, expert| theta.assign(expert.theta()));
        thetas.into_pyarray(py)
    }

    /// Get GP expert variance (ie posterior GP variance)
    ///
    /// Returns
    ///     variances as an array[n_clusters]
    ///
    fn variances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let experts = self.0.experts();
        let mut variances = Array1::zeros(self.0.n_clusters());
        Zip::from(&mut variances)
            .and(experts)
            .for_each(|var, expert| *var = expert.variance());
        variances.into_pyarray(py)
    }

    /// Get reduced likelihood values gotten when fitting the GP experts
    ///
    /// Maybe used to compare various parameterization
    ///
    /// Returns
    ///     likelihood as an array[n_clusters]
    ///
    fn likelihoods<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let experts = self.0.experts();
        let mut likelihoods = Array1::zeros(self.0.n_clusters());
        Zip::from(&mut likelihoods)
            .and(experts)
            .for_each(|lkh, expert| *lkh = expert.likelihood());
        likelihoods.into_pyarray(py)
    }
}
