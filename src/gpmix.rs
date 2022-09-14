use crate::types::*;
use egobox_moe::{Moe, MoeError};
use linfa::{traits::Fit, Dataset};
use ndarray_rand::rand::SeedableRng;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rand_isaac::Isaac64Rng;
#[pyclass]
#[pyo3(text_signature = "()")]
pub(crate) struct GpMix {
    pub n_clusters: usize,
    pub regression_spec: RegressionSpec,
    pub correlation_spec: CorrelationSpec,
    pub recombination: Recombination,
    pub kpls_dim: Option<usize>,
    pub _outdir: Option<String>,
    pub seed: Option<u64>,
    pub training_data: Option<Dataset<f64, f64>>,
    pub moe: Option<Moe>,
}

#[pymethods]
impl GpMix {
    #[new]
    #[args(
        n_clusters = "1",
        regr_spec = "RegressionSpec::ALL",
        corr_spec = "CorrelationSpec::ALL",
        recombination = "Recombination::Smooth",
        kpls_dim = "None",
        outdir = "None",
        seed = "None"
    )]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_clusters: usize,
        regr_spec: u8,
        corr_spec: u8,
        recombination: Recombination,
        kpls_dim: Option<usize>,
        _outdir: Option<String>,
        seed: Option<u64>,
    ) -> Self {
        GpMix {
            n_clusters,
            regression_spec: RegressionSpec(regr_spec),
            correlation_spec: CorrelationSpec(corr_spec),
            recombination,
            kpls_dim,
            _outdir,
            seed,
            training_data: None,
            moe: None,
        }
    }

    fn set_training_values(&mut self, xt: PyReadonlyArray2<f64>, yt: PyReadonlyArray2<f64>) {
        self.training_data = Some(Dataset::new(
            xt.as_array().to_owned(),
            yt.as_array().to_owned(),
        ));
    }

    fn train(&mut self) {
        let recomb = match self.recombination {
            Recombination::Hard => egobox_moe::Recombination::Hard,
            Recombination::Smooth => egobox_moe::Recombination::Smooth(None),
        };
        let rng = if let Some(seed) = self.seed {
            Isaac64Rng::seed_from_u64(seed)
        } else {
            Isaac64Rng::from_entropy()
        };
        self.moe = Some(
            Moe::params()
                .n_clusters(self.n_clusters)
                .recombination(recomb)
                .regression_spec(
                    egobox_moe::RegressionSpec::from_bits(self.regression_spec.0).unwrap(),
                )
                .correlation_spec(
                    egobox_moe::CorrelationSpec::from_bits(self.correlation_spec.0).unwrap(),
                )
                .kpls_dim(self.kpls_dim)
                .with_rng(rng)
                .fit(self.training_data.as_ref().unwrap())
                .expect("MoE model training"),
        )
    }

    fn predict_values<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> &'py PyArray2<f64> {
        self.moe
            .as_ref()
            .unwrap()
            .predict_values(&x.as_array().to_owned())
            .unwrap()
            .into_pyarray(py)
    }

    fn predict_variances<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        self.moe
            .as_ref()
            .unwrap()
            .predict_variances(&x.as_array().to_owned())
            .unwrap()
            .into_pyarray(py)
    }

    fn save(filename: &str) {
        self.moe.save(&filename)
    }
}

pub(crate) struct Gpx {
    pub moe: Option<Box<Moe>>,
}

impl Gpx {
    fn load(filename: &str) -> Gpx {
        let moe = Moe::load(filename);
    }
}
