use crate::types::*;
use egobox_moe::MoeValidParams;
use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rand_isaac::Isaac64Rng;

#[pyclass]
#[pyo3(text_signature = "()")]
pub(crate) struct GpMix {
    pub n_clusters: Option<usize>,
    pub regression_spec: RegressionSpec,
    pub correlation_spec: CorrelationSpec,
    pub kpls_dim: Option<usize>,
    pub outdir: Option<String>,
    pub seed: Option<u64>,
    pub xt: Option<Array2<f64>>,
    pub yt: Option<Array2<f64>>,
    pub moe_params: Option<MoeValidParams<f64, Isaac64Rng>>,
}

#[pymethods]
impl GpMix {
    #[new]
    #[args(
        n_clusters = "1",
        regr_spec = "RegressionSpec::ALL",
        corr_spec = "CorrelationSpec::ALL",
        kpls_dim = "None",
        outdir = "None",
        seed = "None"
    )]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_clusters: Option<usize>,
        regr_spec: u8,
        corr_spec: u8,
        kpls_dim: Option<usize>,
        outdir: Option<String>,
        seed: Option<u64>,
    ) -> Self {
        GpMix {
            n_clusters,
            regression_spec: RegressionSpec(regr_spec),
            correlation_spec: CorrelationSpec(corr_spec),
            kpls_dim,
            outdir,
            seed,
            xt: None,
            yt: None,
            moe_params: None,
        }
    }

    fn set_training_values(&mut self, xt: PyReadonlyArray2<f64>, yt: PyReadonlyArray2<f64>) {
        self.xt = Some(xt.as_array().to_owned());
        self.yt = Some(yt.as_array().to_owned());
    }

    fn train(&mut self) {}

    fn predict_values(&self, x: PyReadonlyArray2<f64>) -> PyArray2<f64> {
        todo!()
    }

    fn predict_variances(&self, x: PyReadonlyArray2<f64>) -> PyArray2<f64> {
        todo!()
    }
}
