//! Mixture of Gaussian process models used by the Egor solver

pub mod mixint;
pub mod spec;

use egobox_gp::ThetaTuning;
use egobox_moe::{
    Clustering, CorrelationSpec, GpMixtureParams, MixtureGpSurrogate, NbClusters, RegressionSpec,
};
use ndarray::{ArrayView1, ArrayView2};

use linfa::ParamGuard;

use crate::Result;
use crate::{SurrogateBuilder, XType};

impl SurrogateBuilder for GpMixtureParams<f64> {
    /// Constructor from domain space specified with types
    /// **panic** if xtypes contains other types than continuous type `Float`
    fn new_with_xtypes(xtypes: &[XType]) -> Self {
        if crate::utils::discrete(xtypes) {
            panic!("GpMixtureParams cannot be created with discrete types!");
        }
        GpMixtureParams::new()
    }

    /// Sets the allowed regression models used in gaussian processes.
    fn set_regression_spec(&mut self, regression_spec: RegressionSpec) {
        *self = self.clone().regression_spec(regression_spec);
    }

    /// Sets the allowed correlation models used in gaussian processes.
    fn set_correlation_spec(&mut self, correlation_spec: CorrelationSpec) {
        *self = self.clone().correlation_spec(correlation_spec);
    }

    /// Sets the number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
    fn set_kpls_dim(&mut self, kpls_dim: Option<usize>) {
        *self = self.clone().kpls_dim(kpls_dim);
    }

    /// Sets the number of clusters used by the mixture of surrogate experts.
    fn set_n_clusters(&mut self, n_clusters: NbClusters) {
        *self = self.clone().n_clusters(n_clusters);
    }

    /// Sets the mode of recombination to get the output prediction from experts prediction
    /// Onlyused if nb clusters is greater than one
    fn set_recombination(&mut self, recombination: egobox_moe::Recombination<f64>) {
        *self = self.clone().recombination(recombination);
    }

    /// Sets the theta tuning used by the expert during training.
    /// When only one element tuning is used for all clusters
    /// When several elements, the length should match the number of clusters
    fn set_theta_tunings(&mut self, theta_tunings: &[ThetaTuning<f64>]) {
        *self = self.clone().theta_tunings(theta_tunings);
    }

    /// Sets the number of clusters used by the mixture of surrogate experts.
    fn set_optim_params(&mut self, n_start: usize, max_eval: usize) {
        *self = self.clone().n_start(n_start).max_eval(max_eval);
    }

    fn train(
        &self,
        xt: ArrayView2<f64>,
        yt: ArrayView1<f64>,
    ) -> Result<Box<dyn MixtureGpSurrogate>> {
        let checked = self.check_ref()?;
        let moe = checked.train(&xt, &yt)?;
        Ok(moe).map(|moe| Box::new(moe) as Box<dyn MixtureGpSurrogate>)
    }

    fn train_on_clusters(
        &self,
        xt: ArrayView2<f64>,
        yt: ArrayView1<f64>,
        clustering: &Clustering,
    ) -> Result<Box<dyn MixtureGpSurrogate>> {
        let checked = self.check_ref()?;
        let moe = checked.train_on_clusters(&xt, &yt, clustering)?;
        Ok(moe).map(|moe| Box::new(moe) as Box<dyn MixtureGpSurrogate>)
    }
}
