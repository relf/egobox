use crate::errors::Result;
use bitflags::bitflags;
#[allow(unused_imports)]
use gp::correlation_models::{
    AbsoluteExponentialCorr, Matern32Corr, Matern52Corr, SquaredExponentialCorr,
};
#[allow(unused_imports)]
use gp::mean_models::{ConstantMean, LinearMean, QuadraticMean};
use linfa::Float;
use linfa_clustering::GaussianMixtureModel;
use ndarray::Array2;
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Recombination<F: Float> {
    Hard,
    Smooth(Option<F>),
}

bitflags! {
    pub struct RegressionSpec: u8 {
        const CONSTANT = 0x01;
        const LINEAR = 0x02;
        const QUADRATIC = 0x04;
        const ALL = RegressionSpec::CONSTANT.bits
                    | RegressionSpec::LINEAR.bits
                    | RegressionSpec::QUADRATIC.bits;
    }
}

bitflags! {
    pub struct CorrelationSpec: u8 {
        const SQUAREDEXPONENTIAL = 0x01;
        const ABSOLUTEEXPONENTIAL = 0x02;
        const MATERN32 = 0x04;
        const MATERN52 = 0x08;
        const ALL = CorrelationSpec::SQUAREDEXPONENTIAL.bits
                    | CorrelationSpec::ABSOLUTEEXPONENTIAL.bits
                    | CorrelationSpec::MATERN32.bits
                    | CorrelationSpec::MATERN52.bits;
    }
}

pub trait MoePredict {
    fn predict_values(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
    fn predict_variances(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
}

pub trait MoeFit {
    fn fit_for_predict(&self, xt: &Array2<f64>, yt: &Array2<f64>) -> Result<Box<dyn MoePredict>>;
}

#[derive(Clone)]
pub struct MoeParams<F: Float, R: Rng + Clone> {
    n_clusters: usize,
    recombination: Recombination<F>,
    regression_spec: RegressionSpec,
    correlation_spec: CorrelationSpec,
    kpls_dim: Option<usize>,
    gmm: Option<Box<GaussianMixtureModel<F>>>,
    rng: R,
}

impl<F: Float> MoeParams<F, Isaac64Rng> {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(n_clusters: usize) -> MoeParams<F, Isaac64Rng> {
        Self::new_with_rng(n_clusters, Isaac64Rng::seed_from_u64(42))
    }
}

impl<F: Float, R: Rng + Clone> MoeParams<F, R> {
    pub fn new_with_rng(n_clusters: usize, rng: R) -> MoeParams<F, R> {
        MoeParams {
            n_clusters,
            recombination: Recombination::Smooth(Some(F::one())),
            regression_spec: RegressionSpec::ALL,
            correlation_spec: CorrelationSpec::ALL,
            kpls_dim: None,
            gmm: None,
            rng,
        }
    }

    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    pub fn recombination(&self) -> Recombination<F> {
        self.recombination
    }

    pub fn regression_spec(&self) -> RegressionSpec {
        self.regression_spec
    }

    pub fn correlation_spec(&self) -> CorrelationSpec {
        self.correlation_spec
    }

    pub fn kpls_dim(&self) -> Option<usize> {
        self.kpls_dim
    }

    pub fn gmm(&self) -> &Option<Box<GaussianMixtureModel<F>>> {
        &self.gmm
    }

    pub fn rng(&self) -> R {
        self.rng.clone()
    }

    pub fn set_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    pub fn set_recombination(mut self, recombination: Recombination<F>) -> Self {
        self.recombination = recombination;
        self
    }

    pub fn set_regression_spec(mut self, regression_spec: RegressionSpec) -> Self {
        self.regression_spec = regression_spec;
        self
    }

    pub fn set_correlation_spec(mut self, correlation_spec: CorrelationSpec) -> Self {
        self.correlation_spec = correlation_spec;
        self
    }

    pub fn set_kpls_dim(mut self, kpls_dim: Option<usize>) -> Self {
        self.kpls_dim = kpls_dim;
        self
    }

    pub fn set_gmm(mut self, gmm: Option<Box<GaussianMixtureModel<F>>>) -> Self {
        self.gmm = gmm;
        self
    }

    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> MoeParams<F, R2> {
        MoeParams {
            n_clusters: self.n_clusters,
            recombination: self.recombination,
            regression_spec: self.regression_spec,
            correlation_spec: self.correlation_spec,
            kpls_dim: self.kpls_dim,
            gmm: self.gmm,
            rng,
        }
    }
}
