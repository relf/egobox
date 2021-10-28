use bitflags::bitflags;
#[allow(unused_imports)]
use gp::correlation_models::{
    AbsoluteExponentialKernel, Matern32Kernel, Matern52Kernel, SquaredExponentialKernel,
};
#[allow(unused_imports)]
use gp::mean_models::{ConstantMean, LinearMean, QuadraticMean};
use gp::Float;
use linfa_clustering::GaussianMixtureModel;
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;

#[derive(Clone, Copy, PartialEq)]
pub enum Recombination {
    Hard,
    Smooth,
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

pub struct MoeParams<F: Float, R: Rng + Clone> {
    n_clusters: usize,
    recombination: Recombination,
    regression_spec: RegressionSpec,
    correlation_spec: CorrelationSpec,
    heaviside_factor: F,
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
            recombination: Recombination::Smooth,
            regression_spec: RegressionSpec::CONSTANT,
            correlation_spec: CorrelationSpec::SQUAREDEXPONENTIAL,
            heaviside_factor: F::one(),
            kpls_dim: None,
            gmm: None,
            rng,
        }
    }

    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    pub fn recombination(&self) -> Recombination {
        self.recombination
    }

    pub fn regression_spec(&self) -> RegressionSpec {
        self.regression_spec
    }

    pub fn correlation_spec(&self) -> CorrelationSpec {
        self.correlation_spec
    }

    pub fn heaviside_factor(&self) -> F {
        self.heaviside_factor
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

    pub fn set_recombination(mut self, recombination: Recombination) -> Self {
        self.recombination = recombination;
        self
    }

    pub fn set_heaviside_factor(mut self, heaviside_factor: F) -> Self {
        self.heaviside_factor = heaviside_factor;
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
            heaviside_factor: self.heaviside_factor,
            kpls_dim: self.kpls_dim,
            gmm: self.gmm,
            rng,
        }
    }
}
