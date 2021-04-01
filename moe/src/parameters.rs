use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;

#[derive(Clone, Copy, PartialEq)]
pub enum Recombination {
    Hard,
    Smooth,
}

pub struct MoeParams<R: Rng + Clone> {
    n_clusters: usize,
    recombination: Recombination,
    heaviside_factor: f64,
    kpls_dim: Option<usize>,
    rng: R,
}

impl MoeParams<Isaac64Rng> {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(n_clusters: usize) -> MoeParams<Isaac64Rng> {
        Self::new_with_rng(n_clusters, Isaac64Rng::seed_from_u64(42))
    }
}

impl<R: Rng + Clone> MoeParams<R> {
    pub fn new_with_rng(n_clusters: usize, rng: R) -> MoeParams<R> {
        MoeParams {
            n_clusters,
            recombination: Recombination::Hard,
            heaviside_factor: 1.0,
            kpls_dim: None,
            rng,
        }
    }

    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    pub fn recombination(&self) -> Recombination {
        self.recombination
    }

    pub fn heaviside_factor(&self) -> f64 {
        self.heaviside_factor
    }

    pub fn kpls_dim(&self) -> Option<usize> {
        self.kpls_dim
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

    pub fn set_heaviside_factor(mut self, heaviside_factor: f64) -> Self {
        self.heaviside_factor = heaviside_factor;
        self
    }

    pub fn set_kpls_dim(mut self, kpls_dim: Option<usize>) -> Self {
        self.kpls_dim = kpls_dim;
        self
    }

    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> MoeParams<R2> {
        MoeParams {
            n_clusters: self.n_clusters,
            recombination: self.recombination,
            heaviside_factor: self.heaviside_factor,
            kpls_dim: self.kpls_dim,
            rng,
        }
    }
}
