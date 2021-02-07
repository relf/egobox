use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;

#[derive(Clone, Copy, PartialEq)]
pub enum Recombination {
    Hard,
    Smooth,
}

pub struct MoeHyperParams<R: Rng + Clone> {
    n_clusters: usize,
    recombination: Recombination,
    heaviside_factor: f64,
    rng: R,
}

impl MoeHyperParams<Isaac64Rng> {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(n_clusters: usize) -> MoeHyperParams<Isaac64Rng> {
        Self::new_with_rng(n_clusters, Isaac64Rng::seed_from_u64(42))
    }
}

impl<R: Rng + Clone> MoeHyperParams<R> {
    pub fn new_with_rng(n_clusters: usize, rng: R) -> MoeHyperParams<R> {
        MoeHyperParams {
            n_clusters,
            recombination: Recombination::Hard,
            heaviside_factor: 1.0,
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

    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> MoeHyperParams<R2> {
        MoeHyperParams {
            n_clusters: self.n_clusters,
            recombination: self.recombination,
            heaviside_factor: self.heaviside_factor,
            rng,
        }
    }
}
