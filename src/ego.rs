use ndarray::{Array1, Array2};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;

pub trait ObjFn: Fn(&[f64]) -> f64 {}
impl<T> ObjFn for T where T: Fn(&[f64]) -> f64 {}

#[derive(Debug)]
pub struct OptimResult {
    x_opt: Array1<f64>,
    y_opt: f64,
}

pub enum QEiStrategy {
    KrigingBeliever,
    KrigingBelieverLowerBound,
    KrigingBelieverUpperBound,
    KrigingBelieverRandom,
    ConstantLiarMinimum,
}

pub struct Ego<F: ObjFn, R: Rng> {
    n_iter: usize,
    n_start: usize,
    n_parallel: usize,
    n_doe: usize,
    x_doe: Option<Array2<f64>>,
    q_ei: QEiStrategy,
    obj: F,
    rng: R,
}

impl<F: ObjFn> Ego<F, Isaac64Rng> {
    pub fn new(f: F) -> Ego<F, Isaac64Rng> {
        Self::new_with_rng(f, Isaac64Rng::seed_from_u64(42))
    }
}

impl<F: ObjFn, R: Rng + Clone> Ego<F, R> {
    pub fn new_with_rng(f: F, rng: R) -> Self {
        Ego {
            n_iter: 10,
            n_start: 10,
            n_parallel: 1,
            n_doe: 5,
            x_doe: None,
            q_ei: QEiStrategy::KrigingBeliever,
            obj: f,
            rng,
        }
    }

    pub fn n_iter(&mut self, n_iter: usize) -> &mut Self {
        self.n_iter = n_iter;
        self
    }

    pub fn n_start(&mut self, n_start: usize) -> &mut Self {
        self.n_start = n_start;
        self
    }

    pub fn n_parallel(&mut self, n_parallel: usize) -> &mut Self {
        self.n_parallel = n_parallel;
        self
    }

    pub fn n_doe(&mut self, n_doe: usize) -> &mut Self {
        self.n_doe = n_doe;
        self
    }

    pub fn x_doe(&mut self, x_doe: &Array2<f64>) -> &mut Self {
        self.x_doe = Some(x_doe.to_owned());
        self
    }

    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> Ego<F, R2> {
        Ego {
            n_iter: self.n_iter,
            n_start: self.n_start,
            n_parallel: self.n_parallel,
            n_doe: self.n_doe,
            x_doe: self.x_doe,
            q_ei: self.q_ei,
            obj: self.obj,
            rng,
        }
    }

    pub fn with_objective<F2: ObjFn>(self, f: F2) -> Ego<F2, R> {
        Ego {
            n_iter: self.n_iter,
            n_start: self.n_start,
            n_parallel: self.n_parallel,
            n_doe: self.n_doe,
            x_doe: self.x_doe,
            q_ei: self.q_ei,
            obj: f,
            rng: self.rng,
        }
    }

    pub fn minimize(&self) -> OptimResult {
        OptimResult {
            x_opt: Array1::default(1),
            y_opt: 0.,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argmin_testfunctions::rosenbrock;

    #[test]
    fn test_ego() {
        fn rosenb(x: &[f64]) -> f64 {
            rosenbrock(x, 100., 1.)
        };

        let res = Ego::new(rosenb).minimize();
        println!("Rosenbrock optim result = {:?}", res);
    }
}
