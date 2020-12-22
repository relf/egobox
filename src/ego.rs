use crate::doe::LHS;
use crate::gaussian_process::{ConstantMean, GaussianProcess, SquaredExponentialKernel};
use libm::erfc;
use ndarray::{Array, Array1, Array2, ArrayView, Axis, Zip};
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_stats::QuantileExt;
use nlopt::*;
use rand_isaac::Isaac64Rng;

const SQRT_2PI: f64 = 2.5066282746310005024157652848110452530069867406099;

pub trait ObjFn: Send + Sync + 'static + Fn(&[f64]) -> f64 {}
impl<T> ObjFn for T where T: Send + Sync + 'static + Fn(&[f64]) -> f64 {}

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
    xlimits: Array2<f64>,
    q_ei: QEiStrategy,
    obj: F,
    rng: R,
}

impl<F: ObjFn> Ego<F, Isaac64Rng> {
    pub fn new(f: F, xlimits: &Array2<f64>) -> Ego<F, Isaac64Rng> {
        Self::new_with_rng(f, &xlimits, Isaac64Rng::seed_from_u64(42))
    }
}

impl<F: ObjFn, R: Rng + Clone> Ego<F, R> {
    pub fn new_with_rng(f: F, xlimits: &Array2<f64>, rng: R) -> Self {
        Ego {
            n_iter: 10,
            n_start: 10,
            n_parallel: 1,
            n_doe: 5,
            x_doe: None,
            xlimits: xlimits.to_owned(),
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
            xlimits: self.xlimits,
            q_ei: self.q_ei,
            obj: self.obj,
            rng,
        }
    }

    pub fn minimize(&mut self) -> OptimResult {
        let rng = self.rng.clone();

        let sampling = LHS::new(&self.xlimits).with_rng(rng);

        let x_data = sampling.sample(self.n_doe);
        let y_data = self.obj_eval(&x_data);

        for k in 0..self.n_iter {
            for p in 0..self.n_parallel {
                let xk = self.find_best_point(&x_data, &y_data, &sampling).unwrap();
                println!("xk = {:}", xk);
                let yk = self.get_virtual_point(&xk, &y_data);
            }
        }

        OptimResult {
            x_opt: Array1::default(1),
            y_opt: 0.,
        }
    }

    pub fn find_best_point(
        &self,
        x_data: &Array2<f64>,
        y_data: &Array2<f64>,
        sampling: &LHS<R>,
    ) -> Result<Array1<f64>, &str> {
        let gpr = GaussianProcess::<ConstantMean, SquaredExponentialKernel>::params(
            ConstantMean::default(),
            SquaredExponentialKernel::default(),
        )
        .fit(&x_data, &y_data)
        .expect("GP training failure");
        let f_min = y_data.min().unwrap();
        let obj = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            Self::ei(x, &gpr, *f_min)
        };
        let mut optimizer = Nlopt::new(Algorithm::Lbfgs, x_data.ncols(), obj, Target::Minimize, ());
        optimizer.set_maxeval(200).unwrap();
        optimizer
            .set_lower_bounds(self.xlimits.column(0).to_owned().as_slice().unwrap())
            .unwrap();
        optimizer
            .set_upper_bounds(self.xlimits.column(1).to_owned().as_slice().unwrap())
            .unwrap();
        let mut success = false;
        let mut n_optim = 1;
        let n_max_optim = 20;
        let mut best_x = None;

        while !success && n_optim <= n_max_optim {
            let mut best_opt = f64::INFINITY;
            let x_start = sampling.sample(self.n_start);

            for i in 0..self.n_start {
                let mut x_opt = x_start.row(i).to_owned().into_raw_vec();
                if let Ok((_, opt)) = optimizer.optimize(&mut x_opt) {
                    if opt < best_opt {
                        best_opt = opt;
                        best_x = Some(Array::from(x_opt));
                        success = true;
                    }
                }
            }
            n_optim += 1;
        }
        best_x.ok_or("Can not find best point")
    }

    pub fn get_virtual_point(&self, xk: &Array1<f64>, y_data: &Array2<f64>) -> Array1<f64> {
        Array1::<f64>::default(1)
    }

    pub fn ei(
        x: &[f64],
        gpr: &GaussianProcess<ConstantMean, SquaredExponentialKernel>,
        f_min: f64,
    ) -> f64 {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        let pred = gpr.predict_values(&pt).unwrap()[[0, 0]];
        let sigma = f64::sqrt(gpr.predict_variances(&pt).unwrap()[[0, 0]]);
        let args0 = f_min - pred / sigma;
        let args1 = (f_min - pred) * Self::norm_cdf(args0);
        let args2 = sigma * Self::norm_pdf(args0);
        args1 + args2
    }

    pub fn norm_cdf(x: f64) -> f64 {
        0.5 * erfc(-x / std::f64::consts::SQRT_2)
    }

    pub fn norm_pdf(x: f64) -> f64 {
        (-0.5 * x * x).exp() / SQRT_2PI
    }

    pub fn obj_eval(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut y = Array1::zeros(x.nrows());
        let obj = &self.obj;
        Zip::from(&mut y)
            .and(x.genrows())
            .par_apply(|yn, xn| *yn = (obj)(xn.to_slice().unwrap()));
        y.insert_axis(Axis(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argmin_testfunctions::rosenbrock;
    use ndarray::array;

    #[test]
    fn test_ego() {
        fn rosenb(x: &[f64]) -> f64 {
            rosenbrock(x, 100., 1.)
        };

        let res = Ego::new(rosenb, &array![[-2., 2.], [-2., 2.]]).minimize();
        println!("Rosenbrock optim result = {:?}", res);
    }
}
