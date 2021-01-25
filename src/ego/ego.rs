use crate::doe::{SamplingMethod, LHS};
use crate::errors::{EgoboxError, Result};
use crate::gaussian_process::{ConstantMean, GaussianProcess, SquaredExponentialKernel};
use finitediff::FiniteDiff;
use libm::erfc;
use ndarray::{s, stack, Array, Array1, Array2, ArrayBase, ArrayView, Axis, Data, Ix2, Zip};
use ndarray_npy::write_npy;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_stats::QuantileExt;
use nlopt::*;
use rand_isaac::Isaac64Rng;

const SQRT_2PI: f64 = 2.5066282746310007;

pub trait ObjFn: Send + Sync + 'static + Fn(&[f64]) -> f64 {}
impl<T> ObjFn for T where T: Send + Sync + 'static + Fn(&[f64]) -> f64 {}

#[derive(Debug)]
pub struct OptimResult {
    x_opt: Array1<f64>,
    y_opt: f64,
}

pub enum QEiStrategy {
    KrigingBeliever,
    // KrigingBelieverLowerBound,
    // KrigingBelieverUpperBound,
    // KrigingBelieverRandom,
    // ConstantLiarMinimum,
}

pub struct Ego<F: ObjFn, R: Rng> {
    pub n_iter: usize,
    pub n_start: usize,
    pub n_parallel: usize,
    pub n_doe: usize,
    pub x_doe: Option<Array2<f64>>,
    pub xlimits: Array2<f64>,
    pub q_ei: QEiStrategy,
    pub obj: F,
    pub rng: R,
}

impl<F: ObjFn> Ego<F, Isaac64Rng> {
    pub fn new(f: F, xlimits: &Array2<f64>) -> Ego<F, Isaac64Rng> {
        Self::new_with_rng(f, &xlimits, Isaac64Rng::seed_from_u64(42))
    }
}

impl<F: ObjFn, R: Rng + Clone> Ego<F, R> {
    pub fn new_with_rng(f: F, xlimits: &Array2<f64>, rng: R) -> Self {
        Ego {
            n_iter: 20,
            n_start: 20,
            n_parallel: 1,
            n_doe: 10,
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

        let mut x_data = if let Some(xdoe) = &self.x_doe {
            xdoe.to_owned()
        } else {
            sampling.sample(self.n_doe)
        };

        let mut y_data = self.obj_eval(&x_data);

        for _ in 0..self.n_iter {
            for _ in 0..self.n_parallel {
                let gpr = GaussianProcess::<ConstantMean, SquaredExponentialKernel>::params(
                    ConstantMean::default(),
                    SquaredExponentialKernel::default(),
                )
                .fit(&x_data, &y_data)
                .expect("GP training failure");

                let f_min = y_data.min().unwrap();
                if false {
                    let xplot = Array::linspace(0., 25., 100).insert_axis(Axis(1));
                    let obj = self.obj_eval(&xplot);
                    let gpr_vals = gpr.predict_values(&xplot).unwrap();
                    let gpr_vars = gpr.predict_variances(&xplot).unwrap();
                    let ei = xplot.map(|x| Self::ei(&[*x], &gpr, *f_min));
                    write_npy("ego_x.npy", xplot).expect("xplot saved");
                    write_npy("ego_obj.npy", obj).expect("obj saved");
                    write_npy("ego_gpr.npy", gpr_vals).expect("gp vals saved");
                    write_npy("ego_gpr_vars.npy", gpr_vars).expect("gp vars saved");
                    write_npy("ego_ei.npy", ei).expect("gp vars saved");
                }

                match self.find_best_point(&x_data, &y_data, &sampling, &gpr) {
                    Ok(xk) => match self.get_virtual_point(&xk, &y_data, &gpr) {
                        Ok(yk) => {
                            y_data = stack![Axis(0), y_data, Array2::from_elem((1, 1), yk)];
                            x_data = stack![Axis(0), x_data, xk.insert_axis(Axis(0))];
                        }
                        Err(_) => {
                            break;
                        }
                    },
                    Err(_) => {
                        break;
                    }
                }
            }
            let n_par = -(self.n_parallel as i32);
            let x_to_eval = x_data.slice(s![n_par.., ..]);
            let y_actual = self.obj_eval(&x_to_eval);
            Zip::from(y_data.slice_mut(s![n_par.., ..]).gencolumns_mut())
                .and(y_actual.gencolumns())
                .apply(|mut y, val| y.assign(&val));
        }

        let best_index = y_data.argmin().unwrap().0;
        OptimResult {
            x_opt: x_data.row(best_index).to_owned(),
            y_opt: y_data.row(best_index)[0],
        }
    }

    pub fn find_best_point(
        &self,
        x_data: &Array2<f64>,
        y_data: &Array2<f64>,
        sampling: &LHS<R>,
        gpr: &GaussianProcess<ConstantMean, SquaredExponentialKernel>,
    ) -> Result<Array1<f64>> {
        let f_min = y_data.min().unwrap();
        let obj = |x: &[f64], gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            if let Some(grad) = gradient {
                let f = |x: &Vec<f64>| -> f64 { -Self::ei(x, &gpr, *f_min) };
                grad[..].copy_from_slice(&x.to_vec().forward_diff(&f));
            }
            -Self::ei(x, &gpr, *f_min)
        };
        let mut optimizer = Nlopt::new(Algorithm::Slsqp, x_data.ncols(), obj, Target::Minimize, ());
        optimizer
            .set_lower_bounds(self.xlimits.column(0).to_owned().as_slice().unwrap())
            .unwrap();
        optimizer
            .set_upper_bounds(self.xlimits.column(1).to_owned().as_slice().unwrap())
            .unwrap();
        optimizer.set_maxeval(200).unwrap();
        optimizer.set_ftol_rel(1e-4).unwrap();
        optimizer.set_ftol_abs(1e-4).unwrap();
        let mut success = false;
        let mut n_optim = 1;
        let n_max_optim = 20;
        let mut best_x = None;

        while !success && n_optim <= n_max_optim {
            let mut best_opt = f64::INFINITY;
            let x_start = sampling.sample(self.n_start);

            for i in 0..self.n_start {
                let mut x_opt = x_start.row(i).to_owned().into_raw_vec();
                match optimizer.optimize(&mut x_opt) {
                    Ok((_, opt)) => {
                        if opt < best_opt {
                            best_opt = opt;
                            best_x = Some(Array::from(x_opt));
                            success = true;
                        }
                    }
                    Err((_, _)) => {}
                }
            }
            n_optim += 1;
        }
        best_x.ok_or_else(|| EgoboxError::EgoError(String::from("Can not find best point")))
    }

    pub fn get_virtual_point(
        &self,
        xk: &Array1<f64>,
        _y_data: &Array2<f64>,
        gpr: &GaussianProcess<ConstantMean, SquaredExponentialKernel>,
    ) -> Result<f64> {
        let x = &xk.to_owned().insert_axis(Axis(0));
        let pred = gpr.predict_values(&x)?[[0, 0]];
        let var = gpr.predict_variances(&x)?[[0, 0]];
        let conf = -3.; // Kriging Believer Lower Bound
        Ok(pred + conf * f64::sqrt(var))
    }

    pub fn ei(
        x: &[f64],
        gpr: &GaussianProcess<ConstantMean, SquaredExponentialKernel>,
        f_min: f64,
    ) -> f64 {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        if let Ok(p) = gpr.predict_values(&pt) {
            if let Ok(s) = gpr.predict_variances(&pt) {
                let pred = p[[0, 0]];
                let sigma = f64::sqrt(s[[0, 0]]);
                let args0 = (f_min - pred) / sigma;
                let args1 = (f_min - pred) * Self::norm_cdf(args0);
                let args2 = sigma * Self::norm_pdf(args0);
                args1 + args2
            } else {
                -f64::INFINITY
            }
        } else {
            -f64::INFINITY
        }
    }

    pub fn norm_cdf(x: f64) -> f64 {
        0.5 * erfc(-x / std::f64::consts::SQRT_2)
    }

    pub fn norm_pdf(x: f64) -> f64 {
        (-0.5 * x * x).exp() / SQRT_2PI
    }

    pub fn obj_eval<D: Data<Elem = f64>>(&self, x: &ArrayBase<D, Ix2>) -> Array2<f64> {
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
    use approx::assert_abs_diff_eq;
    // use argmin_testfunctions::rosenbrock;
    use ndarray::array;
    use std::time::Instant;

    #[test]
    fn test_xsinx() {
        fn xsinx(x: &[f64]) -> f64 {
            (x[0] - 3.5) * f64::sin((x[0] - 3.5) / std::f64::consts::PI)
        };
        let now = Instant::now();
        let res = Ego::new(xsinx, &array![[0.0, 25.0]])
            .n_iter(6)
            .x_doe(&array![[0.], [7.], [25.]])
            .minimize();
        println!("xsinx optim result = {:?}", res);
        println!("Elapsed = {:?}", now.elapsed());
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 5e-1);
    }

    // #[test]
    // fn test_rosenbrock_2d() {
    //     fn rosenb(x: &[f64]) -> f64 {
    //         rosenbrock(x, 1., 100.)
    //     };
    //     let now = Instant::now();
    //     let xlimits = array![[-2., 2.], [-2., 2.]];
    //     let doe = FullFactorial::new(&xlimits).sample(10);
    //     let res = Ego::new(rosenb, &xlimits).x_doe(&doe).n_iter(40).minimize();
    //     println!("Rosenbrock optim result = {:?}", res);
    //     println!("Elapsed = {:?}", now.elapsed());
    //     let expected = array![1., 1.];
    //     assert_abs_diff_eq!(expected, res.x_opt, epsilon = 6e-1);
    // }
}
