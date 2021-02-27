use crate::errors::{EgoboxError, Result};
use doe::{LHSKind, SamplingMethod, LHS};
use finitediff::FiniteDiff;
use gp::{ConstantMean, GaussianProcess, SquaredExponentialKernel};
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

#[derive(Debug, PartialEq)]
pub enum AcqStrategy {
    EI,
    WB2,
    WB2S,
}

#[derive(Debug, PartialEq)]
pub enum QEiStrategy {
    KrigingBeliever,
    KrigingBelieverLowerBound,
    KrigingBelieverUpperBound,
    ConstantLiarMinimum,
}

/// A structure to pass data to objective acquisition function
struct ObjData {
    scale: f64,
    scale_wb2: Option<f64>,
}

pub struct Ego<F: ObjFn, R: Rng> {
    pub n_iter: usize,
    pub n_start: usize,
    pub n_parallel: usize,
    pub n_doe: usize,
    pub x_doe: Option<Array2<f64>>,
    pub xlimits: Array2<f64>,
    pub q_ei: QEiStrategy,
    pub acq: AcqStrategy,
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
            acq: AcqStrategy::EI,
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

    pub fn qei_strategy(&mut self, q_ei: QEiStrategy) -> &mut Self {
        self.q_ei = q_ei;
        self
    }

    pub fn acq_strategy(&mut self, acq: AcqStrategy) -> &mut Self {
        self.acq = acq;
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
            acq: self.acq,
            obj: self.obj,
            rng,
        }
    }

    fn next_points(
        &self,
        x_data: &Array2<f64>,
        y_data: &Array2<f64>,
        sampling: &LHS<R>,
    ) -> (Array2<f64>, Array2<f64>) {
        let mut x_dat = Array2::zeros((0, x_data.ncols()));
        let mut y_dat = Array2::zeros((0, y_data.ncols()));
        for i in 0..self.n_parallel {
            let gpr = GaussianProcess::<ConstantMean, SquaredExponentialKernel>::params(
                ConstantMean::default(),
                SquaredExponentialKernel::default(),
            )
            .fit(&x_data, &y_data)
            .expect("GP training failure");

            if i == 0 {
                let f_min = y_data.min().unwrap();
                let xplot = Array::linspace(0., 25., 100).insert_axis(Axis(1));
                let obj = self.obj_eval(&xplot);
                let gpr_vals = gpr.predict_values(&xplot).unwrap();
                let gpr_vars = gpr.predict_variances(&xplot).unwrap();
                let ei = xplot.map(|x| Self::ei(&[*x], &gpr, *f_min));
                let wb2 = xplot.map(|x| Self::wb2s(&[*x], &gpr, *f_min, None));
                write_npy("ego_x.npy", xplot).expect("xplot saved");
                write_npy("ego_obj.npy", obj).expect("obj saved");
                write_npy("ego_gpr.npy", gpr_vals).expect("gp vals saved");
                write_npy("ego_gpr_vars.npy", gpr_vars).expect("gp vars saved");
                write_npy("ego_ei.npy", ei).expect("ei saved");
                write_npy("ego_wb2.npy", wb2).expect("wb2 saved");
            }

            match self.find_best_point(&x_data, &y_data, &sampling, &gpr) {
                Ok(xk) => match self.get_virtual_point(&xk, &y_data, &gpr) {
                    Ok(yk) => {
                        y_dat = stack![Axis(0), y_dat, Array2::from_elem((1, 1), yk)];
                        x_dat = stack![Axis(0), x_dat, xk.insert_axis(Axis(0))];
                    }
                    Err(_) => {
                        // Error while predict at best point: ignore
                        break;
                    }
                },
                Err(_) => {
                    // Cannot find best point: ignore
                    break;
                }
            }
        }
        (x_dat, y_dat)
    }

    pub fn suggest(&self, x_data: &Array2<f64>, y_data: &Array2<f64>) -> Array2<f64> {
        let rng = self.rng.clone();
        let sampling = LHS::new(&self.xlimits).with_rng(rng).kind(LHSKind::Maximin);
        let (x_dat, _) = self.next_points(&x_data, &y_data, &sampling);
        x_dat
    }

    pub fn minimize(&mut self) -> OptimResult {
        let rng = self.rng.clone();
        let sampling = LHS::new(&self.xlimits).with_rng(rng).kind(LHSKind::Maximin);

        let mut x_data = if let Some(xdoe) = &self.x_doe {
            xdoe.to_owned()
        } else {
            sampling.sample(self.n_doe)
        };

        let mut y_data = self.obj_eval(&x_data);

        for _ in 0..self.n_iter {
            let (x_dat, y_dat) = self.next_points(&x_data, &y_data, &sampling);
            y_data = stack![Axis(0), y_data, y_dat];
            x_data = stack![Axis(0), x_data, x_dat];
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

    fn find_best_point(
        &self,
        x_data: &Array2<f64>,
        y_data: &Array2<f64>,
        sampling: &LHS<R>,
        gpr: &GaussianProcess<ConstantMean, SquaredExponentialKernel>,
    ) -> Result<Array1<f64>> {
        let f_min = y_data.min().unwrap();

        let obj = |x: &[f64], gradient: Option<&mut [f64]>, params: &mut ObjData| -> f64 {
            let ObjData {
                scale, scale_wb2, ..
            } = params;
            if let Some(grad) = gradient {
                let f =
                    |x: &Vec<f64>| -> f64 { self.acq_eval(x, &gpr, *f_min, *scale, *scale_wb2) };
                grad[..].copy_from_slice(&x.to_vec().forward_diff(&f));
            }
            self.acq_eval(x, &gpr, *f_min, *scale, *scale_wb2)
        };

        let mut success = false;
        let mut n_optim = 1;
        let n_max_optim = 20;
        let mut best_x = None;

        while !success && n_optim <= n_max_optim {
            let scaling_points = sampling.sample(100 * self.xlimits.nrows());
            let scale_wb2 = Self::compute_wb2s_scale(&scaling_points, &gpr, *f_min);
            let scale_obj = self._compute_obj_scale(&scaling_points);

            if n_optim == -1 {
                let xplot = Array::linspace(0., 25., 100).insert_axis(Axis(1));
                let wb2s = xplot.map(|x| Self::wb2s(&[*x], &gpr, *f_min, Some(scale_wb2)));
                write_npy("ego_wb2s.npy", wb2s).expect("wb2 saved");
            }

            let mut optimizer = Nlopt::new(
                Algorithm::Slsqp,
                x_data.ncols(),
                obj,
                Target::Minimize,
                ObjData {
                    scale: scale_obj,
                    scale_wb2: Some(scale_wb2),
                },
            );
            optimizer
                .set_lower_bounds(self.xlimits.column(0).to_owned().as_slice().unwrap())
                .unwrap();
            optimizer
                .set_upper_bounds(self.xlimits.column(1).to_owned().as_slice().unwrap())
                .unwrap();
            optimizer.set_maxeval(200).unwrap();
            optimizer.set_ftol_rel(1e-4).unwrap();
            optimizer.set_ftol_abs(1e-4).unwrap();

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

    fn get_virtual_point(
        &self,
        xk: &Array1<f64>,
        y_data: &Array2<f64>,
        gpr: &GaussianProcess<ConstantMean, SquaredExponentialKernel>,
    ) -> Result<f64> {
        if self.q_ei == QEiStrategy::ConstantLiarMinimum {
            Ok(*y_data.min().unwrap())
        } else {
            let x = &xk.to_owned().insert_axis(Axis(0));
            let pred = gpr.predict_values(&x)?[[0, 0]];
            let var = gpr.predict_variances(&x)?[[0, 0]];
            let conf = match self.q_ei {
                QEiStrategy::KrigingBeliever => 0.,
                QEiStrategy::KrigingBelieverLowerBound => -3.,
                QEiStrategy::KrigingBelieverUpperBound => 3.,
                _ => -1., // never used
            };
            Ok(pred + conf * f64::sqrt(var))
        }
    }

    fn ei(
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

    fn wb2s(
        x: &[f64],
        gpr: &GaussianProcess<ConstantMean, SquaredExponentialKernel>,
        f_min: f64,
        scale: Option<f64>,
    ) -> f64 {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        let ei = Self::ei(x, gpr, f_min);
        let scale = scale.unwrap_or(1.);
        scale * ei - gpr.predict_values(&pt).unwrap()[[0, 0]]
    }

    fn compute_wb2s_scale(
        x: &Array2<f64>,
        gpr: &GaussianProcess<ConstantMean, SquaredExponentialKernel>,
        f_min: f64,
    ) -> f64 {
        let ratio = 100.; // TODO: make it a parameter
        let ei_x = x.map_axis(Axis(1), |xi| Self::ei(&xi.as_slice().unwrap(), gpr, f_min));
        let i_max = ei_x.argmax().unwrap();
        let pred_max = gpr
            .predict_values(&x.row(i_max).insert_axis(Axis(1)))
            .unwrap()[[0, 0]];
        let ei_max = ei_x[i_max];
        if ei_max > 0. {
            ratio * pred_max.abs() / ei_max
        } else {
            1.
        }
    }

    fn _compute_obj_scale(&self, x: &Array2<f64>) -> f64 {
        *self.obj_eval(&x).mapv(|v| v.abs()).max().unwrap_or(&1.)
    }

    fn norm_cdf(x: f64) -> f64 {
        0.5 * erfc(-x / std::f64::consts::SQRT_2)
    }

    fn norm_pdf(x: f64) -> f64 {
        (-0.5 * x * x).exp() / SQRT_2PI
    }

    fn acq_eval(
        &self,
        x: &[f64],
        gpr: &GaussianProcess<ConstantMean, SquaredExponentialKernel>,
        f_min: f64,
        scale: f64,
        scale_wb2: Option<f64>,
    ) -> f64 {
        let obj = match self.acq {
            AcqStrategy::EI => -Self::ei(x, gpr, f_min),
            AcqStrategy::WB2 => -Self::wb2s(x, gpr, f_min, Some(1.)),
            AcqStrategy::WB2S => -Self::wb2s(x, gpr, f_min, scale_wb2),
        };
        obj / scale
    }

    fn obj_eval<D: Data<Elem = f64>>(&self, x: &ArrayBase<D, Ix2>) -> Array2<f64> {
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

    #[test]
    fn test_xsinx_wb2() {
        fn xsinx(x: &[f64]) -> f64 {
            (x[0] - 3.5) * f64::sin((x[0] - 3.5) / std::f64::consts::PI)
        };
        let now = Instant::now();
        let res = Ego::new(xsinx, &array![[0.0, 25.0]])
            .n_iter(10)
            .acq_strategy(AcqStrategy::WB2)
            .x_doe(&array![[0.], [7.], [25.]])
            .minimize();
        println!("xsinx optim result = {:?}", res);
        println!("Elapsed = {:?}", now.elapsed());
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 5e-1);
    }

    #[test]
    fn test_xsinx_suggestions() {
        fn xsinx(x: &[f64]) -> f64 {
            (x[0] - 3.5) * f64::sin((x[0] - 3.5) / std::f64::consts::PI)
        };
        let ego = Ego::new(xsinx, &array![[0.0, 25.0]]);

        let mut x_doe = array![[0.], [7.], [25.]];
        let mut y_doe = x_doe.mapv(|x| xsinx(&[x]));
        for _i in 0..6 {
            let x_suggested = ego.suggest(&x_doe, &y_doe);
            println!("{:?}", x_suggested);
            x_doe = stack![Axis(0), x_doe, x_suggested];
            y_doe = x_doe.mapv(|x| xsinx(&[x]));
        }

        let expected = 18.9;
        assert_abs_diff_eq!(expected, x_doe[[8, 0]], epsilon = 5e-1);
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
