use crate::errors::{EgoboxError, Result};
use doe::{LHSKind, SamplingMethod, LHS};
use finitediff::FiniteDiff;
use gp::{ConstantMean, GaussianProcess, SquaredExponentialKernel};
use libm::erfc;
use linfa_pls::Float;
use ndarray::{concatenate, s, Array, Array1, Array2, ArrayBase, ArrayView, Axis, Data, Ix2, Zip};
use ndarray_linalg::{Lapack, Scalar};
use ndarray_npy::write_npy;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_stats::QuantileExt;
use nlopt::*;
use rand_isaac::Isaac64Rng;

/// Add Scalar and Lapack trait bounds to the common Float trait
// pub trait Float: linfa_pls::Float {}
// impl Float for f32 {}
// impl Float for f64 {}

const SQRT_2PI: f64 = 2.5066282746310007;

pub trait ObjFn: Send + Sync + 'static + Fn(&[f64]) -> f64 {}
impl<T> ObjFn for T where T: Send + Sync + 'static + Fn(&[f64]) -> f64 {}

#[derive(Debug)]
pub struct OptimResult<F: Float> {
    x_opt: Array1<F>,
    y_opt: F,
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
struct ObjData<F> {
    scale: F,
    scale_wb2: Option<F>,
}

pub struct Ego<F: Float, O: ObjFn, R: Rng> {
    pub n_iter: usize,
    pub n_start: usize,
    pub n_parallel: usize,
    pub n_doe: usize,
    pub x_doe: Option<Array2<F>>,
    pub xlimits: Array2<F>,
    pub q_ei: QEiStrategy,
    pub acq: AcqStrategy,
    pub obj: O,
    pub rng: R,
}

impl<F: Float, O: ObjFn> Ego<F, O, Isaac64Rng> {
    pub fn new(f: O, xlimits: &Array2<F>) -> Ego<F, O, Isaac64Rng> {
        Self::new_with_rng(f, &xlimits, Isaac64Rng::seed_from_u64(42))
    }
}

impl<F: Float, O: ObjFn, R: Rng + Clone> Ego<F, O, R> {
    pub fn new_with_rng(f: O, xlimits: &Array2<F>, rng: R) -> Self {
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

    pub fn x_doe(&mut self, x_doe: &Array2<F>) -> &mut Self {
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

    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> Ego<F, O, R2> {
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
        n: usize,
        x_data: &Array2<F>,
        y_data: &Array2<F>,
        sampling: &LHS<F, R>,
    ) -> (Array2<F>, Array2<F>) {
        let mut x_dat = Array2::zeros((0, x_data.ncols()));
        let mut y_dat = Array2::zeros((0, y_data.ncols()));
        for _ in 0..self.n_parallel {
            let gpr = GaussianProcess::<F, ConstantMean, SquaredExponentialKernel>::params(
                ConstantMean::default(),
                SquaredExponentialKernel::default(),
            )
            .fit(&x_data, &y_data)
            .expect("GP training failure");

            if n == 6 {
                let f_min = y_data.min().unwrap();
                let xplot = Array::linspace(0., 25., 100)
                    .mapv(F::cast)
                    .insert_axis(Axis(1));
                let obj = self.obj_eval(&xplot);
                let gpr_vals = gpr.predict_values(&xplot).unwrap();
                let gpr_vars = gpr.predict_variances(&xplot).unwrap();
                let ei = xplot.map(|x| Self::ei(&[*x], &gpr, *f_min));
                let wb2 = xplot.map(|x| Self::wb2s(&[*x], &gpr, *f_min, None));
                // write_npy("ego_x.npy", &xplot).expect("xplot saved");
                // write_npy("ego_obj.npy", &obj).expect("obj saved");
                // write_npy("ego_gpr.npy", &gpr_vals).expect("gp vals saved");
                // write_npy("ego_gpr_vars.npy", &gpr_vars).expect("gp vars saved");
                // write_npy("ego_ei.npy", &ei).expect("ei saved");
                // write_npy("ego_wb2.npy", &wb2).expect("wb2 saved");
            }

            match self.find_best_point(&x_data, &y_data, &sampling, &gpr) {
                Ok(xk) => match self.get_virtual_point(&xk, &y_data, &gpr) {
                    Ok(yk) => {
                        y_dat = concatenate![Axis(0), y_dat, Array2::from_elem((1, 1), yk)];
                        x_dat = concatenate![Axis(0), x_dat, xk.insert_axis(Axis(0))];
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

    pub fn suggest(&self, x_data: &Array2<F>, y_data: &Array2<F>) -> Array2<F> {
        let rng = self.rng.clone();
        let sampling = LHS::new(&self.xlimits).with_rng(rng).kind(LHSKind::Maximin);
        let (x_dat, _) = self.next_points(0, &x_data, &y_data, &sampling);
        x_dat
    }

    pub fn minimize(&mut self) -> OptimResult<F> {
        let rng = self.rng.clone();
        let sampling = LHS::new(&self.xlimits).with_rng(rng).kind(LHSKind::Maximin);

        let mut x_data = if let Some(xdoe) = &self.x_doe {
            xdoe.to_owned()
        } else {
            sampling.sample(self.n_doe)
        };

        let mut y_data = self.obj_eval(&x_data);

        for i in 0..self.n_iter {
            let (x_dat, y_dat) = self.next_points(i, &x_data, &y_data, &sampling);
            y_data = concatenate![Axis(0), y_data, y_dat];
            x_data = concatenate![Axis(0), x_data, x_dat];
            let n_par = -(self.n_parallel as i32);
            let x_to_eval = x_data.slice(s![n_par.., ..]);
            let y_actual = self.obj_eval(&x_to_eval);
            Zip::from(y_data.slice_mut(s![n_par.., ..]).columns_mut())
                .and(y_actual.columns())
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
        x_data: &Array2<F>,
        y_data: &Array2<F>,
        sampling: &LHS<F, R>,
        gpr: &GaussianProcess<F, ConstantMean, SquaredExponentialKernel>,
    ) -> Result<Array1<F>> {
        let f_min = y_data.min().unwrap();

        let obj = |x: &[f64], gradient: Option<&mut [f64]>, params: &mut ObjData<F>| -> f64 {
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
            let lower = to_vec_f64(self.xlimits.column(0).as_slice().unwrap());
            optimizer.set_lower_bounds(&lower).unwrap();
            let upper = to_vec_f64(self.xlimits.column(1).as_slice().unwrap());
            optimizer.set_upper_bounds(&upper).unwrap();
            optimizer.set_maxeval(200).unwrap();
            optimizer.set_ftol_rel(1e-4).unwrap();
            optimizer.set_ftol_abs(1e-4).unwrap();

            let mut best_opt = f64::INFINITY;
            let x_start = sampling.sample(self.n_start);

            for i in 0..self.n_start {
                let mut x_opt = to_vec_f64(x_start.row(i).as_slice().unwrap());
                match optimizer.optimize(&mut x_opt) {
                    Ok((_, opt)) => {
                        if opt < best_opt {
                            best_opt = opt;
                            let res = x_opt.iter().map(|v| F::cast(*v)).collect::<Vec<F>>();
                            best_x = Some(Array::from(res));
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
        xk: &Array1<F>,
        y_data: &Array2<F>,
        gpr: &GaussianProcess<F, ConstantMean, SquaredExponentialKernel>,
    ) -> Result<F> {
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
            Ok(pred + F::cast(conf) * Scalar::sqrt(var))
        }
    }

    fn ei(
        x: &[F],
        gpr: &GaussianProcess<F, ConstantMean, SquaredExponentialKernel>,
        f_min: F,
    ) -> F {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        if let Ok(p) = gpr.predict_values(&pt) {
            println!("P = {:?}", p);
            if let Ok(s) = gpr.predict_variances(&pt) {
                let pred = p[[0, 0]];
                let sigma = Scalar::sqrt(s[[0, 0]]);
                let args0 = (f_min - pred) / sigma;
                let args1 = (f_min - pred) * Self::norm_cdf(args0);
                let args2 = sigma * Self::norm_pdf(args0);
                args1 + args2
            } else {
                -F::cast(f32::INFINITY)
            }
        } else {
            -F::cast(f32::INFINITY)
        }
    }

    fn wb2s(
        x: &[F],
        gpr: &GaussianProcess<F, ConstantMean, SquaredExponentialKernel>,
        f_min: F,
        scale: Option<F>,
    ) -> F {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        let ei = Self::ei(x, gpr, f_min);
        let scale = scale.unwrap_or_else(F::one);
        scale * ei - gpr.predict_values(&pt).unwrap()[[0, 0]]
    }

    fn compute_wb2s_scale(
        x: &Array2<F>,
        gpr: &GaussianProcess<F, ConstantMean, SquaredExponentialKernel>,
        f_min: F,
    ) -> F {
        let ratio = F::cast(100.); // TODO: make it a parameter
        let ei_x = x.map_axis(Axis(1), |xi| {
            let ei = Self::ei(xi.as_slice().unwrap(), gpr, f_min);
            ei
        });
        let i_max = ei_x.argmax().unwrap();
        let pred_max = gpr
            .predict_values(&x.row(i_max).insert_axis(Axis(1)))
            .unwrap()[[0, 0]];
        let ei_max = ei_x[i_max];
        if ei_max > F::zero() {
            ratio * F::cast(pred_max) / ei_max
        } else {
            F::cast(1.)
        }
    }

    fn _compute_obj_scale(&self, x: &Array2<F>) -> F {
        *self
            .obj_eval(&x)
            .mapv(|v| F::cast(Scalar::abs(v)))
            .max()
            .unwrap_or(&F::cast(1.))
    }

    fn norm_cdf(x: F) -> F {
        dbg!(to_f64(x));
        let norm = F::cast(0.5 * erfc(-to_f64(x) / std::f64::consts::SQRT_2));
        dbg!(norm);
        norm
    }

    fn norm_pdf(x: F) -> F {
        Scalar::exp(-F::cast(0.5) * x * x) / F::cast(SQRT_2PI)
    }

    fn acq_eval(
        &self,
        x: &[f64],
        gpr: &GaussianProcess<F, ConstantMean, SquaredExponentialKernel>,
        f_min: F,
        scale: F,
        scale_wb2: Option<F>,
    ) -> f64 {
        let x_f = x.iter().map(|v| F::cast(*v)).collect::<Vec<F>>();
        let obj = match self.acq {
            AcqStrategy::EI => -Self::ei(&x_f, gpr, f_min),
            AcqStrategy::WB2 => -Self::wb2s(&x_f, gpr, f_min, Some(F::one())),
            AcqStrategy::WB2S => -Self::wb2s(&x_f, gpr, f_min, scale_wb2),
        };
        to_f64(obj / scale)
    }

    fn obj_eval<D: Data<Elem = F>>(&self, x: &ArrayBase<D, Ix2>) -> Array2<F> {
        let mut y = Array1::zeros(x.nrows());
        let obj = &self.obj;
        Zip::from(&mut y).and(x.rows()).par_apply(|yn, xn| {
            let val = to_vec_f64(xn.to_slice().unwrap());
            *yn = F::cast((obj)(&val))
        });
        y.insert_axis(Axis(1))
    }
}

fn to_f64<F: Float>(a: F) -> f64 {
    unsafe { std::ptr::read(&a as *const F as *const f64) }
}

fn to_vec_f64<F: Float>(v: &[F]) -> Vec<f64> {
    v.to_owned()
        .iter()
        .map(|v| to_f64(*v))
        .collect::<Vec<f64>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // use argmin_testfunctions::rosenbrock;
    use ndarray::array;

    fn xsinx(x: &[f64]) -> f64 {
        (x[0] - 3.5) * f64::sin((x[0] - 3.5) / std::f64::consts::PI)
    }

    #[test]
    fn test_xsinx_ei() {
        let res = Ego::new(xsinx, &array![[0.0, 25.0]])
            .n_iter(10)
            .x_doe(&array![[0.], [7.], [25.]])
            .minimize();
        let expected = -15.1;
        assert_abs_diff_eq!(expected, res.y_opt, epsilon = 0.3);
    }

    #[test]
    fn test_xsinx_wb2() {
        let res = Ego::new(xsinx, &array![[0.0, 25.0]])
            .acq_strategy(AcqStrategy::WB2)
            .minimize();
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    #[test]
    fn test_xsinx_suggestions() {
        let ego = Ego::new(xsinx, &array![[0.0, 25.0]]);

        let mut x_doe = array![[0.], [7.], [20.], [25.]];
        let mut y_doe = x_doe.mapv(|x| xsinx(&[x]));
        for _i in 0..10 {
            let x_suggested = ego.suggest(&x_doe, &y_doe);

            x_doe = concatenate![Axis(0), x_doe, x_suggested];
            y_doe = x_doe.mapv(|x| xsinx(&[x]));
        }

        let expected = -15.1;
        let y_opt = y_doe.min().unwrap();
        assert_abs_diff_eq!(expected, *y_opt, epsilon = 1e-1);
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
