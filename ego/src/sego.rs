use crate::errors::{EgoError, Result};
use crate::sort_axis::*;
use crate::types::*;
use doe::{LHSKind, SamplingMethod, LHS};
use finitediff::FiniteDiff;
use libm::erfc;
use moe::{CorrelationSpec, Moe, RegressionSpec};
use ndarray::{
    concatenate, s, Array, Array1, Array2, ArrayBase, ArrayView, Axis, Data, Ix1, Ix2, Zip,
};
use ndarray_linalg::Scalar;
use ndarray_stats::QuantileExt;
// use ndarray_npy::write_npy;
// use env_logger;
use log::debug;

use ndarray_rand::rand::{Rng, SeedableRng};
use nlopt::*;
use rand_isaac::Isaac64Rng;

pub struct Sego<O: GroupFunc, R: Rng> {
    pub n_iter: usize,
    pub n_start: usize,
    pub n_parallel: usize,
    pub n_doe: usize,
    pub n_cstr: usize,
    pub x_doe: Option<Array2<f64>>,
    pub xlimits: Array2<f64>,
    pub q_ei: QEiStrategy,
    pub infill: InfillStrategy,
    pub infill_optimizer: InfillOptimizer,
    pub regression_spec: RegressionSpec,
    pub correlation_spec: CorrelationSpec,
    pub obj: O,
    pub rng: R,
}

impl<O: GroupFunc> Sego<O, Isaac64Rng> {
    pub fn new(f: O, xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Sego<O, Isaac64Rng> {
        Self::new_with_rng(f, xlimits, Isaac64Rng::seed_from_u64(42))
    }
}

impl<O: GroupFunc, R: Rng + Clone> Sego<O, R> {
    pub fn new_with_rng(f: O, xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>, rng: R) -> Self {
        Sego {
            n_iter: 20,
            n_start: 20,
            n_parallel: 1,
            n_doe: 10,
            n_cstr: 0,
            x_doe: None,
            xlimits: xlimits.to_owned(),
            q_ei: QEiStrategy::KrigingBeliever,
            infill: InfillStrategy::WB2S,
            infill_optimizer: InfillOptimizer::Slsqp,
            regression_spec: RegressionSpec::ALL,
            correlation_spec: CorrelationSpec::ALL,
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

    pub fn n_cstr(&mut self, n_cstr: usize) -> &mut Self {
        self.n_cstr = n_cstr;
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

    pub fn infill_strategy(&mut self, infill: InfillStrategy) -> &mut Self {
        self.infill = infill;
        self
    }

    pub fn infill_optimizer(&mut self, optimizer: InfillOptimizer) -> &mut Self {
        self.infill_optimizer = optimizer;
        self
    }

    pub fn regression_spec(&mut self, regression_spec: RegressionSpec) -> &mut Self {
        self.regression_spec = regression_spec;
        self
    }

    pub fn correlation_spec(&mut self, correlation_spec: CorrelationSpec) -> &mut Self {
        self.correlation_spec = correlation_spec;
        self
    }

    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> Sego<O, R2> {
        Sego {
            n_iter: self.n_iter,
            n_start: self.n_start,
            n_parallel: self.n_parallel,
            n_doe: self.n_doe,
            n_cstr: self.n_cstr,
            x_doe: self.x_doe,
            xlimits: self.xlimits,
            q_ei: self.q_ei,
            infill: self.infill,
            infill_optimizer: self.infill_optimizer,
            regression_spec: self.regression_spec,
            correlation_spec: self.correlation_spec,
            obj: self.obj,
            rng,
        }
    }

    fn next_points(
        &self,
        _n: usize,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        sampling: &LHS<f64, R>,
    ) -> (Array2<f64>, Array2<f64>) {
        let mut x_dat = Array2::zeros((0, x_data.ncols()));
        let mut y_dat = Array2::zeros((0, y_data.ncols()));

        let obj_model = Moe::params(1)
            .set_regression_spec(self.regression_spec)
            .set_correlation_spec(self.correlation_spec)
            .fit(&x_data, &y_data.slice(s![.., 0..1]))
            .expect("GP training failure");

        let mut cstr_models: Vec<Box<Moe>> = Vec::with_capacity(self.n_cstr);
        for k in 0..self.n_cstr {
            cstr_models.push(Box::new(
                Moe::params(1)
                    .fit(&x_data, &y_data.slice(s![.., k + 1..k + 2]))
                    .expect("GP training failure"),
            ))
        }

        for _ in 0..self.n_parallel {
            match self.find_best_point(x_data, &y_data, &sampling, &obj_model, &cstr_models) {
                Ok(xk) => match self.get_virtual_point(&xk, &y_data, &obj_model, &cstr_models) {
                    Ok(yk) => {
                        y_dat = concatenate![
                            Axis(0),
                            y_dat,
                            Array2::from_shape_vec((1, 1 + self.n_cstr), yk).unwrap()
                        ];
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

    pub fn suggest(
        &self,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Array2<f64> {
        let rng = self.rng.clone();
        let sampling = LHS::new(&self.xlimits).with_rng(rng).kind(LHSKind::Maximin);
        let (x_dat, _) = self.next_points(0, &x_data, &y_data, &sampling);
        x_dat
    }

    pub fn minimize(&mut self) -> OptimResult<f64> {
        let rng = self.rng.clone();
        let sampling = LHS::new(&self.xlimits).with_rng(rng).kind(LHSKind::Maximin);

        let mut x_data = if let Some(xdoe) = &self.x_doe {
            xdoe.to_owned()
        } else {
            sampling.sample(self.n_doe)
        };

        let mut y_data = self.obj_eval(&x_data);

        for i in 0..(self.n_iter - x_data.nrows()) {
            let (x_dat, y_dat) = self.next_points(i, &x_data, &y_data, &sampling);
            y_data = concatenate![Axis(0), y_data, y_dat];
            x_data = concatenate![Axis(0), x_data, x_dat];
            let n_par = -(self.n_parallel as i32);
            let x_to_eval = x_data.slice(s![n_par.., ..]);
            let y_actual = self.obj_eval(&x_to_eval);
            Zip::from(y_data.slice_mut(s![n_par.., ..]).columns_mut())
                .and(y_actual.columns())
                .for_each(|mut y, val| y.assign(&val));
        }
        let best_index = self.find_best_result_index(&y_data);
        debug!("{:?}", concatenate![Axis(1), x_data, y_data]);
        OptimResult {
            x_opt: x_data.row(best_index).to_owned(),
            y_opt: y_data.row(best_index).to_owned(),
        }
    }

    fn find_best_point(
        &self,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        sampling: &LHS<f64, R>,
        obj_model: &Moe,
        cstr_models: &Vec<Box<Moe>>,
    ) -> Result<Array1<f64>> {
        let f_min = y_data.min().unwrap();

        let obj = |x: &[f64], gradient: Option<&mut [f64]>, params: &mut ObjData<f64>| -> f64 {
            let ObjData {
                scale_obj,
                scale_wb2,
                ..
            } = params;
            if let Some(grad) = gradient {
                let f = |x: &Vec<f64>| -> f64 {
                    self.infill_eval(x, &obj_model, *f_min, *scale_obj, *scale_wb2)
                };
                grad[..].copy_from_slice(&x.to_vec().forward_diff(&f));
            }
            self.infill_eval(x, &obj_model, *f_min, *scale_obj, *scale_wb2)
        };

        let mut cstrs: Vec<Box<dyn nlopt::ObjFn<ObjData<f64>>>> = Vec::with_capacity(self.n_cstr);
        for i in 0..self.n_cstr {
            let index = i;
            let cstr =
                move |x: &[f64], gradient: Option<&mut [f64]>, params: &mut ObjData<f64>| -> f64 {
                    if let Some(grad) = gradient {
                        let f = |x: &Vec<f64>| -> f64 {
                            cstr_models[i]
                                .predict_values(
                                    &Array::from_shape_vec((1, x.len()), x.to_vec()).unwrap(),
                                )
                                .unwrap()[[0, 0]]
                                / params.scale_cstr[index]
                        };
                        grad[..].copy_from_slice(&x.to_vec().forward_diff(&f));
                    }
                    cstr_models[index]
                        .predict_values(&Array::from_shape_vec((1, x.len()), x.to_vec()).unwrap())
                        .unwrap()[[0, 0]]
                        / params.scale_cstr[index]
                };
            cstrs.push(Box::new(cstr) as Box<dyn nlopt::ObjFn<ObjData<f64>>>);
        }

        let mut success = false;
        let mut n_optim = 1;
        let n_max_optim = 20;
        let mut best_x = None;

        let scaling_points = sampling.sample(100 * self.xlimits.nrows());
        let scale_obj = Self::compute_obj_scale(&scaling_points, &obj_model);
        let scale_cstr = Self::compute_cstr_scales(&scaling_points, &cstr_models);
        let scale_wb2 = if self.infill == InfillStrategy::WB2S {
            Self::compute_wb2s_scale(&scaling_points, &obj_model, *f_min)
        } else {
            1.
        };

        let algorithm = match self.infill_optimizer {
            InfillOptimizer::Slsqp => Algorithm::Slsqp,
            InfillOptimizer::Cobyla => Algorithm::Cobyla,
        };
        while !success && n_optim <= n_max_optim {
            let mut optimizer = Nlopt::new(
                algorithm,
                x_data.ncols(),
                obj,
                Target::Minimize,
                ObjData {
                    scale_obj,
                    scale_wb2: scale_wb2,
                    scale_cstr: scale_cstr.to_owned(),
                },
            );
            let lower = self.xlimits.column(0).to_owned();
            optimizer.set_lower_bounds(&lower.as_slice().unwrap())?;
            let upper = self.xlimits.column(1).to_owned();
            optimizer.set_upper_bounds(&upper.as_slice().unwrap())?;
            optimizer.set_maxeval(200)?;
            optimizer.set_ftol_rel(1e-4)?;
            optimizer.set_ftol_abs(1e-4)?;
            cstrs.iter().enumerate().for_each(|(i, cstr)| {
                optimizer
                    .add_inequality_constraint(
                        cstr,
                        ObjData {
                            scale_obj,
                            scale_wb2,
                            scale_cstr: scale_cstr.to_owned(),
                        },
                        1e-6 / scale_cstr[i],
                    )
                    .unwrap();
            });

            let mut best_opt = f64::INFINITY;
            let x_start = sampling.sample(self.n_start);

            for i in 0..self.n_start {
                let mut x_opt = x_start.row(i).to_vec();
                match optimizer.optimize(&mut x_opt) {
                    Ok((_, opt)) => {
                        if opt < best_opt {
                            best_opt = opt;
                            let res = x_opt.iter().map(|v| *v).collect::<Vec<f64>>();
                            best_x = Some(Array::from(res));
                            success = true;
                        }
                    }
                    Err((_, _)) => {}
                }
            }
            n_optim += 1;
        }
        best_x.ok_or_else(|| EgoError::EgoError(String::from("Can not find best point")))
    }

    fn find_best_result_index(&self, y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> usize {
        if self.n_cstr > 0 {
            let mut index = 0;
            let perm = y_data.sort_axis_by(Axis(0), |i, j| y_data[[i, 0]] < y_data[[j, 0]]);
            let y_sort = y_data.to_owned().permute_axis(Axis(0), &perm);
            for (i, row) in y_sort.axis_iter(Axis(0)).enumerate() {
                if row
                    .slice(s![1..])
                    .iter()
                    .filter(|v| *v > &1e-6)
                    .collect::<Vec<&f64>>()
                    .len()
                    == 0
                {
                    index = i;
                    break;
                }
            }
            perm.indices[index]
        } else {
            y_data.column(0).argmin().unwrap()
        }
    }

    fn get_virtual_point(
        &self,
        xk: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        obj_model: &Moe,
        cstr_models: &Vec<Box<Moe>>,
    ) -> Result<Vec<f64>> {
        let mut res: Vec<f64> = Vec::with_capacity(3);
        if self.q_ei == QEiStrategy::ConstantLiarMinimum {
            let index_min = y_data.slice(s![.., 0]).argmin().unwrap();
            res.push(y_data[[index_min, 0]]);
            for ic in 1..=self.n_cstr {
                res.push(y_data[[index_min, ic]]);
            }
            Ok(res)
        } else {
            let x = &xk.to_owned().insert_axis(Axis(0));
            let pred = obj_model.predict_values(&x)?[[0, 0]];
            let var = obj_model.predict_variances(&x)?[[0, 0]];
            let conf = match self.q_ei {
                QEiStrategy::KrigingBeliever => 0.,
                QEiStrategy::KrigingBelieverLowerBound => -3.,
                QEiStrategy::KrigingBelieverUpperBound => 3.,
                _ => -1., // never used
            };
            res.push(pred + conf * Scalar::sqrt(var));
            for ic in 0..self.n_cstr {
                res.push(cstr_models[ic].predict_values(&x)?[[0, 0]]);
            }
            Ok(res)
        }
    }

    fn ei(x: &[f64], obj_model: &Moe, f_min: f64) -> f64 {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap().to_owned();
        if let Ok(p) = obj_model.predict_values(&pt) {
            if let Ok(s) = obj_model.predict_variances(&pt) {
                let pred = p[[0, 0]];
                let sigma = Scalar::sqrt(s[[0, 0]]);
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

    fn wb2s(x: &[f64], obj_model: &Moe, f_min: f64, scale: f64) -> f64 {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap().to_owned();
        let ei = Self::ei(x, obj_model, f_min);
        scale * ei - obj_model.predict_values(&pt).unwrap()[[0, 0]]
    }

    fn compute_wb2s_scale(x: &Array2<f64>, obj_model: &Moe, f_min: f64) -> f64 {
        let ratio = 100.; // TODO: make it a parameter
        let ei_x = x.map_axis(Axis(1), |xi| {
            let ei = Self::ei(xi.as_slice().unwrap(), obj_model, f_min);
            ei
        });
        let i_max = ei_x.argmax().unwrap();
        let pred_max = obj_model
            .predict_values(&x.row(i_max).insert_axis(Axis(0)).to_owned())
            .unwrap()[[0, 0]];
        let ei_max = ei_x[i_max];
        if ei_max > 0. {
            ratio * pred_max / ei_max
        } else {
            1.
        }
    }

    fn compute_obj_scale(x: &Array2<f64>, obj_model: &Moe) -> f64 {
        let preds = obj_model.predict_values(x).unwrap().mapv(|v| f64::abs(v));
        *preds.max().unwrap_or(&1.0)
    }

    fn compute_cstr_scales(x: &Array2<f64>, cstr_models: &Vec<Box<Moe>>) -> Array1<f64> {
        let scales: Vec<f64> = cstr_models
            .iter()
            .map(|cstr_model| {
                let preds = cstr_model.predict_values(x).unwrap().mapv(|v| f64::abs(v));
                *preds.max().unwrap_or(&1.0)
            })
            .collect();
        Array1::from_shape_vec(cstr_models.len(), scales).unwrap()
    }

    fn norm_cdf(x: f64) -> f64 {
        let norm = 0.5 * erfc(-x / std::f64::consts::SQRT_2);
        norm
    }

    fn norm_pdf(x: f64) -> f64 {
        Scalar::exp(-0.5 * x * x) / SQRT_2PI
    }

    fn infill_eval(
        &self,
        x: &[f64],
        obj_model: &Moe,
        f_min: f64,
        scale: f64,
        scale_wb2: f64,
    ) -> f64 {
        let x_f = x.iter().map(|v| *v).collect::<Vec<f64>>();
        let obj = match self.infill {
            InfillStrategy::EI => -Self::ei(&x_f, obj_model, f_min),
            InfillStrategy::WB2 => -Self::wb2s(&x_f, obj_model, f_min, 1.),
            InfillStrategy::WB2S => -Self::wb2s(&x_f, obj_model, f_min, scale_wb2),
        };
        obj / scale
    }

    fn obj_eval(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        (&self.obj)(&x.view())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use argmin_testfunctions::rosenbrock;
    use ndarray::{array, ArrayView2};
    use std::time::Instant;

    fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
        (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
    }

    #[test]
    fn test_xsinx_ei_sego() {
        let res = Sego::new(xsinx, &array![[0.0, 25.0]])
            .infill_strategy(InfillStrategy::EI)
            .regression_spec(RegressionSpec::QUADRATIC)
            .correlation_spec(CorrelationSpec::ALL)
            .n_iter(10)
            .x_doe(&array![[0.], [7.], [25.]])
            .minimize();
        let expected = array![-15.1];
        assert_abs_diff_eq!(expected, res.y_opt, epsilon = 0.3);
    }

    #[test]
    fn test_xsinx_wb2() {
        let res = Sego::new(xsinx, &array![[0.0, 25.0]])
            .infill_strategy(InfillStrategy::WB2)
            .minimize();
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    #[test]
    fn test_xsinx_suggestions() {
        let mut ego = Sego::new(xsinx, &array![[0.0, 25.0]]);
        let ego = ego.infill_strategy(InfillStrategy::EI);

        let mut x_doe = array![[0.], [7.], [20.], [25.]];
        let mut y_doe = xsinx(&x_doe.view());
        for _i in 0..10 {
            let x_suggested = ego.suggest(&x_doe, &y_doe);

            x_doe = concatenate![Axis(0), x_doe, x_suggested];
            y_doe = xsinx(&x_doe.view());
        }

        let expected = -15.1;
        let y_opt = y_doe.min().unwrap();
        assert_abs_diff_eq!(expected, *y_opt, epsilon = 1e-1);
    }

    fn rosenb(x: &ArrayView2<f64>) -> Array2<f64> {
        let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
        Zip::from(y.rows_mut())
            .and(x.rows())
            .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec(), 1., 100.)]));
        y
    }

    #[test]
    fn test_rosenbrock_2d() {
        let now = Instant::now();
        let xlimits = array![[-2., 2.], [-2., 2.]];
        let doe = LHS::new(&xlimits).sample(10);
        let res = Sego::new(rosenb, &xlimits)
            .infill_strategy(InfillStrategy::EI)
            .x_doe(&doe)
            .n_iter(30)
            .minimize();
        println!("Rosenbrock optim result = {:?}", res);
        println!("Elapsed = {:?}", now.elapsed());
        let expected = array![1., 1.];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 6e-2);
    }

    // Objective
    fn g24(x: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> f64 {
        // Function G24: 1 global optimum y_opt = -5.5080 at x_opt =(2.3295, 3.1785)
        -x[0] - x[1]
    }

    // Constraints < 0
    fn g24_c1(x: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> f64 {
        -2.0 * x[0].powf(4.0) + 8.0 * x[0].powf(3.0) - 8.0 * x[0].powf(2.0) + x[1] - 2.0
    }

    fn g24_c2(x: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> f64 {
        -4.0 * x[0].powf(4.0) + 32.0 * x[0].powf(3.0) - 88.0 * x[0].powf(2.0) + 96.0 * x[0] + x[1]
            - 36.0
    }

    fn f_g24(x: &ArrayView2<f64>) -> Array2<f64> {
        let mut y = Array2::zeros((x.nrows(), 3));
        Zip::from(y.rows_mut())
            .and(x.rows())
            .for_each(|mut yi, xi| {
                yi.assign(&array![g24(&xi), g24_c1(&xi), g24_c2(&xi)]);
            });
        y
    }

    #[test]
    fn test_sego_g24() {
        let x = array![[1., 2.]];
        println!("{:?}", f_g24(&x.view()));
        let xlimits = array![[0., 3.], [0., 4.]];
        let doe = LHS::new(&xlimits).sample(10);
        let res = Sego::new(f_g24, &xlimits)
            .n_cstr(2)
            .infill_strategy(InfillStrategy::EI)
            .infill_optimizer(InfillOptimizer::Cobyla) // test passes also with WB2S and Slsqp
            .x_doe(&doe)
            .n_iter(30)
            .minimize();
        println!("G24 optim result = {:?}", res);
        let expected = array![2.3295, 3.1785];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-2);
    }
}
