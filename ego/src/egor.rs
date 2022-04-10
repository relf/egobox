//! Egor optimizer implements EGO algorithm with basic handling of constraints.
//!
//! ```no_run
//! # use ndarray::{array, Array2, ArrayView1, ArrayView2, Zip};
//! # use egobox_doe::{Lhs, SamplingMethod};
//! # use egobox_ego::{ApproxValue, Egor, InfillStrategy, InfillOptimizer};
//! # use rand_isaac::Isaac64Rng;
//! # use ndarray_rand::rand::SeedableRng;
//! use argmin_testfunctions::rosenbrock;
//!
//! // Rosenbrock test function: minimum y_opt = 0 at x_opt = (1, 1)
//! fn rosenb(x: &ArrayView2<f64>) -> Array2<f64> {
//!     let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
//!     Zip::from(y.rows_mut())
//!         .and(x.rows())
//!         .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec(), 1., 100.)]));
//!     y
//! }
//!
//! let xlimits = array![[-2., 2.], [-2., 2.]];
//! let res = Egor::new(rosenb, &xlimits)
//!     .infill_strategy(InfillStrategy::EI)
//!     .n_doe(10)
//!     .expect(Some(ApproxValue {  // Known solution, the algo exits if reached
//!         value: 0.0,
//!         tolerance: 1e-1,
//!     }))
//!     .n_eval(30)
//!     .minimize()
//!     .expect("Rosenbrock minimization");
//! println!("Rosenbrock min result = {:?}", res);
//! ```
//!
//! Constraints are expected to be evaluated with the objective function
//! meaning that the function passed to the optimizer has to return
//! a vector consisting of [obj, cstr_1, ..., cstr_n] and the cstr values
//! are intended to be negative at the end of the optimization.
//! Constraint number should be declared with `n_cstr` setter.
//! A tolerance can be adjust with `cstr_tol` setter for relaxing constraint violation
//! if specified cstr values should be < `cstr_tol` (instead of < 0)
//!
//! ```no_run
//! # use ndarray::{array, Array2, ArrayView1, ArrayView2, Zip};
//! # use egobox_doe::{Lhs, SamplingMethod};
//! # use egobox_ego::{ApproxValue, Egor, InfillStrategy, InfillOptimizer};
//! # use rand_isaac::Isaac64Rng;
//! # use ndarray_rand::rand::SeedableRng;
//!
//! // Function G24: 1 global optimum y_opt = -5.5080 at x_opt =(2.3295, 3.1785)
//! fn g24(x: &ArrayView1<f64>) -> f64 {
//!    -x[0] - x[1]
//! }
//!
//! // Constraints < 0
//! fn g24_c1(x: &ArrayView1<f64>) -> f64 {
//!     -2.0 * x[0].powf(4.0) + 8.0 * x[0].powf(3.0) - 8.0 * x[0].powf(2.0) + x[1] - 2.0
//! }
//!
//! fn g24_c2(x: &ArrayView1<f64>) -> f64 {
//!     -4.0 * x[0].powf(4.0) + 32.0 * x[0].powf(3.0)
//!     - 88.0 * x[0].powf(2.0) + 96.0 * x[0] + x[1]
//!     - 36.0
//! }
//!
//! // Gouped function : objective + constraints
//! fn f_g24(x: &ArrayView2<f64>) -> Array2<f64> {
//!     let mut y = Array2::zeros((x.nrows(), 3));
//!     Zip::from(y.rows_mut())
//!         .and(x.rows())
//!         .for_each(|mut yi, xi| {
//!             yi.assign(&array![g24(&xi), g24_c1(&xi), g24_c2(&xi)]);
//!         });
//!     y
//! }
//!
//! let xlimits = array![[0., 3.], [0., 4.]];
//! let doe = Lhs::new(&xlimits).sample(10);
//! let res = Egor::new(f_g24, &xlimits)
//!            .n_cstr(2)
//!            .infill_strategy(InfillStrategy::EI)
//!            .infill_optimizer(InfillOptimizer::Cobyla)
//!            .doe(Some(doe))
//!            .n_eval(40)
//!            .expect(Some(ApproxValue {  
//!               value: -5.5080,
//!               tolerance: 1e-3,
//!            }))
//!            .minimize()
//!            .expect("g24 minimized");
//! println!("G24 min result = {:?}", res);
//! ```
//!
use crate::errors::{EgoError, Result};
use crate::sort_axis::*;
use crate::types::*;
use crate::utils::update_data;
use crate::utils::{compute_cstr_scales, compute_obj_scale, compute_wb2s_scale};
use crate::utils::{ei, wb2s};
use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use egobox_moe::{CorrelationSpec, Moe, MoeFit, MoePredict, RegressionSpec};
use env_logger::{Builder, Env};
use finitediff::FiniteDiff;
use log::{debug, info};
use ndarray::{concatenate, s, Array, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip};
use ndarray_linalg::Scalar;
use ndarray_npy::{read_npy, write_npy};
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_stats::QuantileExt;
use nlopt::*;
use rand_isaac::Isaac64Rng;

const DOE_INITIAL_FILE: &str = "egor_initial_doe.npy";
const DOE_FILE: &str = "egobox_doe.npy";

#[derive(Clone, Copy, Debug)]
pub struct ApproxValue {
    pub value: f64,
    pub tolerance: f64,
}

/// Data used by internal infill criteria to be optimized using NlOpt
pub struct ObjData<F> {
    pub scale_obj: F,
    pub scale_cstr: Array1<F>,
    pub scale_wb2: F,
}

/// An interface for "function under optimization" evaluation
pub trait Evaluator {
    fn eval(&self, x: &Array2<f64>) -> Array2<f64>;
}

/// EGO optimization parameterization
#[derive(Clone)]
pub struct Egor<'a, O: GroupFunc, R: Rng> {
    /// Number of function evaluations allocated to find the optimum
    /// Note: if the initial doe has to be evaluated, doe size should be substract.  
    pub n_eval: usize,
    /// Number of starts for multistart approach used for hyperparameters optimization
    pub n_start: usize,
    /// Number of parallel points evaluated for each "function evaluation"
    pub n_parallel: usize,
    /// Number of initial doe drawn using Latin hypercube sampling
    /// Note: n_doe > 0; otherwise n_doe = max(xdim+1, 5)
    pub n_doe: usize,
    /// Number of Constraints
    /// Note: dim function ouput = 1 objective + n_cstr constraints
    pub n_cstr: usize,
    /// Constraints violation tolerance meaning cstr < cstr_tol is considered valid
    pub cstr_tol: f64,
    /// Initial doe can be either \[x\] with x inputs only or an evaluated doe \[x, y\]
    /// Note: x dimension is determined using xlimits
    pub doe: Option<Array2<f64>>,
    /// Matrix (nx, 2) of [lower bound, upper bound] of the nx components of x
    pub xlimits: Array2<f64>,
    /// Parallel strategy used to define several points (n_parallel) evaluations at each iteration
    pub q_ei: QEiStrategy,
    /// Criterium to select next point to evaluate
    pub infill: InfillStrategy,
    /// The optimizer used to optimize infill criterium
    pub infill_optimizer: InfillOptimizer,
    /// Regression specification for GP models used by mixture of experts (see [egobox_moe])
    pub regression_spec: RegressionSpec,
    /// Correlation specification for GP models used by mixture of experts (see [egobox_moe])
    pub correlation_spec: CorrelationSpec,
    /// Optional dimension reduction (see [egobox_moe])
    pub kpls_dim: Option<usize>,
    /// Number of clusters used by mixture of experts (see [egobox_moe])
    pub n_clusters: Option<usize>,
    /// Specification of an expected solution which is used to stop the algorithm once reached
    pub expected: Option<ApproxValue>,
    /// Directory to save intermediate results: inital doe + evalutions at each iteration
    pub outdir: Option<String>,
    /// If true use <outdir> to retrieve and start from previous results
    pub hot_start: bool,
    /// MoE parameters (see [egobox_moe])
    /// Note: if specified takes precedence over individual settings
    pub moe_params: Option<&'a dyn MoeFit>,
    /// An evaluator used to run the function under optimization
    pub evaluator: Option<&'a dyn Evaluator>,
    /// The function under optimization f(x) = [objective, cstr1, ..., cstrn], (n_cstr+1 size)
    pub obj: O,
    /// A random generator used to get reproductible results.
    /// For instance: Isaac64Rng::from_u64_seed(42) for reproducibility
    pub rng: R,
}

impl<'a, O: GroupFunc> Egor<'a, O, Isaac64Rng> {
    pub fn new(f: O, xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Egor<'a, O, Isaac64Rng> {
        Self::new_with_rng(f, xlimits, Isaac64Rng::from_entropy())
    }
}

impl<'a, O: GroupFunc, R: Rng + Clone> Egor<'a, O, R> {
    pub fn new_with_rng(f: O, xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>, rng: R) -> Self {
        let env = Env::new().filter_or("EGOBOX_LOG", "info");
        let mut builder = Builder::from_env(env);
        let builder = builder.target(env_logger::Target::Stdout);
        builder.try_init().ok();
        Egor {
            n_eval: 20,
            n_start: 20,
            n_parallel: 1,
            n_doe: 0,
            n_cstr: 0,
            cstr_tol: 1e-6,
            doe: None,
            xlimits: xlimits.to_owned(),
            q_ei: QEiStrategy::KrigingBeliever,
            infill: InfillStrategy::WB2,
            infill_optimizer: InfillOptimizer::Slsqp,
            regression_spec: RegressionSpec::ALL,
            correlation_spec: CorrelationSpec::ALL,
            kpls_dim: None,
            n_clusters: Some(1),
            expected: None,
            outdir: None,
            hot_start: false,
            moe_params: None,
            evaluator: None,
            obj: f,
            rng,
        }
    }

    pub fn n_eval(&mut self, n_eval: usize) -> &mut Self {
        self.n_eval = n_eval;
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

    pub fn cstr_tol(&mut self, tol: f64) -> &mut Self {
        self.cstr_tol = tol;
        self
    }

    pub fn doe(&mut self, doe: Option<Array2<f64>>) -> &mut Self {
        self.doe = doe.map(|x| x.to_owned());
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

    pub fn kpls_dim(&mut self, kpls_dim: Option<usize>) -> &mut Self {
        self.kpls_dim = kpls_dim;
        self
    }

    pub fn n_clusters(&mut self, n_clusters: Option<usize>) -> &mut Self {
        self.n_clusters = n_clusters;
        self
    }

    pub fn expect(&mut self, expected: Option<ApproxValue>) -> &mut Self {
        self.expected = expected;
        self
    }

    pub fn outdir(&mut self, outdir: Option<String>) -> &mut Self {
        self.outdir = outdir;
        self
    }

    pub fn hot_start(&mut self, hot_start: bool) -> &mut Self {
        self.hot_start = hot_start;
        self
    }

    pub fn moe_params(&mut self, moe_params: Option<&'a dyn MoeFit>) -> &mut Self {
        self.moe_params = moe_params;
        self
    }

    pub fn evaluator(&mut self, evaluator: Option<&'a dyn Evaluator>) -> &mut Self {
        self.evaluator = evaluator;
        self
    }

    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> Egor<'a, O, R2> {
        Egor {
            n_eval: self.n_eval,
            n_start: self.n_start,
            n_parallel: self.n_parallel,
            n_doe: self.n_doe,
            n_cstr: self.n_cstr,
            cstr_tol: self.cstr_tol,
            doe: self.doe,
            xlimits: self.xlimits,
            q_ei: self.q_ei,
            infill: self.infill,
            infill_optimizer: self.infill_optimizer,
            regression_spec: self.regression_spec,
            correlation_spec: self.correlation_spec,
            kpls_dim: self.kpls_dim,
            n_clusters: self.n_clusters,
            expected: self.expected,
            outdir: self.outdir,
            hot_start: self.hot_start,
            moe_params: self.moe_params,
            evaluator: self.evaluator,
            obj: self.obj,
            rng,
        }
    }

    /// Given an evaluated doe (x, y) data, return the next promising x point
    /// where optimum may occurs regarding the infill criterium.
    /// This function inverse the control of the optimization and can used
    /// ask-and-tell interface to the EGO optmizer.
    pub fn suggest(&self, x_data: &Array2<f64>, y_data: &Array2<f64>) -> Array2<f64> {
        let rng = self.rng.clone();
        let sampling = Lhs::new(&self.xlimits).with_rng(rng).kind(LhsKind::Maximin);
        let (x_dat, _) = self.next_points(0, x_data, y_data, &sampling);
        x_dat
    }

    /// Minimize using EGO algorithm
    pub fn minimize(&self) -> Result<OptimResult<f64>> {
        let rng = self.rng.clone();
        let sampling = Lhs::new(&self.xlimits).with_rng(rng).kind(LhsKind::Maximin);

        let hstart_doe: Option<Array2<f64>> = if self.hot_start && self.outdir.is_some() {
            let path: &String = self.outdir.as_ref().unwrap();
            let filepath = std::path::Path::new(&path).join(DOE_FILE);
            if filepath.is_file() {
                info!("Reading DOE from {:?}", filepath);
                Some(read_npy(filepath)?)
            } else if std::path::Path::new(&path).join(DOE_INITIAL_FILE).is_file() {
                let filepath = std::path::Path::new(&path).join(DOE_INITIAL_FILE);
                info!("Reading DOE from {:?}", filepath);
                Some(read_npy(filepath)?)
            } else {
                None
            }
        } else {
            None
        };

        let doe = hstart_doe.as_ref().or(self.doe.as_ref());

        let (mut y_data, mut x_data, n_iter) = if let Some(doe) = doe {
            if doe.ncols() == self.xlimits.nrows() {
                // only x are specified
                info!("Compute initial DOE on specified {} points", doe.nrows());
                (
                    self.eval(doe),
                    doe.to_owned(),
                    self.n_eval.saturating_sub(doe.nrows()),
                )
            } else {
                // split doe in x and y
                info!("Use specified DOE {} samples", doe.nrows());
                (
                    doe.slice(s![.., self.xlimits.nrows()..]).to_owned(),
                    doe.slice(s![.., ..self.xlimits.nrows()]).to_owned(),
                    self.n_eval.saturating_sub(doe.nrows()),
                )
            }
        } else {
            let n_doe = if self.n_doe == 0 {
                (self.xlimits.nrows() + 1).max(5)
            } else {
                self.n_doe
            };
            info!("Compute initial LHS with {} points", n_doe);
            let x = sampling.sample(n_doe);
            (self.eval(&x), x, self.n_eval - n_doe)
        };
        let doe = concatenate![Axis(1), x_data, y_data];
        if self.outdir.is_some() {
            let path = self.outdir.as_ref().unwrap();
            std::fs::create_dir_all(path)?;
            let filepath = std::path::Path::new(path).join(DOE_INITIAL_FILE);
            info!("Save initial doe in {:?}", filepath);
            write_npy(filepath, &doe).expect("Write initial doe");
        }

        const MAX_RETRY: i32 = 10;
        let mut no_point_added_retries = MAX_RETRY;
        for i in 1..=n_iter {
            let (x_dat, y_dat) = self.next_points(i, &x_data, &y_data, &sampling);
            let rejected_count = update_data(&mut x_data, &mut y_data, &x_dat, &y_dat);
            if rejected_count > 0 {
                info!(
                    "Reject {}/{} point{} too close to previous ones",
                    rejected_count,
                    x_dat.nrows(),
                    if rejected_count > 1 { "s" } else { "" }
                );
            }
            debug!("New pts rejected = {} / {}", rejected_count, x_dat.nrows());
            if rejected_count == x_dat.nrows() {
                no_point_added_retries -= 1;
                if no_point_added_retries == 0 {
                    info!("Max number of retries ({}) without adding point", MAX_RETRY);
                    break;
                }
                info!("End iteration {}/{}", i, n_iter);
            } else {
                no_point_added_retries = MAX_RETRY;
                let count = (self.n_parallel - rejected_count) as i32;
                let x_to_eval = x_data.slice(s![-count.., ..]).to_owned();
                info!(
                    "Add {} point{}:",
                    count,
                    if rejected_count > 1 { "s" } else { "" }
                );
                info!("  {}", x_dat);
                let y_actual = self.eval(&x_to_eval);
                Zip::from(y_data.slice_mut(s![-count.., ..]).columns_mut())
                    .and(y_actual.columns())
                    .for_each(|mut y, val| y.assign(&val));
                let doe = concatenate![Axis(1), x_data, y_data];
                if self.outdir.is_some() {
                    let path = self.outdir.as_ref().unwrap();
                    std::fs::create_dir_all(path)?;
                    let filepath = std::path::Path::new(path).join(DOE_FILE);
                    info!("Save doe in {:?}", filepath);
                    write_npy(filepath, &doe).expect("Write current doe");
                }
                let best_index = self.find_best_result_index(&y_data);
                info!(
                    "End iteration {}/{}: Best fun(x)={} at x={}",
                    i,
                    n_iter,
                    y_data.row(best_index),
                    x_data.row(best_index)
                );
                if let Some(sol) = self.expected {
                    if (y_data[[best_index, 0]] - sol.value).abs() < sol.tolerance {
                        info!("Expected optimum : {:?}", sol);
                        info!("Expected optimum reached!");
                        break;
                    }
                }
            }
        }
        let best_index = self.find_best_result_index(&y_data);
        info!("{}", concatenate![Axis(1), x_data, y_data]);
        Ok(OptimResult {
            x_opt: x_data.row(best_index).to_owned(),
            y_opt: y_data.row(best_index).to_owned(),
        })
    }

    fn next_points(
        &'a self,
        _n: usize,
        x_data: &Array2<f64>,
        y_data: &Array2<f64>,
        sampling: &Lhs<f64, R>,
    ) -> (Array2<f64>, Array2<f64>) {
        let mut x_dat = Array2::zeros((0, x_data.ncols()));
        let mut y_dat = Array2::zeros((0, y_data.ncols()));
        let n_clusters = self.n_clusters.unwrap_or(1);

        let default_params = &Moe::params(n_clusters)
            .set_kpls_dim(self.kpls_dim)
            .set_regression_spec(self.regression_spec)
            .set_correlation_spec(self.correlation_spec)
            as &dyn MoeFit;
        let params = self.moe_params.unwrap_or(default_params);

        let obj_model = params
            .fit_for_predict(x_data, &y_data.slice(s![.., 0..1]).to_owned())
            .expect("GP training failure");

        let mut cstr_models: Vec<Box<dyn MoePredict>> = Vec::with_capacity(self.n_cstr);
        for k in 0..self.n_cstr {
            cstr_models.push(
                params
                    .fit_for_predict(x_data, &y_data.slice(s![.., k + 1..k + 2]).to_owned())
                    .expect("GP training failure"),
            )
        }
        for _ in 0..self.n_parallel {
            match self.find_best_point(x_data, y_data, sampling, obj_model.as_ref(), &cstr_models) {
                Ok(xk) => {
                    match self.get_virtual_point(&xk, y_data, obj_model.as_ref(), &cstr_models) {
                        Ok(yk) => {
                            y_dat = concatenate![
                                Axis(0),
                                y_dat,
                                Array2::from_shape_vec((1, 1 + self.n_cstr), yk).unwrap()
                            ];
                            x_dat = concatenate![Axis(0), x_dat, xk.insert_axis(Axis(0))];
                        }
                        Err(err) => {
                            // Error while predict at best point: ignore
                            info!("Error while getting virtual point: {}", err);
                            break;
                        }
                    }
                }
                Err(err) => {
                    // Cannot find best point: ignore
                    debug!("Find best point error: {}", err);
                    break;
                }
            }
        }
        (x_dat, y_dat)
    }

    fn find_best_point(
        &self,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        sampling: &Lhs<f64, R>,
        obj_model: &dyn MoePredict,
        cstr_models: &[Box<dyn MoePredict>],
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
                    self.infill_eval(x, obj_model, *f_min, *scale_obj, *scale_wb2)
                };
                grad[..].copy_from_slice(&x.to_vec().forward_diff(&f));
            }
            self.infill_eval(x, obj_model, *f_min, *scale_obj, *scale_wb2)
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
        let scale_obj = compute_obj_scale(&scaling_points, obj_model);
        let scale_cstr = compute_cstr_scales(&scaling_points, cstr_models);
        let scale_wb2 = if self.infill == InfillStrategy::WB2S {
            compute_wb2s_scale(&scaling_points, obj_model, *f_min)
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
                    scale_wb2,
                    scale_cstr: scale_cstr.to_owned(),
                },
            );
            let lower = self.xlimits.column(0).to_owned();
            optimizer.set_lower_bounds(lower.as_slice().unwrap())?;
            let upper = self.xlimits.column(1).to_owned();
            optimizer.set_upper_bounds(upper.as_slice().unwrap())?;
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
                            let res = x_opt.to_vec();
                            best_x = Some(Array::from(res));
                            success = true;
                        }
                    }
                    Err((err, code)) => {
                        debug!("Nlopt Err: {:?} (y_opt={})", err, code);
                    }
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
                if !row.slice(s![1..]).iter().any(|v| *v > self.cstr_tol) {
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
        obj_model: &dyn MoePredict,
        cstr_models: &[Box<dyn MoePredict>],
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
            let pred = obj_model.predict_values(x)?[[0, 0]];
            let var = obj_model.predict_variances(x)?[[0, 0]];
            let conf = match self.q_ei {
                QEiStrategy::KrigingBeliever => 0.,
                QEiStrategy::KrigingBelieverLowerBound => -3.,
                QEiStrategy::KrigingBelieverUpperBound => 3.,
                _ => -1., // never used
            };
            res.push(pred + conf * Scalar::sqrt(var));
            for cstr_model in cstr_models {
                res.push(cstr_model.predict_values(x)?[[0, 0]]);
            }
            Ok(res)
        }
    }

    fn infill_eval(
        &self,
        x: &[f64],
        obj_model: &dyn MoePredict,
        f_min: f64,
        scale: f64,
        scale_wb2: f64,
    ) -> f64 {
        let x_f = x.to_vec();
        let obj = match self.infill {
            InfillStrategy::EI => -ei(&x_f, obj_model, f_min),
            InfillStrategy::WB2 => -wb2s(&x_f, obj_model, f_min, 1.),
            InfillStrategy::WB2S => -wb2s(&x_f, obj_model, f_min, scale_wb2),
        };
        obj / scale
    }
}

impl<'a, O: GroupFunc, R: Rng + Clone> Evaluator for Egor<'a, O, R> {
    fn eval(&self, x: &Array2<f64>) -> Array2<f64> {
        if let Some(evaluator) = self.evaluator {
            (&self.obj)(&evaluator.eval(x).view())
        } else {
            (&self.obj)(&x.view())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use argmin_testfunctions::rosenbrock;
    use ndarray::{array, ArrayView2};
    use ndarray_npy::read_npy;
    use serial_test::serial;
    use std::time::Instant;

    fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
        (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
    }

    #[test]
    #[serial]
    fn test_xsinx_ei_egor() {
        let initial_doe = array![[0.], [7.], [25.]];
        let res = Egor::new(xsinx, &array![[0.0, 25.0]])
            .infill_strategy(InfillStrategy::EI)
            .regression_spec(RegressionSpec::QUADRATIC)
            .correlation_spec(CorrelationSpec::ALL)
            .n_eval(15)
            .doe(Some(initial_doe.to_owned()))
            .expect(Some(ApproxValue {
                value: -15.1,
                tolerance: 1e-1,
            }))
            .outdir(Some(".".to_string()))
            .minimize()
            .expect("Minimize failure");
        let expected = array![-15.1];
        assert_abs_diff_eq!(expected, res.y_opt, epsilon = 0.3);
        let saved_doe: Array2<f64> = read_npy(DOE_INITIAL_FILE).unwrap();
        assert_abs_diff_eq!(initial_doe, saved_doe.slice(s![..3, ..1]), epsilon = 1e-6);
    }

    #[test]
    #[serial]
    fn test_xsinx_wb2() {
        let res = Egor::new(xsinx, &array![[0.0, 25.0]])
            .n_eval(20)
            .minimize()
            .expect("Minimize failure");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    #[test]
    #[serial]
    fn test_xsinx_suggestions() {
        let mut ego = Egor::new(xsinx, &array![[0.0, 25.0]]);
        let ego = ego.infill_strategy(InfillStrategy::EI);

        let mut doe = array![[0.], [7.], [20.], [25.]];
        let mut y_doe = xsinx(&doe.view());
        for _i in 0..10 {
            let x_suggested = ego.suggest(&doe, &y_doe);

            doe = concatenate![Axis(0), doe, x_suggested];
            y_doe = xsinx(&doe.view());
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
    #[serial]
    fn test_rosenbrock_2d() {
        let now = Instant::now();
        let xlimits = array![[-2., 2.], [-2., 2.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Isaac64Rng::seed_from_u64(42))
            .sample(10);
        let res = Egor::new(rosenb, &xlimits)
            .with_rng(Isaac64Rng::seed_from_u64(42))
            .infill_strategy(InfillStrategy::EI)
            .doe(Some(doe))
            .n_eval(30)
            .expect(Some(ApproxValue {
                value: 0.0,
                tolerance: 1e-2,
            }))
            .minimize()
            .expect("Minimize failure");
        println!("Rosenbrock optim result = {:?}", res);
        println!("Elapsed = {:?}", now.elapsed());
        let expected = array![1., 1.];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 2e-1);
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
    #[serial]
    fn test_egor_g24() {
        let x = array![[1., 2.]];
        println!("{:?}", f_g24(&x.view()));
        let xlimits = array![[0., 3.], [0., 4.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Isaac64Rng::seed_from_u64(42))
            .sample(10);
        let res = Egor::new(f_g24, &xlimits)
            .with_rng(Isaac64Rng::seed_from_u64(42))
            .n_cstr(2)
            .infill_strategy(InfillStrategy::EI)
            .infill_optimizer(InfillOptimizer::Cobyla) // test passes also with WB2S and Slsqp
            .doe(Some(doe))
            .n_eval(40)
            // .expect(Some(ApproxValue {
            //     value: -5.5080,
            //     tolerance: 1e-3,
            // }))
            .minimize()
            .expect("Minimize failure");
        println!("G24 optim result = {:?}", res);
        let expected = array![2.3295, 3.1785];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 2e-2);
    }
}
