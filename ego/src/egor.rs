//! Egor optimizer implements EGO algorithm with basic handling of constraints.
//!
//! ```no_run
//! # use ndarray::{array, Array2, ArrayView1, ArrayView2, Zip};
//! # use egobox_doe::{Lhs, SamplingMethod};
//! # use egobox_ego::{EgorBuilder, InfillStrategy, InfillOptimizer};
//!
//! # use rand_xoshiro::Xoshiro256Plus;
//! # use ndarray_rand::rand::SeedableRng;
//! use argmin_testfunctions::rosenbrock;
//!
//! // Rosenbrock test function: minimum y_opt = 0 at x_opt = (1, 1)
//! fn rosenb(x: &ArrayView2<f64>) -> Array2<f64> {
//!     let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
//!     Zip::from(y.rows_mut())
//!         .and(x.rows())
//!         .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec())]));
//!     y
//! }
//!
//! let xlimits = array![[-2., 2.], [-2., 2.]];
//! let res = EgorBuilder::optimize(rosenb).configure(|config|
//!         config
//!             .infill_strategy(InfillStrategy::EI)
//!             .n_doe(10)
//!             .target(1e-1)
//!             .max_iters(30))
//!     .min_within(&xlimits)
//!     .run()
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
//! # use egobox_ego::{EgorBuilder, InfillStrategy, InfillOptimizer};
//! # use rand_xoshiro::Xoshiro256Plus;
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
//! let res = EgorBuilder::optimize(f_g24).configure(|config|
//!             config
//!                 .n_cstr(2)
//!                 .infill_strategy(InfillStrategy::EI)
//!                 .infill_optimizer(InfillOptimizer::Cobyla)
//!                 .doe(&doe)
//!                 .max_iters(40)
//!                 .target(-5.5080))
//!            .min_within(&xlimits)
//!            .run()
//!            .expect("g24 minimized");
//! println!("G24 min result = {:?}", res);
//! ```
//!
use crate::egor_config::*;
use crate::errors::Result;
use crate::gpmix::mixint::*;
use crate::types::*;
use crate::{to_xtypes, EgorSolver};

use egobox_moe::GpMixtureParams;
use log::info;
use ndarray::{concatenate, ArrayBase, Axis, Data, Ix2};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use argmin::core::{Executor, State};

/// EGO optimizer builder allowing to specify function to be minimized
/// subject to constraints intended to be negative.
///
pub struct EgorBuilder<O: GroupFunc> {
    fobj: O,
    config: EgorConfig,
}

impl<O: GroupFunc> EgorBuilder<O> {
    /// Function to be minimized domain should be basically R^nx -> R^ny
    /// where nx is the dimension of input x and ny the output dimension
    /// equal to 1 (obj) + n (cstrs).
    /// But function has to be able to evaluate several points in one go
    /// hence take an (p, nx) matrix and return an (p, ny) matrix
    pub fn optimize(fobj: O) -> Self {
        EgorBuilder {
            fobj,
            config: EgorConfig::default(),
        }
    }

    pub fn configure<F: FnOnce(EgorConfig) -> EgorConfig>(mut self, init: F) -> Self {
        self.config = init(self.config);
        self
    }

    /// Build an Egor optimizer to minimize the function within
    /// the continuous `xlimits` specified as [[lower, upper], ...] array where the
    /// number of rows gives the dimension of the inputs (continuous optimization)
    /// and the ith row is the interval of the ith component of the input x.
    pub fn min_within(
        self,
        xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Egor<O, GpMixtureParams<f64>> {
        let rng = if let Some(seed) = self.config.seed {
            Xoshiro256Plus::seed_from_u64(seed)
        } else {
            Xoshiro256Plus::from_entropy()
        };
        let config = EgorConfig {
            xtypes: to_xtypes(xlimits),
            ..self.config.clone()
        };
        Egor {
            fobj: ObjFunc::new(self.fobj),
            solver: EgorSolver::new(config, rng),
        }
    }

    /// Build an Egor optimizer to minimize the function R^n -> R^p taking
    /// inputs specified with given xtypes where some of components may be
    /// discrete variables (mixed-integer optimization).
    pub fn min_within_mixint_space(self, xtypes: &[XType]) -> Egor<O, MixintGpMixtureParams> {
        let rng = if let Some(seed) = self.config.seed {
            Xoshiro256Plus::seed_from_u64(seed)
        } else {
            Xoshiro256Plus::from_entropy()
        };
        let config = EgorConfig {
            xtypes: xtypes.into(),
            ..self.config.clone()
        };
        Egor {
            fobj: ObjFunc::new(self.fobj),
            solver: EgorSolver::new(config, rng),
        }
    }
}

/// Egor optimizer structure used to parameterize the underlying `argmin::Solver`
/// and trigger the optimization using `argmin::Executor`.
#[derive(Clone)]
pub struct Egor<O: GroupFunc, SB: SurrogateBuilder> {
    fobj: ObjFunc<O>,
    solver: EgorSolver<SB>,
}

impl<O: GroupFunc, SB: SurrogateBuilder> Egor<O, SB> {
    /// Runs the (constrained) optimization of the objective function.
    pub fn run(&self) -> Result<OptimResult<f64>> {
        let xtypes = self.solver.config.xtypes.clone();
        info!("{:?}", self.solver.config);

        let result = Executor::new(self.fobj.clone(), self.solver.clone()).run()?;
        info!("{}", result);
        let (x_data, y_data) = result.state().clone().take_data().unwrap();

        let res = if !self.solver.config.discrete() {
            info!("History: \n{}", concatenate![Axis(1), x_data, y_data]);
            OptimResult {
                x_opt: result.state.get_best_param().unwrap().to_owned(),
                y_opt: result.state.get_full_best_cost().unwrap().to_owned(),
                x_hist: x_data,
                y_hist: y_data,
                state: result.state,
            }
        } else {
            let x_data = to_discrete_space(&xtypes, &x_data.view());
            info!("History: \n{}", concatenate![Axis(1), x_data, y_data]);

            let x_opt = result
                .state
                .get_best_param()
                .unwrap()
                .to_owned()
                .insert_axis(Axis(0));
            let x_opt = to_discrete_space(&xtypes, &x_opt.view());
            OptimResult {
                x_opt: x_opt.row(0).to_owned(),
                y_opt: result.state.get_full_best_cost().unwrap().to_owned(),
                x_hist: x_data,
                y_hist: y_data,
                state: result.state,
            }
        };
        info!("Optim Result: min f(x)={} at x={}", res.y_opt, res.x_opt);

        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use argmin_testfunctions::rosenbrock;
    use egobox_doe::{Lhs, SamplingMethod};
    use ndarray::{array, s, Array2, ArrayView2, Ix1, Zip};

    use ndarray_npy::read_npy;

    use serial_test::serial;
    use std::time::Instant;

    use crate::{gpmix::spec::*, DOE_FILE, DOE_INITIAL_FILE};

    #[cfg(not(feature = "blas"))]
    use linfa_linalg::norm::*;
    #[cfg(feature = "blas")]
    use ndarray_linalg::Norm;

    fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
        (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
    }

    #[test]
    #[serial]
    fn test_xsinx_ei_quadratic_egor_builder() {
        let outdir = "target/test_egor_builder_01";
        let outfile = format!("{outdir}/{DOE_INITIAL_FILE}");
        let _ = std::fs::remove_file(&outfile);
        let initial_doe = array![[0.], [7.], [25.]];
        let res = EgorBuilder::optimize(xsinx)
            .configure(|cfg| {
                cfg.infill_strategy(InfillStrategy::EI)
                    .regression_spec(RegressionSpec::QUADRATIC)
                    .correlation_spec(CorrelationSpec::ALL)
                    .max_iters(30)
                    .doe(&initial_doe)
                    .target(-15.1)
                    .outdir(outdir)
            })
            .min_within(&array![[0.0, 25.0]])
            .run()
            .expect("Egor should minimize xsinx");
        let expected = array![-15.1];
        assert_abs_diff_eq!(expected, res.y_opt, epsilon = 0.5);
        let saved_doe: Array2<f64> = read_npy(&outfile).unwrap();
        let _ = std::fs::remove_file(&outfile);
        assert_abs_diff_eq!(initial_doe, saved_doe.slice(s![..3, ..1]), epsilon = 1e-6);
    }

    #[test]
    #[serial]
    fn test_xsinx_wb2_egor_builder() {
        let res = EgorBuilder::optimize(xsinx)
            .configure(|config| {
                config
                    .max_iters(20)
                    .regression_spec(RegressionSpec::ALL)
                    .correlation_spec(CorrelationSpec::ALL)
                    .seed(1)
            })
            .min_within(&array![[0.0, 25.0]])
            .run()
            .expect("Egor should minimize");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    #[test]
    #[serial]
    fn test_xsinx_optmod_egor() {
        let res = EgorBuilder::optimize(xsinx)
            .configure(|config| config.max_iters(20).n_optmod(3))
            .min_within(&array![[0.0, 25.0]])
            .run()
            .expect("Egor should minimize");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    #[test]
    #[serial]
    fn test_xsinx_auto_clustering_egor_builder() {
        let res = EgorBuilder::optimize(xsinx)
            .configure(|config| config.n_clusters(0).max_iters(20))
            .min_within(&array![[0.0, 25.0]])
            .run()
            .expect("Egor with auto clustering should minimize xsinx");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    #[test]
    #[serial]
    fn test_xsinx_with_hotstart_egor_builder() {
        let outdir = "target/test_hotstart_01";
        let _ = std::fs::remove_file(format!("{outdir}/{DOE_INITIAL_FILE}"));
        let _ = std::fs::remove_file(format!("{outdir}/{DOE_FILE}"));
        let xlimits = array![[0.0, 25.0]];
        let doe = Lhs::new(&xlimits).sample(10);
        let res = EgorBuilder::optimize(xsinx)
            .configure(|config| config.max_iters(15).doe(&doe).outdir(outdir).seed(42))
            .min_within(&xlimits)
            .run()
            .expect("Minimize failure");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);

        let res = EgorBuilder::optimize(xsinx)
            .configure(|config| config.max_iters(5).outdir(outdir).hot_start(true).seed(42))
            .min_within(&xlimits)
            .run()
            .expect("Egor should minimize xsinx");
        let _ = std::fs::remove_file(format!("target/test_egor_builder/{DOE_INITIAL_FILE}"));
        let _ = std::fs::remove_file(format!("target/test_egor_builder/{DOE_FILE}"));
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    fn rosenb(x: &ArrayView2<f64>) -> Array2<f64> {
        let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
        Zip::from(y.rows_mut())
            .and(x.rows())
            .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec())]));
        y
    }

    #[test]
    #[serial]
    fn test_rosenbrock_2d_egor_builder() {
        let now = Instant::now();
        let xlimits = array![[-2., 2.], [-2., 2.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(10);
        let res = EgorBuilder::optimize(rosenb)
            .configure(|config| {
                config
                    .doe(&doe)
                    .max_iters(100)
                    .regression_spec(RegressionSpec::ALL)
                    .correlation_spec(CorrelationSpec::ALL)
                    .target(1e-2)
                    .seed(42)
            })
            .min_within(&xlimits)
            .run()
            .expect("Minimize failure");
        println!("Rosenbrock optim result = {res:?}");
        println!("Elapsed = {:?}", now.elapsed());
        let expected = array![1., 1.];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 5e-1);
    }

    #[test]
    #[serial]
    fn test_rosenbrock_2d_infill_target_egor_builder() {
        let now = Instant::now();
        let xlimits = array![[-2., 2.], [-2., 2.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(10);
        let res = EgorBuilder::optimize(rosenb)
            .configure(|config| config.doe(&doe).max_iters(20).seed(42))
            .min_within(&xlimits)
            .run()
            .expect("Minimize failure");
        println!("Rosenbrock optim result = {res:?}");
        println!("Elapsed = {:?}", now.elapsed());
        let expected = array![1., 1.];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 5e-1);
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
    fn test_egor_g24_basic_egor_builder() {
        let xlimits = array![[0., 3.], [0., 4.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(3);
        let res = EgorBuilder::optimize(f_g24)
            .configure(|config| {
                config
                    .n_cstr(2)
                    .doe(&doe)
                    .max_iters(20)
                    .cstr_tol(array![2e-6, 1e-6])
                    .seed(42)
            })
            .min_within(&xlimits)
            .run()
            .expect("Minimize failure");
        println!("G24 optim result = {res:?}");
        let expected = array![2.3295, 3.1785];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 2e-2);
    }

    #[test]
    #[serial]
    fn test_egor_g24_qei_egor_builder() {
        let xlimits = array![[0., 3.], [0., 4.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(10);
        let res = EgorBuilder::optimize(f_g24)
            .configure(|config| {
                config
                    .regression_spec(RegressionSpec::ALL)
                    .correlation_spec(CorrelationSpec::ALL)
                    .n_cstr(2)
                    .cstr_tol(array![2e-6, 2e-6])
                    .q_points(2)
                    .qei_strategy(QEiStrategy::KrigingBeliever)
                    .doe(&doe)
                    .target(-5.5030)
                    .max_iters(30)
                    .seed(42)
            })
            .min_within(&xlimits)
            .run()
            .expect("Egor minimization");
        println!("G24 optim result = {res:?}");
        let expected = array![2.3295, 3.1785];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 2e-2);
    }

    // Mixed-integer tests

    fn mixsinx(x: &ArrayView2<f64>) -> Array2<f64> {
        if (x.mapv(|v| v.round()).norm_l2() - x.norm_l2()).abs() < 1e-6 {
            (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
        } else {
            panic!("Error: mixsinx works only on integer, got {:?}", x)
        }
    }

    #[test]
    #[serial]
    fn test_mixsinx_ei_mixint_egor_builder() {
        let max_iters = 30;
        let doe = array![[0.], [7.], [25.]];
        let xtypes = vec![XType::Int(0, 25)];

        let res = EgorBuilder::optimize(mixsinx)
            .configure(|config| {
                config
                    .doe(&doe)
                    .max_iters(max_iters)
                    .target(-15.1)
                    .infill_strategy(InfillStrategy::EI)
                    .seed(42)
            })
            .min_within_mixint_space(&xtypes)
            .run()
            .unwrap();
        assert_abs_diff_eq!(array![18.], res.x_opt, epsilon = 2.);
    }

    #[test]
    #[serial]
    fn test_mixsinx_reclustering_mixint_egor_builder() {
        let max_iters = 30;
        let doe = array![[0.], [7.], [25.]];
        let xtypes = vec![XType::Int(0, 25)];

        let res = EgorBuilder::optimize(mixsinx)
            .configure(|config| {
                config
                    .doe(&doe)
                    .max_iters(max_iters)
                    .target(-15.1)
                    .infill_strategy(InfillStrategy::EI)
                    .seed(42)
            })
            .min_within_mixint_space(&xtypes)
            .run()
            .unwrap();
        assert_abs_diff_eq!(array![18.], res.x_opt, epsilon = 2.);
    }

    #[test]
    #[serial]
    fn test_mixsinx_wb2_mixint_egor_builder() {
        let max_iters = 30;
        let xtypes = vec![XType::Int(0, 25)];

        let res = EgorBuilder::optimize(mixsinx)
            .configure(|config| {
                config
                    .regression_spec(egobox_moe::RegressionSpec::CONSTANT)
                    .correlation_spec(egobox_moe::CorrelationSpec::SQUAREDEXPONENTIAL)
                    .max_iters(max_iters)
                    .seed(42)
            })
            .min_within_mixint_space(&xtypes)
            .run()
            .unwrap();
        assert_abs_diff_eq!(&array![18.], &res.x_opt, epsilon = 3.);
    }

    fn mixobj(x: &ArrayView2<f64>) -> Array2<f64> {
        // XType.Float
        let x1 = x.column(0);
        // XType.ENUM 1
        let c1 = x.column(1);
        let x2 = c1.mapv(|v| (v == 0.) as i32 as f64);
        let x3 = c1.mapv(|v| (v == 1.) as i32 as f64);
        let x4 = c1.mapv(|v| (v == 2.) as i32 as f64);
        // XType.ENUM 2
        let c2 = x.column(2);
        let x5 = c2.mapv(|v| (v == 0.) as i32 as f64);
        let x6 = c2.mapv(|v| (v == 1.) as i32 as f64);
        // XTypes.ORD
        let i = x.column(3);

        let y = (x2.clone() + x3.mapv(|v| 2. * v) + x4.mapv(|v| 3. * v)) * x5 * x1
            + (x2 + x3.mapv(|v| 2. * v) + x4.mapv(|v| 3. * v)) * x6 * 0.95 * x1
            + i;
        let d = y.len();
        y.into_shape((d, 1)).unwrap()
    }

    #[test]
    #[serial]
    fn test_mixobj_mixint_egor_builder() {
        let env = env_logger::Env::new().filter_or("EGOBOX_LOG", "info");
        let mut builder = env_logger::Builder::from_env(env);
        let builder = builder.target(env_logger::Target::Stdout);
        builder.try_init().ok();
        let max_iters = 10;
        let xtypes = vec![
            XType::Cont(-5., 5.),
            XType::Enum(3),
            XType::Enum(2),
            XType::Ord(vec![0., 2., 3.]),
        ];

        let res = EgorBuilder::optimize(mixobj)
            .configure(|config| {
                config
                    .regression_spec(egobox_moe::RegressionSpec::CONSTANT)
                    .correlation_spec(egobox_moe::CorrelationSpec::SQUAREDEXPONENTIAL)
                    .max_iters(max_iters)
                    .seed(42)
            })
            .min_within_mixint_space(&xtypes)
            .run()
            .unwrap();
        println!("res={:?}", res);
        assert_abs_diff_eq!(&array![-15.], &res.y_opt, epsilon = 1.);
    }

    #[test]
    #[serial]
    fn test_mixobj_mixint_hotstart_egor_builder() {
        let env = env_logger::Env::new().filter_or("EGOBOX_LOG", "info");
        let mut builder = env_logger::Builder::from_env(env);
        let builder = builder.target(env_logger::Target::Stdout);
        builder.try_init().ok();

        let outdir = "target/test_hotstart_02";
        let outfile = format!("{outdir}/{DOE_INITIAL_FILE}");
        let _ = std::fs::remove_file(format!("{outdir}/{DOE_INITIAL_FILE}"));
        let _ = std::fs::remove_file(format!("{outdir}/{DOE_FILE}"));

        let xtypes = vec![
            XType::Cont(-5., 5.),
            XType::Enum(3),
            XType::Enum(2),
            XType::Ord(vec![0., 2., 3.]),
        ];
        let xlimits = as_continuous_limits::<f64>(&xtypes);

        EgorBuilder::optimize(mixobj)
            .configure(|config| config.outdir(outdir).max_iters(1).seed(42))
            .min_within_mixint_space(&xtypes)
            .run()
            .unwrap();

        // Check saved DOE has continuous (unfolded) dimension + one output
        let saved_doe: Array2<f64> = read_npy(outfile).unwrap();
        assert_eq!(saved_doe.shape()[1], xlimits.nrows() + 1);

        // Check that with no iteration, obj function is never called
        // as the DOE does not need to be evaluated!
        EgorBuilder::optimize(|_x| panic!("Should not call objective function!"))
            .configure(|config| config.outdir(outdir).hot_start(true).max_iters(0).seed(42))
            .min_within_mixint_space(&xtypes)
            .run()
            .unwrap();
    }
}
