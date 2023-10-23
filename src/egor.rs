//! `egobox`, Rust toolbox for efficient global optimization
//!
//! Thanks to the [PyO3 project](https://pyo3.rs), which makes Rust well suited for building Python extensions,
//! the EGO algorithm written in Rust (aka `Egor`) is binded in Python. You can install the Python package using:
//!
//! ```bash
//! pip install egobox
//! ```
//!
//! See the [tutorial notebook](https://github.com/relf/egobox/doc/Egor_Tutorial.ipynb) for usage.
//!

use crate::types::*;
use ndarray::Array1;
use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Utility function converting `xlimits` float data list specifying bounds of x components
/// to x specified as a list of XType.Float types [egobox.XType]
///
/// # Parameters
///     xlimits : nx-size list of [lower_bound, upper_bound] where `nx` is the dimension of x
///
/// # Returns
///     xtypes: nx-size list of XSpec(XType(FLOAT), [lower_bound, upper_bounds]) where `nx` is the dimension of x
#[pyfunction]
pub(crate) fn to_specs(py: Python, xlimits: Vec<Vec<f64>>) -> PyResult<PyObject> {
    if xlimits.is_empty() || xlimits[0].is_empty() {
        let err = "Error: xspecs argument cannot be empty";
        return Err(PyValueError::new_err(err.to_string()));
    }
    Ok(xlimits
        .iter()
        .map(|xlimit| XSpec::new(XType(XType::FLOAT), xlimit.clone(), vec![]))
        .collect::<Vec<XSpec>>()
        .into_py(py))
}

/// Optimizer constructor
///
///    fun: array[n, nx]) -> array[n, ny]
///         the function to be minimized
///         fun(x) = [obj(x), cstr_1(x), ... cstr_k(x)] where
///            obj is the objective function [n, nx] -> [n, 1]
///            cstr_i is the ith constraint function [n, nx] -> [n, 1]
///            an k the number of constraints (n_cstr)
///            hence ny = 1 (obj) + k (cstrs)
///         cstr functions are expected be negative (<=0) at the optimum.
///
///     n_cstr (int):
///         the number of constraint functions.
///
///     cstr_tol (list(n_cstr,)):
///         List of tolerances for constraints to be satisfied (cstr < tol), list size should be equal to n_cstr.
///         None by default means zero tolerances.
///
///     xspecs (list(XSpec)) where XSpec(xtype=FLOAT|INT|ORD|ENUM, xlimits=[<f(xtype)>] or tags=[strings]):
///         Specifications of the nx components of the input x (eg. len(xspecs) == nx)
///         Depending on the x type we get the following for xlimits:
///         * when FLOAT: xlimits is [float lower_bound, float upper_bound],
///         * when INT: xlimits is [int lower_bound, int upper_bound],
///         * when ORD: xlimits is [float_1, float_2, ..., float_n],
///         * when ENUM: xlimits is just the int size of the enumeration otherwise a list of tags is specified
///           (eg xlimits=[3] or tags=["red", "green", "blue"], tags are there for documention purpose but
///            tags specific values themselves are not used only indices in the enum are used hence
///            we can just specify the size of the enum, xlimits=[3]),
///
///     n_start (int > 0):
///         Number of runs of infill strategy optimizations (best result taken)
///
///     n_doe (int >= 0):
///         Number of samples of initial LHS sampling (used when DOE not provided by the user).
///         When 0 a number of points is computed automatically regarding the number of input variables
///         of the function under optimization.
///
///     doe (array[ns, nt]):
///         Initial DOE containing ns samples:
///             either nt = nx then only x are specified and ns evals are done to get y doe values,
///             or nt = nx + ny then x = doe[:, :nx] and y = doe[:, nx:] are specified  
///
///     regr_spec (RegressionSpec flags, an int in [1, 7]):
///         Specification of regression models used in gaussian processes.
///         Can be RegressionSpec.CONSTANT (1), RegressionSpec.LINEAR (2), RegressionSpec.QUADRATIC (4) or
///         any bit-wise union of these values (e.g. RegressionSpec.CONSTANT | RegressionSpec.LINEAR)
///
///     corr_spec (CorrelationSpec flags, an int in [1, 15]):
///         Specification of correlation models used in gaussian processes.
///         Can be CorrelationSpec.SQUARED_EXPONENTIAL (1), CorrelationSpec.ABSOLUTE_EXPONENTIAL (2),
///         CorrelationSpec.MATERN32 (4), CorrelationSpec.MATERN52 (8) or
///         any bit-wise union of these values (e.g. CorrelationSpec.MATERN32 | CorrelationSpec.MATERN52)
///
///     infill_strategy (InfillStrategy enum)
///         Infill criteria to decide best next promising point.
///         Can be either InfillStrategy.EI, InfillStrategy.WB2 or InfillStrategy.WB2S.
///
///     q_points (int > 0):
///         Number of points to be evaluated to allow parallel evaluation of the function under optimization.
///
///     par_infill_strategy (ParInfillStrategy enum)
///         Parallel infill criteria (aka qEI) to get virtual next promising points in order to allow
///         q parallel evaluations of the function under optimization.
///         Can be either ParInfillStrategy.KB (Kriging Believer),
///         ParInfillStrategy.KBLB (KB Lower Bound), ParInfillStrategy.KBUB (KB Upper Bound),
///         ParInfillStrategy.CLMIN (Constant Liar Minimum)
///
///     infill_optimizer (InfillOptimizer enum)
///         Internal optimizer used to optimize infill criteria.
///         Can be either InfillOptimizer.COBYLA or InfillOptimizer.SLSQP
///
///     kpls_dim (0 < int < nx)
///         Number of components to be used when PLS projection is used (a.k.a KPLS method).
///         This is used to address high-dimensional problems typically when nx > 9.
///
///     n_clusters (int >= 0)
///         Number of clusters used by the mixture of surrogate experts.
///         When set to 0, the number of cluster is determined automatically and refreshed every
///         10-points addition (should say 'tentative addition' because addition may fail for some points
///         but it is counted anyway).
///   
///     target (float)
///         Known optimum used as stopping criterion.
///
///     outdir (String)
///         Directory to write optimization history and used as search path for hot start doe
///
///     hot_start (bool)
///         Start by loading initial doe from <outdir> directory
///
///     seed (int >= 0)
///         Random generator seed to allow computation reproducibility.
///      
#[pyclass]
pub(crate) struct Egor {
    pub fun: PyObject,
    pub xspecs: PyObject,
    pub n_cstr: usize,
    pub cstr_tol: Option<Vec<f64>>,
    pub n_start: usize,
    pub n_doe: usize,
    pub doe: Option<Array2<f64>>,
    pub regression_spec: RegressionSpec,
    pub correlation_spec: CorrelationSpec,
    pub infill_strategy: InfillStrategy,
    pub q_points: usize,
    pub par_infill_strategy: ParInfillStrategy,
    pub infill_optimizer: InfillOptimizer,
    pub kpls_dim: Option<usize>,
    pub n_clusters: Option<usize>,
    pub target: f64,
    pub outdir: Option<String>,
    pub hot_start: bool,
    pub seed: Option<u64>,
}

#[pyclass]
pub(crate) struct OptimResult {
    #[pyo3(get)]
    x_opt: Py<PyArray1<f64>>,
    #[pyo3(get)]
    y_opt: Py<PyArray1<f64>>,
    #[pyo3(get)]
    x_hist: Py<PyArray2<f64>>,
    #[pyo3(get)]
    y_hist: Py<PyArray2<f64>>,
}

#[pymethods]
impl Egor {
    #[new]
    #[pyo3(signature = (
        fun,
        xspecs,
        n_cstr = 0,
        cstr_tol = None,
        n_start = 20,
        n_doe = 0,
        doe = None,
        regr_spec = RegressionSpec::CONSTANT,
        corr_spec = CorrelationSpec::SQUARED_EXPONENTIAL,
        infill_strategy = InfillStrategy::WB2,
        q_points = 1,
        par_infill_strategy = ParInfillStrategy::KB,
        infill_optimizer = InfillOptimizer::COBYLA,
        kpls_dim = None,
        n_clusters = 1,
        target = f64::NEG_INFINITY,
        outdir = None,
        hot_start = false,
        seed = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python,
        fun: PyObject,
        xspecs: PyObject,
        n_cstr: usize,
        cstr_tol: Option<Vec<f64>>,
        n_start: usize,
        n_doe: usize,
        doe: Option<PyReadonlyArray2<f64>>,
        regr_spec: u8,
        corr_spec: u8,
        infill_strategy: InfillStrategy,
        q_points: usize,
        par_infill_strategy: ParInfillStrategy,
        infill_optimizer: InfillOptimizer,
        kpls_dim: Option<usize>,
        n_clusters: Option<usize>,
        target: f64,
        outdir: Option<String>,
        hot_start: bool,
        seed: Option<u64>,
    ) -> Self {
        let doe = doe.map(|x| x.to_owned_array());
        Egor {
            fun: fun.to_object(py),
            xspecs,
            n_cstr,
            cstr_tol,
            n_start,
            n_doe,
            doe,
            regression_spec: RegressionSpec(regr_spec),
            correlation_spec: CorrelationSpec(corr_spec),
            infill_strategy,
            q_points,
            par_infill_strategy,
            infill_optimizer,
            kpls_dim,
            n_clusters,
            target,
            outdir,
            hot_start,
            seed,
        }
    }

    /// This function finds the minimum of a given function `fun`
    ///
    /// # Parameters
    ///     n_iter:
    ///         the iteration budget, number of fun calls is n_doe + q_points * n_iter.
    ///
    /// # Returns
    ///     optimization result
    ///         x_opt (array[1, nx]): x value  where fun is at its minimum subject to constraint
    ///         y_opt (array[1, nx]): fun(x_opt)
    ///
    #[pyo3(signature = (n_iter = 20))]
    fn minimize(&self, py: Python, n_iter: usize) -> PyResult<OptimResult> {
        let fun = self.fun.to_object(py);
        let obj = move |x: &ArrayView2<f64>| -> Array2<f64> {
            Python::with_gil(|py| {
                let args = (x.to_owned().into_pyarray(py),);
                let res = fun.call1(py, args).unwrap();
                let pyarray: &PyArray2<f64> = res.extract(py).unwrap();
                pyarray.to_owned_array()
            })
        };

        let infill_strategy = match self.infill_strategy {
            InfillStrategy::EI => egobox_ego::InfillStrategy::EI,
            InfillStrategy::WB2 => egobox_ego::InfillStrategy::WB2,
            InfillStrategy::WB2S => egobox_ego::InfillStrategy::WB2S,
        };

        let qei_strategy = match self.par_infill_strategy {
            ParInfillStrategy::KB => egobox_ego::QEiStrategy::KrigingBeliever,
            ParInfillStrategy::KBLB => egobox_ego::QEiStrategy::KrigingBelieverLowerBound,
            ParInfillStrategy::KBUB => egobox_ego::QEiStrategy::KrigingBelieverUpperBound,
            ParInfillStrategy::CLMIN => egobox_ego::QEiStrategy::ConstantLiarMinimum,
        };

        let infill_optimizer = match self.infill_optimizer {
            InfillOptimizer::COBYLA => egobox_ego::InfillOptimizer::Cobyla,
            InfillOptimizer::SLSQP => egobox_ego::InfillOptimizer::Slsqp,
        };

        let xspecs: Vec<XSpec> = self.xspecs.extract(py).expect("Error in xspecs conversion");
        if xspecs.is_empty() {
            panic!("Error: xspecs argument cannot be empty")
        }

        let xtypes: Vec<egobox_ego::XType> = xspecs
            .iter()
            .map(|spec| match spec.xtype {
                XType(XType::FLOAT) => egobox_ego::XType::Cont(spec.xlimits[0], spec.xlimits[1]),
                XType(XType::INT) => {
                    egobox_ego::XType::Int(spec.xlimits[0] as i32, spec.xlimits[1] as i32)
                }
                XType(XType::ORD) => egobox_ego::XType::Ord(spec.xlimits.clone()),
                XType(XType::ENUM) => {
                    if spec.tags.is_empty() {
                        egobox_ego::XType::Enum(spec.xlimits[0] as usize)
                    } else {
                        egobox_ego::XType::Enum(spec.tags.len())
                    }
                },
                XType(i) => panic!(
                    "Bad variable type: should be either XType.FLOAT {}, XType.INT {}, XType.ORD {}, XType.ENUM {}, got {}",
                    XType::FLOAT,
                    XType::INT,
                    XType::ORD,
                    XType::ENUM,
                    i
                ),
            })
            .collect();
        println!("{:?}", xtypes);

        let mut mixintegor_build = egobox_ego::EgorBuilder::optimize(obj);
        if let Some(seed) = self.seed {
            mixintegor_build = mixintegor_build.random_seed(seed);
        };

        let cstr_tol = self.cstr_tol.clone().unwrap_or(vec![0.0; self.n_cstr]);
        let cstr_tol = Array1::from_vec(cstr_tol);

        let mut mixintegor = mixintegor_build
            .min_within_mixint_space(&xtypes)
            .n_cstr(self.n_cstr)
            .n_iter(n_iter)
            .n_start(self.n_start)
            .n_doe(self.n_doe)
            .cstr_tol(&cstr_tol)
            .regression_spec(egobox_moe::RegressionSpec::from_bits(self.regression_spec.0).unwrap())
            .correlation_spec(
                egobox_moe::CorrelationSpec::from_bits(self.correlation_spec.0).unwrap(),
            )
            .infill_strategy(infill_strategy)
            .q_points(self.q_points)
            .qei_strategy(qei_strategy)
            .infill_optimizer(infill_optimizer)
            .target(self.target)
            .hot_start(self.hot_start);
        if let Some(doe) = self.doe.as_ref() {
            mixintegor = mixintegor.doe(doe);
        };
        if let Some(kpls_dim) = self.kpls_dim {
            mixintegor = mixintegor.kpls_dim(kpls_dim);
        };
        if let Some(n_clusters) = self.n_clusters {
            mixintegor = mixintegor.n_clusters(n_clusters);
        };
        if let Some(outdir) = self.outdir.as_ref().cloned() {
            mixintegor = mixintegor.outdir(outdir);
        };

        let res = py.allow_threads(|| {
            mixintegor
                .run()
                .expect("Egor should optimize the objective function")
        });
        let x_opt = res.x_opt.into_pyarray(py).to_owned();
        let y_opt = res.y_opt.into_pyarray(py).to_owned();
        let x_hist = res.x_hist.into_pyarray(py).to_owned();
        let y_hist = res.y_hist.into_pyarray(py).to_owned();
        Ok(OptimResult {
            x_opt,
            y_opt,
            x_hist,
            y_hist,
        })
    }
}
