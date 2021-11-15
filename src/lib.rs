use doe::{SamplingMethod, LHS};
use ego::Sego;
use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3_log;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn lhs<'py>(py: Python<'py>, xlimits: PyReadonlyArray2<f64>, a: usize) -> &'py PyArray2<f64> {
    let actual = LHS::new(&xlimits.as_array()).sample(a);
    actual.into_pyarray(py)
}

#[pyclass]
#[pyo3(text_signature = "(xlimits, /)")]
struct SegoOptimizer {
    pub xlimits: Array2<f64>,
    // pub n_iter: usize,
    // pub n_start: usize,
    // pub n_parallel: usize,
    // pub n_doe: usize,
    // pub n_cstr: usize,
    // pub x_doe: Option<Array2<f64>>,
    // pub q_ei: QEiStrategy,
    // pub acq: AcqStrategy,
    // pub acq_optimizer: AcqOptimizer,
    // pub regr_spec: RegressionSpec,
    // pub corr_spec: CorrelationSpec,
}

#[pyclass]
struct OptimResult {
    #[pyo3(get)]
    x_opt: Vec<f64>,
    #[pyo3(get)]
    y_opt: Vec<f64>,
}

#[pymethods]
impl SegoOptimizer {
    #[new]
    fn new(py_xlimits: PyReadonlyArray2<f64>) -> Self {
        let xlimits = py_xlimits.to_owned_array();
        SegoOptimizer { xlimits }
    }

    /// This function finds the minimum of a given function
    ///
    /// Parameters
    ///
    ///     fun: the function to be minimized
    ///          fun: array[n, nx]) -> array[n, ny]
    ///          fun(x) = [obj(x), cstr_1(x), ... cstr_k(x)] where
    ///             obj is the objective function [n, nx] -> [n, 1]
    ///             cstr_i is the ith constraint function [n, nx] -> [n, 1]
    ///             an k the number of constraints (n_cstr)
    ///             hence ny = 1 (obj) + k (cstrs)
    ///          cstr functions are expected be negative (<=0) at the optimum.
    ///
    ///     n_cstr (int): the number of constraints (default 0)
    ///             
    ///     n_iter (int): the iteration budget, number of fun calls (default 20)
    ///
    /// Returns
    ///
    ///     optimization result
    ///         x_opt (array[1, nx]): x value  where fun is at its minimum subject to constraint
    ///         y_opt (array[1, nx]): fun(x_opt)
    ///
    #[args(n_iter = "20", n_cstr = "0")]
    #[pyo3(text_signature = "(fun, n_cstr, n_iter, /)")]
    fn minimize(&self, fun: &PyAny, n_cstr: usize, n_iter: usize) -> OptimResult {
        let obj: PyObject = fun.into();
        let obj = move |x: &ArrayView2<f64>| -> Array2<f64> {
            let gil = Python::acquire_gil();
            let py = gil.python();
            let args = (x.to_owned().into_pyarray(py),);
            let res = obj.call1(py, args).unwrap();
            let pyarray: &PyArray2<f64> = res.extract(py).unwrap();
            let val = pyarray.to_owned_array();
            val
        };

        let res = Sego::new(obj, &self.xlimits)
            .n_cstr(n_cstr)
            .n_iter(n_iter)
            .minimize();

        OptimResult {
            x_opt: res.x_opt.to_vec(),
            y_opt: res.y_opt.to_vec(),
        }
    }
}

#[pymodule]
fn egobox(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(lhs, m)?)?;
    m.add_class::<SegoOptimizer>()?;
    m.add_class::<OptimResult>()?;
    Ok(())
}
