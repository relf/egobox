use doe::{SamplingMethod, LHS};
use ego::{AcqStrategy, Ego};
use ndarray::{array, Ix2};
use numpy::{IntoPyArray, PyArray, PyReadonlyArray};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn lhs<'py>(
    py: Python<'py>,
    xlimits: PyReadonlyArray<f64, Ix2>,
    a: usize,
) -> &'py PyArray<f64, Ix2> {
    let actual = LHS::new(&xlimits.as_array()).sample(a);

    actual.into_pyarray(py)
}

#[pyclass]
struct Optimizer {}

#[pyclass]
struct OptimResult {
    #[pyo3(get)]
    x_opt: Vec<f64>,
    #[pyo3(get)]
    y_opt: Vec<f64>,
}

unsafe impl Send for OptimResult {}

#[pymethods]
impl Optimizer {
    #[new]
    fn new() -> Self {
        Optimizer {}
    }

    fn minimize(&self, py_callback: &PyAny) -> OptimResult {
        let callback: PyObject = py_callback.into();

        let obj = move |x: &[f64]| -> f64 {
            let gil = Python::acquire_gil();
            let py = gil.python();

            let args = PyTuple::new(py, &[x.clone()]);
            let res = callback.call1(py, args);

            let val = res.unwrap().extract::<f64>(py).unwrap();
            val
        };

        let res = Ego::new(obj, &array![[0.0, 25.0]])
            .acq_strategy(AcqStrategy::WB2)
            .minimize();

        OptimResult {
            x_opt: res.x_opt.to_vec(),
            y_opt: res.y_opt.to_vec(),
        }
    }
}

#[pymodule]
fn egobox(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lhs, m)?)?;
    m.add_class::<Optimizer>()?;
    m.add_class::<OptimResult>()?;
    Ok(())
}
