pub mod ego;
pub mod errors;

use doe::{SamplingMethod, LHS};
use ndarray::{arr2, Array2, Ix1, Ix2};
use numpy::{IntoPyArray, PyArray, PyReadonlyArray};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyTuple};

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
struct Ego {}

#[pyclass]
struct OptimResult {
    #[pyo3(get)]
    x_opt: Vec<f64>,
    #[pyo3(get)]
    y_opt: f64,
}

unsafe impl Send for OptimResult {}

#[pymethods]
impl Ego {
    #[new]
    fn new() -> Self {
        Ego {}
    }

    fn optimize(&self, test: &PyAny) -> OptimResult {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let x = vec![0.1];
        let args = PyTuple::new(py, &[x.clone()]);
        let res = test.call1(args);

        OptimResult {
            x_opt: x.clone(),
            y_opt: res.unwrap().extract::<f64>().unwrap(),
        }
        // let f = |&[f64]| {
        // let res = Ego::new(f, &array![[0.0, 25.0]])
        // .acq_strategy(AcqStrategy::WB2)
        // .minimize()
        // };
    }
}

#[pymodule]
fn egobox(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lhs, m)?)?;
    m.add_class::<Ego>()?;
    m.add_class::<OptimResult>()?;
    Ok(())
}
