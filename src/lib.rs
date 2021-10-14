pub mod ego;
pub mod errors;

use doe::{SamplingMethod, LHS};
use ndarray::{arr2, Array2, Ix2};
use numpy::{IntoPyArray, PyArray, PyReadonlyArray};
use pyo3::prelude::*;

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

#[pymodule]
fn egobox(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lhs, m)?)?;

    Ok(())
}
