#![doc = include_str!("../README.md")]

mod egor;

use egor::*;
use pyo3::prelude::*;

#[doc(hidden)]
#[pymodule]
fn egobox(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(to_specs, m)?)?;

    m.add_function(wrap_pyfunction!(lhs, m)?)?;

    m.add_class::<Egor>()?;
    m.add_class::<RegressionSpec>()?;
    m.add_class::<CorrelationSpec>()?;
    m.add_class::<InfillStrategy>()?;
    m.add_class::<ParInfillStrategy>()?;
    m.add_class::<InfillOptimizer>()?;
    m.add_class::<Vtype>()?;
    m.add_class::<Vspec>()?;
    m.add_class::<OptimResult>()?;
    m.add_class::<ExpectedOptimum>()?;
    Ok(())
}
