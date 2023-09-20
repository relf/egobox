#![doc = include_str!("../README.md")]

mod egor;
mod gpmix;
mod sampling;
pub(crate) mod types;

use egor::*;
use gpmix::*;
use sampling::*;
use types::*;

use pyo3::prelude::*;

#[doc(hidden)]
#[pymodule]
fn egobox(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    // utils
    m.add_function(wrap_pyfunction!(to_specs, m)?)?;
    m.add_function(wrap_pyfunction!(lhs, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::sampling, m)?)?;

    // types
    m.add_class::<sampling::Sampling>()?;
    m.add_class::<RegressionSpec>()?;
    m.add_class::<CorrelationSpec>()?;
    m.add_class::<InfillStrategy>()?;
    m.add_class::<ParInfillStrategy>()?;
    m.add_class::<InfillOptimizer>()?;
    m.add_class::<XType>()?;
    m.add_class::<XSpec>()?;
    m.add_class::<OptimResult>()?;
    m.add_class::<ExpectedOptimum>()?;
    m.add_class::<Recombination>()?;

    // Surrogate Model
    m.add_class::<GpMix>()?;
    m.add_class::<Gpx>()?;

    // Optimizer
    m.add_class::<Egor>()?;

    Ok(())
}
