#![doc = include_str!("../README.md")]

pub(crate) mod domain;
mod egor;
mod gp_config;
mod gp_mix;
mod sampling;
mod sparse_gp_mix;
pub(crate) mod types;

use egor::*;
use gp_mix::*;
use sampling::*;
use sparse_gp_mix::*;
use types::*;

use env_logger::{Builder, Env};
use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

#[doc(hidden)]
#[pymodule]
fn egobox(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    let env = Env::new().filter_or("EGOBOX_LOG", "info");
    let mut builder = Builder::from_env(env);
    let builder = builder.target(env_logger::Target::Stdout);
    builder.try_init().ok();

    // utils
    m.add_function(wrap_pyfunction!(to_specs, m)?)?;
    m.add_function(wrap_pyfunction!(lhs, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::sampling, m)?)?;

    // types
    m.add_class::<sampling::Sampling>()?;
    m.add_class::<gp_config::GpConfig>()?;
    m.add_class::<RegressionSpec>()?;
    m.add_class::<CorrelationSpec>()?;
    m.add_class::<InfillStrategy>()?;
    m.add_class::<ConstraintStrategy>()?;
    m.add_class::<QInfillStrategy>()?;
    m.add_class::<InfillOptimizer>()?;
    m.add_class::<XType>()?;
    m.add_class::<XSpec>()?;
    m.add_class::<OptimResult>()?;
    m.add_class::<ExpectedOptimum>()?;
    m.add_class::<Recombination>()?;

    // Surrogate Model
    m.add_class::<GpMix>()?;
    m.add_class::<Gpx>()?;
    m.add_class::<SparseGpMix>()?;
    m.add_class::<SparseGpx>()?;
    m.add_class::<SparseMethod>()?;

    // Optimizer
    m.add_class::<Egor>()?;

    Ok(())
}

// Define a function to gather stub information.
define_stub_info_gatherer!(stub_info);
