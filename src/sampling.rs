use crate::types::*;
use egobox_doe::{LhsKind, SamplingMethod};
use egobox_ego::MixintContext;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

#[pyclass(rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy)]
pub(crate) enum Sampling {
    Lhs = 1,
    FullFactorial = 2,
    Random = 3,
    LhsClassic = 4,
    LhsCentered = 5,
    LhsMaximin = 6,
    LhsCenteredMaximin = 7,
}

/// Samples generation using given method
///
/// # Parameters
///     method: LHS, FULL_FACTORIAL or RANDOM
///     xspecs: list of XSpec
///     n_samples: number of samples
///     seed: random seed
///
/// # Returns
///    ndarray of shape (n_samples, n_variables)
///
#[pyfunction]
pub fn sampling(
    py: Python,
    method: Sampling,
    xspecs: PyObject,
    n_samples: usize,
    seed: Option<u64>,
) -> &PyArray2<f64> {
    let specs: Vec<XSpec> = xspecs.extract(py).expect("Error in xspecs conversion");
    if specs.is_empty() {
        panic!("Error: xspecs argument cannot be empty")
    }
    let xtypes: Vec<egobox_ego::XType> = specs
        .iter()
        .map(|spec| match spec.xtype {
            XType::Float => egobox_ego::XType::Cont(spec.xlimits[0], spec.xlimits[1]),
            XType::Int => egobox_ego::XType::Int(spec.xlimits[0] as i32, spec.xlimits[1] as i32),
            XType::Ord => egobox_ego::XType::Ord(spec.xlimits.clone()),
            XType::Enum => {
                if spec.tags.is_empty() {
                    egobox_ego::XType::Enum(spec.xlimits[0] as usize)
                } else {
                    egobox_ego::XType::Enum(spec.tags.len())
                }
            }
        })
        .collect();

    let doe = match method {
        Sampling::Lhs => MixintContext::new(&xtypes)
            .create_lhs_sampling(LhsKind::default(), seed)
            .sample(n_samples),
        Sampling::LhsClassic => MixintContext::new(&xtypes)
            .create_lhs_sampling(LhsKind::Classic, seed)
            .sample(n_samples),
        Sampling::LhsMaximin => MixintContext::new(&xtypes)
            .create_lhs_sampling(LhsKind::Maximin, seed)
            .sample(n_samples),
        Sampling::LhsCentered => MixintContext::new(&xtypes)
            .create_lhs_sampling(LhsKind::Centered, seed)
            .sample(n_samples),
        Sampling::LhsCenteredMaximin => MixintContext::new(&xtypes)
            .create_lhs_sampling(LhsKind::CenteredMaximin, seed)
            .sample(n_samples),
        Sampling::FullFactorial => egobox_ego::MixintContext::new(&xtypes)
            .create_ffact_sampling()
            .sample(n_samples),
        Sampling::Random => egobox_ego::MixintContext::new(&xtypes)
            .create_rand_sampling(seed)
            .sample(n_samples),
    };
    doe.into_pyarray(py)
}

/// Samples generation using optimized Latin Hypercube Sampling
///
/// # Parameters
///     xspecs: list of XSpec
///     n_samples: number of samples
///     seed: random seed
///
/// # Returns
///    ndarray of shape (n_samples, n_variables)
///
#[pyfunction]
pub(crate) fn lhs(
    py: Python,
    xspecs: PyObject,
    n_samples: usize,
    seed: Option<u64>,
) -> &PyArray2<f64> {
    sampling(py, Sampling::Lhs, xspecs, n_samples, seed)
}
