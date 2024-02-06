use crate::types::*;
use egobox_doe::{LhsKind, SamplingMethod};
use egobox_ego::MixintContext;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

#[pyclass(rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy)]
pub enum Sampling {
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

    let mixin = MixintContext::new(&xtypes);
    let doe = match method {
        Sampling::Lhs => Box::new(mixin.create_lhs_sampling(LhsKind::default(), seed))
            as Box<dyn SamplingMethod<_>>,
        Sampling::LhsClassic => Box::new(mixin.create_lhs_sampling(LhsKind::Classic, seed))
            as Box<dyn SamplingMethod<_>>,
        Sampling::LhsMaximin => Box::new(mixin.create_lhs_sampling(LhsKind::Maximin, seed))
            as Box<dyn SamplingMethod<_>>,
        Sampling::LhsCentered => Box::new(mixin.create_lhs_sampling(LhsKind::Centered, seed))
            as Box<dyn SamplingMethod<_>>,
        Sampling::LhsCenteredMaximin => {
            Box::new(mixin.create_lhs_sampling(LhsKind::CenteredMaximin, seed))
                as Box<dyn SamplingMethod<_>>
        }
        Sampling::FullFactorial => Box::new(mixin.create_ffact_sampling()),
        Sampling::Random => {
            Box::new(mixin.create_rand_sampling(seed)) as Box<dyn SamplingMethod<_>>
        }
    }
    .sample(n_samples);
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
