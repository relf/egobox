use crate::types::*;
use egobox_doe::SamplingMethod;
use egobox_ego::MixintContext;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub enum Method {
    LHS = 0,
    #[allow(non_camel_case_types)]
    FULL_FACTORIAL = 1,
    RANDOM = 2,
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
    method: Method,
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

    let doe = match method {
        Method::LHS => MixintContext::new(&xtypes)
            .create_lhs_sampling(seed)
            .sample(n_samples),
        Method::FULL_FACTORIAL => egobox_ego::MixintContext::new(&xtypes)
            .create_ffact_sampling()
            .sample(n_samples),
        Method::RANDOM => egobox_ego::MixintContext::new(&xtypes)
            .create_rand_sampling(seed)
            .sample(n_samples),
    };
    doe.into_pyarray(py)
}

/// Samples generation using Latin Hypercube Sampling
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
    sampling(py, Method::LHS, xspecs, n_samples, seed)
}
