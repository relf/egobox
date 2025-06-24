use crate::types::{XSpec, XType};
//use ndarray::{array, concatenate, Array1, Array2, ArrayView2, Axis};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

//use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

#[derive(FromPyObject)]
pub(crate) enum Domain<'py> {
    Xlists(Vec<Vec<f64>>),
    Xrows(PyReadonlyArray2<'py, f64>),
    Xspecs(Vec<XSpec>),
}

impl Domain<'_> {
    /// Returns true if the domain is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Domain::Xlists(v) => v.is_empty(),
            Domain::Xrows(arr) => arr.shape()[0] == 0 || arr.shape()[1] == 0,
            Domain::Xspecs(v) => v.is_empty(),
        }
    }

    /// Returns the dimensionality of the domain, if available.
    pub fn ndim(&self) -> Option<usize> {
        match self {
            Domain::Xlists(v) => v.first().map(|row| row.len()),
            Domain::Xrows(arr) => Some(arr.shape()[1]),
            Domain::Xspecs(v) => Some(v.len()),
        }
    }
}

/// Translate Python domain specifications into a vector of `XType`
pub(crate) fn parse(py: Python, xspecs: PyObject) -> Vec<egobox_ego::XType> {
    let domain: Domain = xspecs.extract(py).expect("Error in xspecs conversion");
    if domain.is_empty() {
        panic!("Error: domain argument cannot be empty")
    }
    if domain.ndim().is_none() {
        panic!("Error: domain argument badly formed")
    }

    match domain {
        Domain::Xspecs(xspecs) => xtypes_from_xspecs(xspecs),
        Domain::Xrows(xlimits) => xtypes_from_ndarray(xlimits),
        Domain::Xlists(floats) => xtypes_from_floats(floats),
    }
}

fn xtypes_from_floats(floats: Vec<Vec<f64>>) -> Vec<egobox_ego::XType> {
    let xtypes: Vec<egobox_ego::XType> = floats
        .iter()
        .map(|v| egobox_ego::XType::Float(v[0], v[1]))
        .collect();
    xtypes
}

fn xtypes_from_ndarray(xlimits: PyReadonlyArray2<f64>) -> Vec<egobox_ego::XType> {
    let ary = xlimits.as_array();
    let xtypes = ary.outer_iter().fold(Vec::new(), |mut acc, row| {
        acc.push(egobox_ego::XType::Float(row[0], row[1]));
        acc
    });
    xtypes
}

fn xtypes_from_xspecs(xspecs: Vec<XSpec>) -> Vec<egobox_ego::XType> {
    let xtypes: Vec<egobox_ego::XType> = xspecs
        .iter()
        .map(|spec| match spec.xtype {
            XType::Float => egobox_ego::XType::Float(spec.xlimits[0], spec.xlimits[1]),
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
    xtypes
}
