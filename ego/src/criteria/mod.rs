pub mod ei;
pub mod wb2;

pub use ei::{ExpectedImprovement, EI};
pub use wb2::{WB2Criterion, WB2, WB2S};

use dyn_clonable::*;
use egobox_moe::ClusteredSurrogate;
use ndarray::{Array1, ArrayView2};

/// A trait for infill criterion which maximmum location will
/// determine the next most promising point expected to be the
/// optimum location of the objective function
#[clonable]
#[typetag::serde(tag = "type")]
pub trait InfillCriterion: Clone + Sync {
    /// Criterion value at given point x with regards to given
    /// surrogate of the objectove function, the current found min
    /// and an optional acaling factor
    fn value(
        &self,
        x: &[f64],
        obj_model: &dyn ClusteredSurrogate,
        f_min: f64,
        scale: Option<f64>,
    ) -> f64;

    /// Derivatives wrt x components of the criterion value
    fn grad(
        &self,
        x: &[f64],
        obj_model: &dyn ClusteredSurrogate,
        f_min: f64,
        scale: Option<f64>,
    ) -> Array1<f64>;

    /// Scaling factor computation
    fn scaling(&self, x: &ArrayView2<f64>, obj_model: &dyn ClusteredSurrogate, f_min: f64) -> f64;
}
