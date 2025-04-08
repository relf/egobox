use crate::types::*;
use crate::EgorSolver;

use egobox_gp::ThetaTuning;
use ndarray::Array1;
use ndarray::Array2;
use serde::de::DeserializeOwned;

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + DeserializeOwned,
    C: CstrFn,
{
    /// Set active components to xcoop using xopt values
    /// active and values must have the same size
    pub(crate) fn setx(xcoop: &mut [f64], active: &[usize], values: &[f64]) {
        std::iter::zip(active, values).for_each(|(&i, &xi)| xcoop[i] = xi)
    }

    pub(crate) fn full_activity(&self) -> Array2<usize> {
        Array2::from_shape_vec(
            (1, self.xlimits.nrows()),
            (0..self.xlimits.nrows()).collect(),
        )
        .unwrap()
    }

    pub(crate) fn set_partial_theta_tuning(
        &self,
        active: &[usize],
        theta_tunings: &mut [ThetaTuning<f64>],
    ) {
        theta_tunings.iter_mut().for_each(|theta| {
            *theta = match theta {
                ThetaTuning::Fixed(init) => ThetaTuning::Partial {
                    init: init.clone(),
                    bounds: Array1::from_vec(vec![ThetaTuning::<f64>::DEFAULT_BOUNDS; init.len()]),
                    active: active.to_vec(),
                },
                ThetaTuning::Full { init, bounds } => ThetaTuning::Partial {
                    init: init.clone(),
                    bounds: bounds.clone(),
                    active: active.to_vec(),
                },
                ThetaTuning::Partial {
                    init,
                    bounds,
                    active: _,
                } => ThetaTuning::Partial {
                    init: init.clone(),
                    bounds: bounds.clone(),
                    active: active.to_vec(),
                },
            };
        });
    }
}
