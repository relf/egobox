use crate::errors::Result;
use crate::types::*;
use crate::utils::find_best_result_index_from;
use crate::EgorSolver;

use egobox_gp::ThetaTuning;
use egobox_moe::MixtureGpSurrogate;
use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix1};
use serde::de::DeserializeOwned;

#[cfg(not(feature = "nlopt"))]
use crate::types::ObjFn;
#[cfg(feature = "nlopt")]
use nlopt::ObjFn;

/// Whether GP objective improves when setting a component set
/// TODO: at the moment not sure improvement check is required, to be validated
pub const COEGO_IMPROVEMENT_CHECK: bool = false;
const CSTR_DOUBT: f64 = 3.;

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

    #[allow(clippy::type_complexity)]
    pub(crate) fn is_objective_improved(
        &self,
        current_best: &(f64, Array1<f64>, Array1<f64>, Array1<f64>),
        xcoop: &Array1<f64>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        cstr_tols: &Array1<f64>,
        cstr_funcs: &[&(dyn ObjFn<InfillObjData<f64>> + Sync)],
    ) -> (bool, (f64, Array1<f64>, Array1<f64>, Array1<f64>)) {
        let mut y_data = Array2::zeros((2, 1 + self.config.n_cstr));
        // Assign metamodelized objective (1) and constraints (n_cstr) values
        y_data.slice_mut(s![0, ..]).assign(&current_best.2);
        y_data
            .slice_mut(s![1, ..])
            .assign(&self.predict_point(xcoop, obj_model, cstr_models).unwrap());
        // Assign function constraints values
        let mut c_data = Array2::zeros((2, cstr_funcs.len()));
        c_data.slice_mut(s![0, ..]).assign(&current_best.3);
        let evals = self.eval_fcstrs(cstr_funcs, &xcoop.to_owned().insert_axis(Axis(0)));
        if !cstr_funcs.is_empty() {
            c_data.slice_mut(s![1, ..]).assign(&evals.row(0));
        }
        let best_index = find_best_result_index_from(0, 1, &y_data, &c_data, cstr_tols);

        let best = if best_index == 0 {
            current_best.clone()
        } else {
            (
                y_data[[best_index, 0]],
                xcoop.to_owned(),
                y_data.row(1).to_owned(),
                c_data.row(1).to_owned(),
            )
        };
        (best_index != 0, best)
    }

    /// Compute predicted objective and constraints values at the given x location
    pub(crate) fn predict_point(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
    ) -> Result<Array1<f64>> {
        let mut res: Vec<f64> = vec![];
        let x = &x.view().insert_axis(Axis(0));
        let sigma = obj_model.predict_var(&x.view()).unwrap()[[0, 0]].sqrt();
        // Use lower trust bound for a minimization
        let pred = obj_model.predict(x)?[0] - CSTR_DOUBT * sigma;
        res.push(pred);
        for cstr_model in cstr_models {
            let sigma = cstr_model.predict_var(&x.view()).unwrap()[[0, 0]].sqrt();
            // Use upper trust bound
            res.push(cstr_model.predict(x)?[0] + CSTR_DOUBT * sigma);
            res.push(cstr_model.predict(x)?[0]);
        }
        Ok(Array1::from_vec(res))
    }
}
