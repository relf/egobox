use crate::errors::Result;
use crate::types::*;
use crate::utils::find_best_result_index_from;
use crate::EgorSolver;

use egobox_gp::ThetaTuning;
use egobox_moe::MixtureGpSurrogate;
use ndarray::{s, Array, Array1, Array2, ArrayBase, Axis, Data, Ix1, RemoveAxis};
use serde::de::DeserializeOwned;

use ndarray_rand::rand::seq::SliceRandom;

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
    /// active may be longer than values
    pub(crate) fn setx(xcoop: &mut [f64], active: &[usize], values: &[f64]) {
        std::iter::zip(&active[..values.len()], values).for_each(|(&i, &xi)| xcoop[i] = xi)
    }

    /// Get active components from given ndarray following given axis
    /// active may contain out of range indices meaning it should be ignore
    pub(crate) fn getx<A, D>(xcoop: &Array<A, D>, axis: Axis, active: &[usize]) -> Array<A, D>
    where
        A: Clone,
        D: RemoveAxis,
    {
        let size = xcoop.len_of(axis);
        let selection = active
            .iter()
            .filter(|&&i| i < size)
            .cloned()
            .collect::<Vec<usize>>();
        xcoop.select(axis, &selection)
    }

    /// Compute array of components indices each row being used as
    /// active component during partial optimizations
    /// Array is (group nb, group size) where nb and size are computed
    /// from nx dimension and n_coop configuration.  
    pub(crate) fn get_random_activity(&mut self) -> Array2<usize> {
        let xdim = self.xlimits.nrows();
        let g_size = xdim / self.config.coego.n_coop.max(1);
        let g_nb = if xdim % self.config.coego.n_coop == 0 {
            xdim / g_size
        } else {
            xdim / g_size + 1
        }
        .min(xdim);

        // When n_coop is not a diviser of xdim, indice is set to xdim
        // (ie out of range) as to be filtered when handling last activity row
        let mut idx: Vec<usize> = (0..xdim).collect();
        idx.shuffle(&mut self.rng);
        let mut indices = vec![xdim; g_size * g_nb];
        indices[..xdim].copy_from_slice(&idx);

        Array2::from_shape_vec((g_nb, g_size), indices.to_vec()).unwrap()
    }

    /// Returns activity when optimization is not partial, that is
    /// all components are activated hence the result is an (1, nx) array
    /// containing [0, nx-1] integers.
    pub(crate) fn full_activity(&self) -> Array2<usize> {
        Array2::from_shape_vec(
            (1, self.xlimits.nrows()),
            (0..self.xlimits.nrows()).collect(),
        )
        .unwrap()
    }

    /// Set intial partial theta tunings from active components
    /// and optional set of initial values.
    /// Use during initialisation of theta partial optimizations
    pub(crate) fn set_initial_partial_theta_tuning(
        shape: (usize, usize),
        active: &[usize],
        theta_inits: Option<Array2<f64>>,
    ) -> Vec<ThetaTuning<f64>> {
        let default_init = Array2::from_elem(shape, ThetaTuning::<f64>::DEFAULT_INIT);
        theta_inits
            .clone()
            .unwrap_or(default_init)
            .outer_iter()
            .map(|init| ThetaTuning::Partial {
                init: init.to_owned(),
                bounds: Array1::from_vec(vec![ThetaTuning::<f64>::DEFAULT_BOUNDS; init.len()]),
                active: Self::strip(active, init.len()),
            })
            .collect()
    }

    /// Set partial theta tuning from previous theta tuning and given activity
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
                    active: Self::strip(active, init.len()),
                },
                ThetaTuning::Full { init, bounds } => ThetaTuning::Partial {
                    init: init.clone(),
                    bounds: bounds.clone(),
                    active: Self::strip(active, init.len()),
                },
                ThetaTuning::Partial {
                    init,
                    bounds,
                    active: _,
                } => ThetaTuning::Partial {
                    init: init.clone(),
                    bounds: bounds.clone(),
                    active: Self::strip(active, init.len()),
                },
            };
        });
    }

    /// Used to remove out of range indices from activity last row
    /// Indeed the last row of activity matrix may be incomplete
    /// as n_coop might not be a divider of nx. so this last row
    /// may contain indices with an 'nx value' used as a marker
    /// meaning to discard the indice information while keeping
    /// rows of the same length.
    fn strip(active: &[usize], dim: usize) -> Vec<usize> {
        active.iter().filter(|&&i| i < dim).cloned().collect()
    }

    /// Check whether the objective is improved using surrogates at xcoop location
    /// Given a current best information, surrogates are used
    /// to predict objective and constraints at xcoop location
    /// to find which one is best.
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

    /// Compute predicted objective and constraints values at the given x location.
    /// An optimistic approach is used as lower bound is taken for the objective
    /// and upper bound for constraints
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
        }
        Ok(Array1::from_vec(res))
    }
}
