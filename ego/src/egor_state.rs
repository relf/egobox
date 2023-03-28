/// Implementation of `argmin::IterState` for Egor optimizer
use crate::sort_axis::*;
use argmin::core::{ArgminFloat, Problem, State, TerminationReason, TerminationStatus};
use egobox_doe::Lhs;
use egobox_moe::Clustering;
use linfa::Float;
use ndarray::{array, s, Array1, Array2, ArrayBase, Axis, Data, Ix2, Zip};
use ndarray_stats::QuantileExt;
use rand_xoshiro::Xoshiro256Plus;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Max number of retry when adding a new point. Point addition may fail
/// if new point is too close to a previous point in the growing doe used
/// to train surrogate models modeling objective and constraints functions.
pub const MAX_POINT_ADDITION_RETRY: i32 = 3;

/// Find best (eg minimal) cost value (y_data\[0\]) with valid constraints (y_data\[1..\] < cstr_tol).
/// y_data containing ns samples [objective, cstr_1, ... cstr_nc] is given as a matrix (ns, nc + 1)  
pub fn find_best_result_index<F: Float>(
    y_data: &ArrayBase<impl Data<Elem = F>, Ix2>,
    cstr_tol: F,
) -> usize {
    if y_data.ncols() > 1 {
        // Compute sum of violated constraints
        let cstrs = y_data.slice(s![.., 1..]);
        let mut c_obj = Array2::zeros((y_data.nrows(), 2));

        Zip::from(c_obj.rows_mut())
            .and(cstrs.rows())
            .and(y_data.slice(s![.., 0]))
            .for_each(|mut c_obj_row, c_row, obj| {
                let c_sum = c_row
                    .to_owned()
                    .into_iter()
                    .filter(|c| *c > cstr_tol)
                    .fold(F::zero(), |acc, c| acc + (c - cstr_tol).abs());
                c_obj_row.assign(&array![c_sum, *obj]);
            });
        let min_csum_index = c_obj.slice(s![.., 0]).argmin().ok();

        if let Some(min_index) = min_csum_index {
            if c_obj[[min_index, 0]] > F::zero() {
                // There is no feasible point take minimal cstr sum as the best one
                min_index
            } else {
                // There is at least one or several point feasible, take the minimal objective among them
                let mut index = 0;
                let mut y_best = F::infinity();
                Zip::indexed(c_obj.rows()).for_each(|i, c_o| {
                    if c_o[0] == F::zero() && c_o[1] < y_best {
                        y_best = c_o[1];
                        index = i;
                    }
                });
                index
            }
        } else {
            // Take min obj without looking at constraints
            let mut index = 0;

            // sort regardoing minimal objective
            let perm = y_data.sort_axis_by(Axis(0), |i, j| y_data[[i, 0]] < y_data[[j, 0]]);
            let y_sort = y_data.to_owned().permute_axis(Axis(0), &perm);

            // Take the first one which do not violate constraints
            for (i, row) in y_sort.axis_iter(Axis(0)).enumerate() {
                if !row.slice(s![1..]).iter().any(|v| *v > cstr_tol) {
                    index = i;
                    break;
                }
            }
            perm.indices[index]
        }
    } else {
        // unconstrained optimization
        y_data.column(0).argmin().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_find_best_obj() {
        // respect constraint (0, 1, 2) and minimize obj (1)
        let ydata = array![[1.0, -0.15], [-1.0, -0.01], [2.0, -0.2], [-3.0, 2.0]];
        let cstr_tol = 0.1;
        assert_abs_diff_eq!(1, find_best_result_index(&ydata, cstr_tol));

        // respect constraint (0, 1, 2) and minimize obj (2)
        let ydata = array![[1.0, -0.15], [-1.0, -0.01], [-2.0, -0.2], [-3.0, 2.0]];
        let cstr_tol = 0.1;
        assert_abs_diff_eq!(2, find_best_result_index(&ydata, cstr_tol));

        // all out of tolerance => minimize constraint overshoot sum (0)
        let ydata = array![[1.0, 0.15], [-1.0, 0.3], [2.0, 0.2], [-3.0, 2.0]];
        let cstr_tol = 0.1;
        assert_abs_diff_eq!(0, find_best_result_index(&ydata, cstr_tol));

        // all in tolerance => min obj
        let ydata = array![[1.0, 0.15], [-1.0, 0.3], [2.0, 0.2], [-3.0, 2.0]];
        let cstr_tol = 3.0;
        assert_abs_diff_eq!(3, find_best_result_index(&ydata, cstr_tol));

        // unconstrained => min obj
        let ydata = array![[1.0], [-1.0], [2.0], [-3.0]];
        let cstr_tol = 0.1;
        assert_abs_diff_eq!(3, find_best_result_index(&ydata, cstr_tol));
    }
}

/// Maintains the state from iteration to iteration of the [crate::EgorSolver].
///
/// This struct is passed from one iteration of an algorithm to the next.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EgorState<F: Float> {
    /// Current parameter vector
    pub param: Option<Array1<F>>,
    /// Previous parameter vector
    pub prev_param: Option<Array1<F>>,
    /// Current best parameter vector
    pub best_param: Option<Array1<F>>,
    /// Previous best parameter vector
    pub prev_best_param: Option<Array1<F>>,

    /// Current cost function value
    /// The first component is the actual cost value
    /// while the remaining ones are the constraints values
    pub cost: Option<Array1<F>>,
    /// Previous cost function value
    pub prev_cost: Option<Array1<F>>,
    /// Current best cost function value
    pub best_cost: Option<Array1<F>>,
    /// Previous best cost function value
    pub prev_best_cost: Option<Array1<F>>,
    /// Target cost function value
    pub target_cost: F,

    /// Current iteration
    pub iter: u64,
    /// Iteration number of last best cost
    pub last_best_iter: u64,
    /// Maximum number of iterations
    pub max_iters: u64,
    /// Evaluation counts
    pub counts: HashMap<String, u64>,
    /// Time required so far
    pub time: Option<instant::Duration>,
    /// Optimization status
    pub termination_status: TerminationStatus,

    /// Initial doe size
    pub(crate) doe_size: usize,
    /// Number of added points
    pub(crate) added: usize,
    /// Previous number of added points
    pub(crate) prev_added: usize,
    /// Current number of retry without adding point
    pub(crate) no_point_added_retries: i32,
    /// run_lhs_optim
    pub(crate) lhs_optim: bool,

    /// Current clusterings for objective and constraints mixture surrogate models
    pub(crate) clusterings: Option<Vec<Option<Clustering>>>,
    /// Historic data (params, objective and constraints)
    pub(crate) data: Option<(Array2<F>, Array2<F>)>,
    /// Sampling method used to generate space filling samples
    pub(crate) sampling: Option<Lhs<F, Xoshiro256Plus>>,
    /// Constraint tolerance cstr < cstr_tol.
    /// It used to assess the validity of the param point and hence the corresponding cost
    pub(crate) cstr_tol: F,
}

impl<F> EgorState<F>
where
    Self: State<Float = F>,
    F: Float,
{
    /// Set parameter vector. This shifts the stored parameter vector to the previous parameter
    /// vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State};
    /// # let state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # let param_old = vec![1.0f64, 2.0f64];
    /// # let state = state.param(param_old);
    /// # assert!(state.prev_param.is_none());
    /// # assert_eq!(state.param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let param = vec![0.0f64, 3.0f64];
    /// let state = state.param(param);
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn param(mut self, param: Array1<F>) -> Self {
        std::mem::swap(&mut self.prev_param, &mut self.param);
        self.param = Some(param);
        self
    }

    /// Set target cost.
    ///
    /// When this cost is reached, the algorithm will stop. The default is
    /// `Self::Float::NEG_INFINITY`.
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego    ::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let state: EgorState<f64> = EgorState::new();
    /// # assert_eq!(state.target_cost.to_ne_bytes(), f64::NEG_INFINITY.to_ne_bytes());
    /// let state = state.target_cost(0.0);
    /// # assert_eq!(state.target_cost.to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn target_cost(mut self, target_cost: F) -> Self {
        self.target_cost = target_cost;
        self
    }

    /// Set maximum number of iterations
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let state: EgorState<f64> = EgorState::new();
    /// # assert_eq!(state.max_iters, std::u64::MAX);
    /// let state = state.max_iters(1000);
    /// # assert_eq!(state.max_iters, 1000);
    /// ```
    #[must_use]
    pub fn max_iters(mut self, iters: u64) -> Self {
        self.max_iters = iters;
        self
    }

    /// Set the current cost function value. This shifts the stored cost function value to the
    /// previous cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use argmin::core::State;
    /// # use egobox_ego::EgorState;
    /// # let state: EgorState<f64> = EgorState::new();
    /// # let cost_old = 1.0f64;
    /// # let state = state.cost(array![cost_old]);
    /// # assert!(state.prev_cost.is_none());
    /// # assert_eq!(state.cost.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # let cost = 0.0f64;
    /// let state = state.cost(array![cost]);
    /// # assert_eq!(state.prev_cost.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.cost.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn cost(mut self, cost: Array1<F>) -> Self {
        std::mem::swap(&mut self.prev_cost, &mut self.cost);
        self.cost = Some(cost);
        self
    }

    /// Set the current clusterings used by surrogate models
    pub fn clusterings(mut self, clustering: Vec<Option<Clustering>>) -> Self {
        self.clusterings = Some(clustering);
        self
    }

    /// Moves the current clusterings out and replaces it internally with `None`.
    pub fn take_clusterings(&mut self) -> Option<Vec<Option<Clustering>>> {
        self.clusterings.take()
    }

    /// Set the current data points as training points for the surrogate models
    /// These points are gradually selected by the EGO algorithm regarding an infill criterion.
    /// Data is expressed as a couple (xdata, ydata) where xdata is a (p, nx matrix)
    /// and ydata is a (p, 1 + nb of cstr) matrix and ydata_i = fcost(xdata_i) for i in [1, p].  
    pub fn data(mut self, data: (Array2<F>, Array2<F>)) -> Self {
        self.data = Some(data);
        self
    }

    /// Moves the current data out and replaces it internally with `None`.
    pub fn take_data(mut self) -> Option<(Array2<F>, Array2<F>)> {
        self.data.take()
    }

    /// Set the sampling method used to draw random points
    /// The sampling method is saved as part of the state to allow reproducible
    /// optimization.   
    pub fn sampling(mut self, sampling: Lhs<F, Xoshiro256Plus>) -> Self {
        self.sampling = Some(sampling);
        self
    }

    /// Moves the current sampling out and replaces it internally with `None`.
    pub fn take_sampling(mut self) -> Option<Lhs<F, Xoshiro256Plus>> {
        self.sampling.take()
    }

    /// Returns current cost (ie objective) function and constraint values.
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.cost = Some(array![12.0, 0.1]);
    /// let cost = state.get_full_cost();
    /// # assert_eq!(cost.unwrap()[0].to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// # assert_eq!(cost.unwrap()[1].to_ne_bytes(), 0.1f64.to_ne_bytes());
    /// ```
    pub fn get_full_cost(&self) -> Option<&Array1<F>> {
        self.cost.as_ref()
    }

    /// Returns current cost (ie objective) function and constraint values.
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.best_cost = Some(array![12.0, 0.1]);
    /// let cost = state.get_full_best_cost();
    /// # assert_eq!(cost.unwrap()[0].to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// # assert_eq!(cost.unwrap()[1].to_ne_bytes(), 0.1f64.to_ne_bytes());
    /// ```
    pub fn get_full_best_cost(&self) -> Option<&Array1<F>> {
        self.best_cost.as_ref()
    }
}

impl<F> State for EgorState<F>
where
    F: Float + ArgminFloat,
{
    /// Type of parameter vector
    type Param = Array1<F>;
    /// Floating point precision
    type Float = F;

    /// Create new `EgorState` instance
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate instant;
    /// # use instant;
    /// # use std::collections::HashMap;
    /// # use argmin::core::{State, TerminationStatus};
    /// use egobox_ego::EgorState;
    /// let state: EgorState<f64> = EgorState::new();
    ///
    /// # assert!(state.param.is_none());
    /// # assert!(state.prev_param.is_none());
    /// # assert!(state.best_param.is_none());
    /// # assert!(state.prev_best_param.is_none());
    /// # assert!(state.cost.is_none());
    /// # assert!(state.prev_cost.is_none());
    /// # assert!(state.best_cost.is_none());
    /// # assert!(state.prev_best_cost.is_none());
    /// # assert_eq!(state.target_cost, f64::NEG_INFINITY);
    /// # assert_eq!(state.iter, 0);
    /// # assert_eq!(state.last_best_iter, 0);
    /// # assert_eq!(state.max_iters, std::u64::MAX);
    /// # assert_eq!(state.counts, HashMap::new());
    /// # assert_eq!(state.time.unwrap(), instant::Duration::new(0, 0));
    /// # assert_eq!(state.termination_status, TerminationStatus::NotTerminated);
    /// ```
    fn new() -> Self {
        EgorState {
            param: None,
            prev_param: None,
            best_param: None,
            prev_best_param: None,

            cost: None,
            prev_cost: None,
            best_cost: None,
            prev_best_cost: None,
            target_cost: F::neg_infinity(),

            iter: 0,
            last_best_iter: 0,
            max_iters: std::u64::MAX,
            counts: HashMap::new(),
            time: Some(instant::Duration::new(0, 0)),
            termination_status: TerminationStatus::NotTerminated,

            doe_size: 0,
            added: 0,
            prev_added: 0,
            no_point_added_retries: MAX_POINT_ADDITION_RETRY,
            lhs_optim: false,
            clusterings: None,
            data: None,
            sampling: None,
            cstr_tol: F::cast(1e-6),
        }
    }

    /// Checks if the current parameter vector is better than the previous best parameter value. If
    /// a new best parameter vector was found, the state is updated accordingly.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{State, ArgminFloat};
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    ///
    /// let mut state: EgorState<f64> = EgorState::new();
    ///
    /// // Simulating a new, better parameter vector
    /// let mut state = state.data((array![[1.0f64], [2.0f64]], array![[10.0],[5.0]]));
    /// state.iter = 2;
    /// state.param = Some(array![2.0f64]);
    /// state.cost = Some(array![5.0]);
    ///
    /// // Calling update
    /// state.update();
    ///
    /// // Check if update was successful
    /// assert_eq!(state.best_param.as_ref().unwrap()[0], 2.0f64);
    /// assert_eq!(state.best_cost.as_ref().unwrap()[0], 5.0);
    /// assert!(state.is_best());
    /// ```
    fn update(&mut self) {
        // TODO: better implementation should only track
        // current and best index in data and compare just them
        // without finding best in data each time
        let data = self.data.as_ref();
        match data {
            None => {
                // should not occur data should be some
                println!("Warning: update should occur after data initialization");
            }
            Some((x_data, y_data)) => {
                let best_index = find_best_result_index(y_data, self.cstr_tol);
                let best_iter = best_index.saturating_sub(self.doe_size) as u64 + 1;

                if best_iter > self.last_best_iter {
                    let param = x_data.row(best_index).to_owned();
                    std::mem::swap(&mut self.prev_best_param, &mut self.best_param);
                    self.best_param = Some(param);

                    let cost = y_data.row(best_index).to_owned();
                    std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
                    self.best_cost = Some(cost);
                    self.last_best_iter = best_iter;
                }
            }
        };
    }

    /// Returns a reference to the current parameter vector
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # assert!(state.param.is_none());
    /// # state.param = Some(array![1.0, 2.0]);
    /// # assert_eq!(state.param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let param = state.get_param();  // Option<&P>
    /// # assert_eq!(param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    fn get_param(&self) -> Option<&Array1<F>> {
        self.param.as_ref()
    }

    /// Returns a reference to the current best parameter vector
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    ///
    /// # let mut state: EgorState<f64> = EgorState::new();

    /// # assert!(state.best_param.is_none());
    /// # state.best_param = Some(array![1.0, 2.0]);
    /// # assert_eq!(state.best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let best_param = state.get_best_param();  // Option<&P>
    /// # assert_eq!(best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    fn get_best_param(&self) -> Option<&Array1<F>> {
        self.best_param.as_ref()
    }

    /// Sets the termination status to [`Terminated`](`TerminationStatus::Terminated`) with the given reason
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat, TerminationReason, TerminationStatus};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert_eq!(state.termination_status, TerminationStatus::NotTerminated);
    /// let state = state.terminate_with(TerminationReason::MaxItersReached);
    /// # assert_eq!(state.termination_status, TerminationStatus::Terminated(TerminationReason::MaxItersReached));
    /// ```
    fn terminate_with(mut self, reason: TerminationReason) -> Self {
        self.termination_status = TerminationStatus::Terminated(reason);
        self
    }

    /// Sets the time required so far.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate instant;
    /// # use instant;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat, TerminationReason};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// let state = state.time(Some(instant::Duration::new(0, 12)));
    /// # assert_eq!(state.time.unwrap(), instant::Duration::new(0, 12));
    /// ```
    fn time(&mut self, time: Option<instant::Duration>) -> &mut Self {
        self.time = time;
        self
    }

    /// Returns current cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.cost = Some(array![12.0]);
    /// let cost = state.get_cost();
    /// # assert_eq!(cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_cost(&self) -> Self::Float {
        match self.cost.as_ref() {
            Some(c) => *(c.get(0).unwrap_or(&Self::Float::infinity())),
            None => Self::Float::infinity(),
        }
    }

    /// Returns current best cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.best_cost = Some(array![12.0]);
    /// let best_cost = state.get_best_cost();
    /// # assert_eq!(best_cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_best_cost(&self) -> Self::Float {
        match self.best_cost.as_ref() {
            Some(c) => *(c.get(0).unwrap_or(&Self::Float::infinity())),
            None => Self::Float::infinity(),
        }
    }

    /// Returns target cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.target_cost = 12.0;
    /// let target_cost = state.get_target_cost();
    /// # assert_eq!(target_cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_target_cost(&self) -> Self::Float {
        self.target_cost
    }

    /// Returns current number of iterations.
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.iter = 12;
    /// let iter = state.get_iter();
    /// # assert_eq!(iter, 12);
    /// ```
    fn get_iter(&self) -> u64 {
        self.iter
    }

    /// Returns iteration number of last best parameter vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.last_best_iter = 12;
    /// let last_best_iter = state.get_last_best_iter();
    /// # assert_eq!(last_best_iter, 12);
    /// ```
    fn get_last_best_iter(&self) -> u64 {
        self.last_best_iter
    }

    /// Returns the maximum number of iterations.
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.max_iters = 12;
    /// let max_iters = state.get_max_iters();
    /// # assert_eq!(max_iters, 12);
    /// ```
    fn get_max_iters(&self) -> u64 {
        self.max_iters
    }

    /// Returns the termination status.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat, TerminationStatus};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// let termination_status = state.get_termination_status();
    /// # assert_eq!(*termination_status, TerminationStatus::NotTerminated);
    /// ```
    fn get_termination_status(&self) -> &TerminationStatus {
        &self.termination_status
    }

    /// Returns the termination reason if terminated, otherwise None.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat, TerminationReason};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// let termination_reason = state.get_termination_reason();
    /// # assert_eq!(termination_reason, None);
    /// ```
    fn get_termination_reason(&self) -> Option<&TerminationReason> {
        match &self.termination_status {
            TerminationStatus::Terminated(reason) => Some(reason),
            TerminationStatus::NotTerminated => None,
        }
    }

    /// Returns the time elapsed since the start of the optimization.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate instant;
    /// # use instant;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// let time = state.get_time();
    /// # assert_eq!(time.unwrap(), instant::Duration::new(0, 0));
    /// ```
    fn get_time(&self) -> Option<instant::Duration> {
        self.time
    }

    /// Increments the number of iterations by one
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # assert_eq!(state.iter, 0);
    /// state.increment_iter();
    /// # assert_eq!(state.iter, 1);
    /// ```
    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    /// Set all function evaluation counts to the evaluation counts of another `Problem`.
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{Problem, State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # assert_eq!(state.counts, HashMap::new());
    /// # state.counts.insert("test2".to_string(), 10u64);
    /// #
    /// # #[derive(Eq, PartialEq, Debug)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # let mut problem = Problem::new(UserDefinedProblem {});
    /// # problem.counts.insert("test1", 10u64);
    /// # problem.counts.insert("test2", 2);
    /// state.func_counts(&problem);
    /// # let mut hm = HashMap::new();
    /// # hm.insert("test1".to_string(), 10u64);
    /// # hm.insert("test2".to_string(), 2u64);
    /// # assert_eq!(state.counts, hm);
    /// ```
    fn func_counts<O>(&mut self, problem: &Problem<O>) {
        for (k, &v) in problem.counts.iter() {
            let count = self.counts.entry(k.to_string()).or_insert(0);
            *count = v
        }
    }

    /// Returns function evaluation counts
    ///
    /// # Example
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # assert_eq!(state.counts, HashMap::new());
    /// # state.counts.insert("test2".to_string(), 10u64);
    /// let counts = state.get_func_counts();
    /// # let mut hm = HashMap::new();
    /// # hm.insert("test2".to_string(), 10u64);
    /// # assert_eq!(*counts, hm);
    /// ```
    fn get_func_counts(&self) -> &HashMap<String, u64> {
        &self.counts
    }

    /// Returns whether the current parameter vector is also the best parameter vector found so
    /// far.
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.last_best_iter = 12;
    /// # state.iter = 12;
    /// let is_best = state.is_best();
    /// # assert!(is_best);
    /// # state.last_best_iter = 12;
    /// # state.iter = 21;
    /// # let is_best = state.is_best();
    /// # assert!(!is_best);
    /// ```
    fn is_best(&self) -> bool {
        self.last_best_iter == self.iter
    }
}
