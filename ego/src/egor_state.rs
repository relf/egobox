use crate::sort_axis::*;
use argmin::core::{ArgminFloat, Problem, State, TerminationReason};
use egobox_doe::Lhs;
use egobox_moe::Clustering;
use linfa::Float;
use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_stats::QuantileExt;
use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Find best (eg minimal) cost value (y_data[0]) with valid constraints (y_data[1..] < cstr_tol).
/// y_data containing ns samples [objective, cstr_1, ... cstr_nc] is given as a matrix (ns, nc + 1)  
pub fn find_best_result_index<F: Float>(
    y_data: &ArrayBase<impl Data<Elem = F>, Ix2>,
    cstr_tol: F,
) -> usize {
    if y_data.ncols() > 1 {
        // constraint optimization
        let mut index = 0;
        let perm = y_data.sort_axis_by(Axis(0), |i, j| y_data[[i, 0]] < y_data[[j, 0]]);
        let y_sort = y_data.to_owned().permute_axis(Axis(0), &perm);
        for (i, row) in y_sort.axis_iter(Axis(0)).enumerate() {
            if !row.slice(s![1..]).iter().any(|v| *v > cstr_tol) {
                index = i;
                break;
            }
        }
        perm.indices[index]
    } else {
        // unconstrained optimization
        y_data.column(0).argmin().unwrap()
    }
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
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
    /// Reason of termination
    pub termination_reason: TerminationReason,

    /// Number of added points
    pub added: usize,
    /// Previous number of added points
    pub prev_added: usize,
    /// Current number of retry without adding point
    pub no_point_added_retries: i32,
    /// run_lhs_optim
    pub lhs_optim: bool,

    /// Current clusterings for objective and constraints mixture surrogate models
    pub clusterings: Option<Vec<Option<Clustering>>>,
    /// Historic data (params, objective and constraints)
    pub data: Option<(Array2<F>, Array2<F>)>,
    /// Sampling method used to generate space filling samples
    pub sampling: Option<Lhs<F, Xoshiro256Plus>>,
    /// Constraint tolerance cstr < cstr_tol.
    /// It used to assess the validity of the param point and hence the corresponding cost
    pub cstr_tol: F,
}

impl<F> EgorState<F>
where
    Self: State<Float = F>,
    F: Float,
{
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

    pub fn clusterings(mut self, clustering: Vec<Option<Clustering>>) -> Self {
        self.clusterings = Some(clustering);
        self
    }
    pub fn take_clusterings(&mut self) -> Option<Vec<Option<Clustering>>> {
        self.clusterings.take()
    }

    pub fn data(mut self, data: (Array2<F>, Array2<F>)) -> Self {
        self.data = Some(data);
        self
    }
    pub fn take_data(mut self) -> Option<(Array2<F>, Array2<F>)> {
        self.data.take()
    }

    pub fn sampling(mut self, sampling: Lhs<F, Xoshiro256Plus>) -> Self {
        self.sampling = Some(sampling);
        self
    }
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

pub const MAX_POINT_ADDITION_RETRY: i32 = 3;

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
    /// # use argmin::core::{State, TerminationReason};
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
    /// # assert_eq!(state.termination_reason, TerminationReason::NotTerminated);
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
            termination_reason: TerminationReason::NotTerminated,

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
        // TODO: better implementation should track only track
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

                let param = x_data.row(best_index).to_owned();
                std::mem::swap(&mut self.prev_best_param, &mut self.best_param);
                self.best_param = Some(param);

                let cost = y_data.row(best_index).to_owned();
                std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
                self.best_cost = Some(cost);
                self.last_best_iter = self.iter;
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

    /// Sets the termination reason (default: [`TerminationReason::NotTerminated`])
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat, TerminationReason};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # assert_eq!(state.termination_reason, TerminationReason::NotTerminated);
    /// let state = state.terminate_with(TerminationReason::MaxItersReached);
    /// # assert_eq!(state.termination_reason, TerminationReason::MaxItersReached);
    /// ```
    fn terminate_with(mut self, reason: TerminationReason) -> Self {
        self.termination_reason = reason;
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

    /// Returns the termination reason.
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat, TerminationReason};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// let termination_reason = state.get_termination_reason();
    /// # assert_eq!(termination_reason, TerminationReason::NotTerminated);
    /// ```
    fn get_termination_reason(&self) -> TerminationReason {
        self.termination_reason
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
