use argmin::core::{ArgminFloat, Problem, State, TerminationReason};
use egobox_doe::Lhs;
use egobox_moe::Clustering;
use linfa::Float;
use ndarray::{Array1, Array2};
use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    pub cost: F,
    /// Previous cost function value
    pub prev_cost: F,
    /// Current best cost function value
    pub best_cost: F,
    /// Previous best cost function value
    pub prev_best_cost: F,
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
    pub no_point_added_retries: usize,
    pub clusterings: Option<Vec<Option<Clustering>>>,
    pub data: Option<(Array2<F>, Array2<F>)>,
    pub sampling: Option<Lhs<F, Xoshiro256Plus>>,
}

impl<F> EgorState<F>
where
    Self: State<Float = F>,
    F: Float + ArgminFloat,
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
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// # let state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
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
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// # let state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
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
    /// # use argmin::core::{LinearProgramState, State};
    /// # let state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
    /// # let cost_old = 1.0f64;
    /// # let state = state.cost(cost_old);
    /// # assert_eq!(state.prev_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.cost.to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # let cost = 0.0f64;
    /// let state = state.cost(cost);
    /// # assert_eq!(state.prev_cost.to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.cost.to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn cost(mut self, cost: F) -> Self {
        std::mem::swap(&mut self.prev_cost, &mut self.cost);
        self.cost = cost;
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
}

const MAX_RETRY: usize = 3;

impl<F> State for EgorState<F>
where
    F: Float + ArgminFloat,
{
    /// Type of parameter vector
    type Param = Array1<F>;
    /// Floating point precision
    type Float = F;

    /// Create new `LinearProgramState` instance
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate instant;
    /// # use instant;
    /// # use std::collections::HashMap;
    /// # use argmin::core::TerminationReason;
    /// use argmin::core::{LinearProgramState, State};
    /// let state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
    ///
    /// # assert!(state.param.is_none());
    /// # assert!(state.prev_param.is_none());
    /// # assert!(state.best_param.is_none());
    /// # assert!(state.prev_best_param.is_none());
    /// # assert_eq!(state.cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.prev_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.best_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.prev_best_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.target_cost.to_ne_bytes(), f64::NEG_INFINITY.to_ne_bytes());
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
            cost: Self::Float::infinity(),
            prev_cost: Self::Float::infinity(),
            best_cost: Self::Float::infinity(),
            prev_best_cost: Self::Float::infinity(),
            target_cost: Self::Float::neg_infinity(),
            iter: 0,
            last_best_iter: 0,
            max_iters: std::u64::MAX,
            counts: HashMap::new(),
            time: Some(instant::Duration::new(0, 0)),
            termination_reason: TerminationReason::NotTerminated,

            added: 0,
            prev_added: 0,
            no_point_added_retries: MAX_RETRY,
            clusterings: None,
            data: None,
            sampling: None,
        }
    }

    /// Checks if the current parameter vector is better than the previous best parameter value. If
    /// a new best parameter vector was found, the state is updated accordingly.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
    ///
    /// // Simulating a new, better parameter vector
    /// state.best_param = Some(vec![1.0f64]);
    /// state.best_cost = 10.0;
    /// state.param = Some(vec![2.0f64]);
    /// state.cost = 5.0;
    ///
    /// // Calling update
    /// state.update();
    ///
    /// // Check if update was successful
    /// assert_eq!(state.best_param.as_ref().unwrap()[0], 2.0f64);
    /// assert_eq!(state.best_cost.to_ne_bytes(), state.best_cost.to_ne_bytes());
    /// assert!(state.is_best());
    /// ```
    ///
    /// For algorithms which do not compute the cost function, every new parameter vector will be
    /// the new best:
    ///
    /// ```
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
    ///
    /// // Simulating a new, better parameter vector
    /// state.best_param = Some(vec![1.0f64]);
    /// state.param = Some(vec![2.0f64]);
    ///
    /// // Calling update
    /// state.update();
    ///
    /// // Check if update was successful
    /// assert_eq!(state.best_param.as_ref().unwrap()[0], 2.0f64);
    /// assert_eq!(state.best_cost.to_ne_bytes(), state.best_cost.to_ne_bytes());
    /// assert!(state.is_best());
    /// ```
    fn update(&mut self) {
        // check if parameters are the best so far
        // Comparison is done using `<` to avoid new solutions with the same cost function value as
        // the current best to be accepted. However, some solvers to not compute the cost function
        // value (such as the Newton method). Those will always have `Inf` cost. Therefore if both
        // the new value and the previous best value are `Inf`, the solution is also accepted. Care
        // is taken that both `Inf` also have the same sign.
        if self.cost < self.best_cost
            || (self.cost.is_infinite()
                && self.best_cost.is_infinite()
                && self.cost.is_sign_positive() == self.best_cost.is_sign_positive())
        {
            let param = (*self.param.as_ref().unwrap()).clone();
            let cost = self.cost;
            std::mem::swap(&mut self.prev_best_param, &mut self.best_param);
            self.best_param = Some(param);
            std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
            self.best_cost = cost;
            self.last_best_iter = self.iter;
        }
    }

    /// Returns a reference to the current parameter vector
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
    /// # assert!(state.param.is_none());
    /// # state.param = Some(vec![1.0, 2.0]);
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
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
    /// # assert!(state.best_param.is_none());
    /// # state.best_param = Some(vec![1.0, 2.0]);
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
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat, TerminationReason};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
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
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat, TerminationReason};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
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
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
    /// # state.cost = 12.0;
    /// let cost = state.get_cost();
    /// # assert_eq!(cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_cost(&self) -> Self::Float {
        self.cost
    }

    /// Returns current best cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
    /// # state.best_cost = 12.0;
    /// let best_cost = state.get_best_cost();
    /// # assert_eq!(best_cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_best_cost(&self) -> Self::Float {
        self.best_cost
    }

    /// Returns target cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
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
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
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
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
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
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
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
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat, TerminationReason};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
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
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
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
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
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
    /// # use argmin::core::{Problem, LinearProgramState, State, ArgminFloat};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
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
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
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
    /// # use argmin::core::{LinearProgramState, State, ArgminFloat};
    /// # let mut state: LinearProgramState<Vec<f64>, f64> = LinearProgramState::new();
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
