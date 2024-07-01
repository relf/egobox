use crate::solver::egor_solver::EgorSolver;
use crate::utils::{find_best_result_index, find_best_result_index_from};
use crate::{EgorState, MAX_POINT_ADDITION_RETRY};

use crate::types::*;
use crate::utils::update_data;

use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use log::{debug, info, warn};
use ndarray::{concatenate, s, Array1, Array2, Axis, Zip};
use ndarray_npy::{read_npy, write_npy};

use argmin::argmin_error_closure;
use argmin::core::{
    CostFunction, Problem, Solver, State, TerminationReason, TerminationStatus, KV,
};

use std::time::Instant;

/// Numpy filename for initial DOE dump
pub const DOE_INITIAL_FILE: &str = "egor_initial_doe.npy";
/// Numpy Filename for current DOE dump
pub const DOE_FILE: &str = "egor_doe.npy";

/// Default tolerance value for constraints to be satisfied (ie cstr < tol)
pub const DEFAULT_CSTR_TOL: f64 = 1e-6;

impl<O, SB> Solver<O, EgorState<f64>> for EgorSolver<SB>
where
    O: CostFunction<Param = Array2<f64>, Output = Array2<f64>>,
    SB: SurrogateBuilder,
{
    const NAME: &'static str = "Egor";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> std::result::Result<(EgorState<f64>, Option<KV>), argmin::core::Error> {
        let rng = self.rng.clone();
        let sampling = Lhs::new(&self.xlimits).with_rng(rng).kind(LhsKind::Maximin);

        let hstart_doe: Option<Array2<f64>> =
            if self.config.hot_start && self.config.outdir.is_some() {
                let path: &String = self.config.outdir.as_ref().unwrap();
                let filepath = std::path::Path::new(&path).join(DOE_FILE);
                if filepath.is_file() {
                    info!("Reading DOE from {:?}", filepath);
                    Some(read_npy(filepath)?)
                } else if std::path::Path::new(&path).join(DOE_INITIAL_FILE).is_file() {
                    let filepath = std::path::Path::new(&path).join(DOE_INITIAL_FILE);
                    info!("Reading DOE from {:?}", filepath);
                    Some(read_npy(filepath)?)
                } else {
                    None
                }
            } else {
                None
            };

        let doe = hstart_doe.as_ref().or(self.config.doe.as_ref());

        let (y_data, x_data) = if let Some(doe) = doe {
            if doe.ncols() == self.xlimits.nrows() {
                // only x are specified
                info!("Compute initial DOE on specified {} points", doe.nrows());
                (self.eval_obj(problem, doe), doe.to_owned())
            } else {
                // split doe in x and y
                info!("Use specified DOE {} samples", doe.nrows());
                (
                    doe.slice(s![.., self.xlimits.nrows()..]).to_owned(),
                    doe.slice(s![.., ..self.xlimits.nrows()]).to_owned(),
                )
            }
        } else {
            let n_doe = if self.config.n_doe == 0 {
                (self.xlimits.nrows() + 1).max(5)
            } else {
                self.config.n_doe
            };
            info!("Compute initial LHS with {} points", n_doe);
            let x = sampling.sample(n_doe);
            (self.eval_obj(problem, &x), x)
        };
        let doe = concatenate![Axis(1), x_data, y_data];
        if self.config.outdir.is_some() {
            let path = self.config.outdir.as_ref().unwrap();
            std::fs::create_dir_all(path)?;
            let filepath = std::path::Path::new(path).join(DOE_INITIAL_FILE);
            info!("Save initial doe shape {:?} in {:?}", doe.shape(), filepath);
            write_npy(filepath, &doe).expect("Write initial doe");
        }

        let clusterings = vec![None; self.config.n_cstr + 1];
        let theta_inits = vec![None; self.config.n_cstr + 1];
        let no_point_added_retries = MAX_POINT_ADDITION_RETRY;

        let mut initial_state = state
            .data((x_data, y_data.clone()))
            .clusterings(clusterings)
            .theta_inits(theta_inits)
            .sampling(sampling);
        initial_state.doe_size = doe.nrows();
        initial_state.max_iters = self.config.max_iters as u64;
        initial_state.added = doe.nrows();
        initial_state.no_point_added_retries = no_point_added_retries;
        initial_state.cstr_tol = self
            .config
            .cstr_tol
            .clone()
            .unwrap_or(Array1::from_elem(self.config.n_cstr, DEFAULT_CSTR_TOL));
        initial_state.target_cost = self.config.target;

        let best_index = find_best_result_index(&y_data, &initial_state.cstr_tol);
        initial_state.best_index = Some(best_index);
        initial_state.last_best_iter = 0;
        debug!("INITIAL STATE = {:?}", initial_state);
        Ok((initial_state, None))
    }

    fn next_iter(
        &mut self,
        fobj: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> std::result::Result<(EgorState<f64>, Option<KV>), argmin::core::Error> {
        debug!(
            "********* Start iteration {}/{}",
            state.get_iter() + 1,
            state.get_max_iters()
        );
        let now = Instant::now();
        // Retrieve Egor internal state
        let mut clusterings =
            state
                .clone()
                .take_clusterings()
                .ok_or_else(argmin_error_closure!(
                    PotentialBug,
                    "EgorSolver: No clustering!"
                ))?;
        let mut theta_inits =
            state
                .clone()
                .take_theta_inits()
                .ok_or_else(argmin_error_closure!(
                    PotentialBug,
                    "EgorSolver: No theta inits!"
                ))?;
        let sampling = state
            .clone()
            .take_sampling()
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "EgorSolver: No sampling!"
            ))?;
        let (mut x_data, mut y_data) = state
            .clone()
            .take_data()
            .ok_or_else(argmin_error_closure!(PotentialBug, "EgorSolver: No data!"))?;

        let mut new_state = state.clone();
        let (rejected_count, models) = loop {
            let recluster = self.have_to_recluster(new_state.added, new_state.prev_added);
            let init = new_state.get_iter() == 0;
            let lhs_optim_seed = if new_state.no_point_added_retries == 0 {
                debug!("Try Lhs optimization!");
                Some(new_state.added as u64)
            } else {
                debug!(
                    "Try point addition {}/{}",
                    MAX_POINT_ADDITION_RETRY - new_state.no_point_added_retries,
                    MAX_POINT_ADDITION_RETRY
                );
                None
            };

            let (x_dat, y_dat, infill_value, models) = self.next_points(
                init,
                state.get_iter(),
                recluster,
                &mut clusterings,
                &mut theta_inits,
                &x_data,
                &y_data,
                &state.cstr_tol,
                &sampling,
                lhs_optim_seed,
            );

            debug!("Try adding {}", x_dat);
            let added_indices = update_data(&mut x_data, &mut y_data, &x_dat, &y_dat);

            new_state = new_state
                .clusterings(clusterings.clone())
                .theta_inits(theta_inits.clone())
                .data((x_data.clone(), y_data.clone()))
                .infill_value(infill_value)
                .sampling(sampling.clone())
                .param(x_dat.row(0).to_owned())
                .cost(y_dat.row(0).to_owned());
            info!(
                "Infill criterion max value {} = {}",
                self.config.infill_criterion.name(),
                state.get_infill_value()
            );

            let rejected_count = x_dat.nrows() - added_indices.len();
            for i in 0..x_dat.nrows() {
                debug!(
                    "  {} {}",
                    if added_indices.contains(&i) { "A" } else { "R" },
                    x_dat.row(i)
                );
            }
            if rejected_count > 0 {
                info!(
                    "Reject {}/{} point{} too close to previous ones",
                    rejected_count,
                    x_dat.nrows(),
                    if rejected_count > 1 { "s" } else { "" }
                );
            }
            if rejected_count == x_dat.nrows() {
                new_state.no_point_added_retries -= 1;
                if new_state.no_point_added_retries == 0 {
                    info!("Max number of retries ({}) without adding point", 3);
                    info!("Use LHS optimization to ensure a point addition");
                }
                if new_state.no_point_added_retries < 0 {
                    // no luck with LHS optimization
                    warn!("Fail to add another point to improve the surrogate models. Abort!");
                    return Ok((
                        state.terminate_with(TerminationReason::SolverExit(
                            "Even LHS optimization failed to add a new point".to_string(),
                        )),
                        None,
                    ));
                }
            } else {
                // ok point added we can go on, just output number of rejected point
                break (rejected_count, models);
            }
        };

        let add_count = (self.config.q_points - rejected_count) as i32;
        let x_to_eval = x_data.slice(s![-add_count.., ..]).to_owned();
        debug!(
            "Eval {} point{} {}",
            add_count,
            if add_count > 1 { "s" } else { "" },
            if new_state.no_point_added_retries == 0 {
                " from sampling"
            } else {
                ""
            }
        );
        new_state.prev_added = new_state.added;
        new_state.added += add_count as usize;
        info!("+{} point(s), total: {} points", add_count, new_state.added);
        new_state.no_point_added_retries = MAX_POINT_ADDITION_RETRY; // reset as a point is added

        let y_actual = self.eval_obj(fobj, &x_to_eval);
        Zip::from(y_data.slice_mut(s![-add_count.., ..]).rows_mut())
            .and(y_actual.rows())
            .for_each(|mut y, val| y.assign(&val));
        let doe = concatenate![Axis(1), x_data, y_data];
        if self.config.outdir.is_some() {
            let path = self.config.outdir.as_ref().unwrap();
            std::fs::create_dir_all(path)?;
            let filepath = std::path::Path::new(path).join(DOE_FILE);
            info!("Save doe shape {:?} in {:?}", doe.shape(), filepath);
            write_npy(filepath, &doe).expect("Write current doe");
        }

        let best_index = find_best_result_index_from(
            state.best_index.unwrap(),
            y_data.nrows() - add_count as usize,
            &y_data,
            &new_state.cstr_tol,
        );

        let best_index = self.trego_step(
            fobj,
            models,
            sampling,
            best_index,
            &mut x_data,
            &mut y_data,
            &state,
            &mut new_state,
        );

        // let best = find_best_result_index(&y_data, &new_state.cstr_tol);
        // assert!(best_index == best);
        new_state.best_index = Some(best_index);
        info!(
            "********* End iteration {}/{} in {:.3}s: Best fun(x)={} at x={}",
            state.get_iter() + 1,
            state.get_max_iters(),
            now.elapsed().as_secs_f64(),
            y_data.row(best_index),
            x_data.row(best_index)
        );
        new_state = new_state.data((x_data, y_data.clone()));

        Ok((new_state, None))
    }

    fn terminate(&mut self, state: &EgorState<f64>) -> TerminationStatus {
        debug!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> end iteration");
        debug!("Current Cost {:?}", state.get_cost());
        debug!("Best cost {:?}", state.get_best_cost());
        debug!("Best index {:?}", state.best_index);
        debug!("Data {:?}", state.data.as_ref().unwrap());

        TerminationStatus::NotTerminated
    }
}
