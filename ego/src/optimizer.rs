use crate::lhs_optimizer::LhsOptimizer;
use crate::ObjData;
use ndarray::{arr1, Array1, Array2, ArrayView1};

#[cfg(not(feature = "nlopt"))]
use crate::types::ObjFn;
#[cfg(not(feature = "nlopt"))]
use cobyla::RhoBeg;

#[cfg(feature = "nlopt")]
use nlopt::ObjFn;

use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

#[derive(Copy, Clone, Debug)]
pub enum Algorithm {
    Cobyla,
    Slsqp,
    Lhs,
}

/// Facade for various optimization algorithms
pub(crate) struct Optimizer<'a> {
    algo: Algorithm,
    fun: &'a (dyn ObjFn<ObjData<f64>> + Sync),
    cons: Vec<&'a (dyn ObjFn<ObjData<f64>> + Sync)>,
    bounds: Array2<f64>,
    user_data: &'a ObjData<f64>,
    max_eval: usize,
    xinit: Option<Array1<f64>>,
    ftol_abs: Option<f64>,
    ftol_rel: Option<f64>,
    seed: Option<u64>,
}

impl<'a> Optimizer<'a> {
    pub fn new(
        algo: Algorithm,
        fun: &'a (dyn ObjFn<ObjData<f64>> + Sync),
        cons: Vec<&'a (dyn ObjFn<ObjData<f64>> + Sync)>,
        user_data: &'a ObjData<f64>,
        bounds: &Array2<f64>,
    ) -> Self {
        Optimizer {
            algo,
            fun,
            cons,
            bounds: bounds.clone(),
            user_data,
            max_eval: 200,
            xinit: None,
            ftol_abs: None,
            ftol_rel: None,
            seed: None,
        }
    }

    pub fn ftol_abs(&mut self, ftol_abs: f64) -> &mut Self {
        self.ftol_abs = Some(ftol_abs);
        self
    }

    pub fn ftol_rel(&mut self, ftol_rel: f64) -> &mut Self {
        self.ftol_rel = Some(ftol_rel);
        self
    }

    pub fn max_eval(&mut self, max_eval: usize) -> &mut Self {
        self.max_eval = max_eval;
        self
    }

    pub fn seed(&mut self, seed: u64) -> &mut Self {
        self.seed = Some(seed);
        self
    }

    pub fn xinit(&mut self, xinit: &ArrayView1<f64>) -> &mut Self {
        self.xinit = Some(xinit.to_owned());
        self
    }

    pub fn minimize(&self) -> (f64, Array1<f64>) {
        match self.algo {
            Algorithm::Cobyla => {
                let xinit = self.xinit.clone().unwrap().to_vec();
                let bounds: Vec<_> = self
                    .bounds
                    .outer_iter()
                    .map(|row| (row[0], row[1]))
                    .collect();
                let cstrs: Vec<_> = self
                    .cons
                    .iter()
                    .map(|f| |x: &[f64], u: &mut ObjData<f64>| (*f)(x, None, u))
                    .collect();
                let res = cobyla::minimize(
                    |x: &[f64], u: &mut ObjData<f64>| (self.fun)(x, None, u),
                    &xinit,
                    &bounds,
                    &cstrs,
                    self.user_data.clone(),
                    self.max_eval,
                    RhoBeg::All(0.5),
                    Some(cobyla::StopTols {
                        ftol_rel: self.ftol_rel.unwrap_or(0.0),
                        ftol_abs: self.ftol_abs.unwrap_or(0.0),
                        ..cobyla::StopTols::default()
                    }),
                );
                match res {
                    Ok((_, x_opt, y_opt)) => (y_opt, arr1(&x_opt)),
                    Err((_, x_opt, _)) => (f64::INFINITY, arr1(&x_opt)),
                }
            }
            Algorithm::Slsqp => {
                let xinit = self.xinit.clone().unwrap().to_vec();
                let bounds: Vec<_> = self
                    .bounds
                    .outer_iter()
                    .map(|row| (row[0], row[1]))
                    .collect();
                let res = slsqp::minimize(
                    self.fun,
                    &xinit,
                    &bounds,
                    &self.cons,
                    self.user_data.clone(),
                    self.max_eval,
                    Some(slsqp::StopTols {
                        ftol_rel: self.ftol_rel.unwrap_or(0.0),
                        ftol_abs: self.ftol_abs.unwrap_or(0.0),
                        ..slsqp::StopTols::default()
                    }),
                );
                match res {
                    Ok((_, x_opt, y_opt)) => (y_opt, arr1(&x_opt)),
                    Err((_, x_opt, _)) => (f64::INFINITY, arr1(&x_opt)),
                }
            }
            Algorithm::Lhs => {
                let res = LhsOptimizer::new(&self.bounds, self.fun, &self.cons, self.user_data);
                let res = if let Some(seed) = self.seed {
                    res.with_rng(Xoshiro256Plus::seed_from_u64(seed))
                } else {
                    res.with_rng(Xoshiro256Plus::from_entropy())
                };
                res.minimize()
            }
        }
    }
}
