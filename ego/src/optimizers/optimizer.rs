use crate::optimizers::LhsOptimizer;
use crate::InfillObjData;
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
    fun: &'a (dyn ObjFn<InfillObjData<f64>> + Sync),
    cons: Vec<&'a (dyn ObjFn<InfillObjData<f64>> + Sync)>,
    cstr_tol: Option<Array1<f64>>,
    bounds: Array2<f64>,
    user_data: &'a InfillObjData<f64>,
    max_eval: usize,
    xinit: Option<Array1<f64>>,
    ftol_abs: Option<f64>,
    ftol_rel: Option<f64>,
    seed: Option<u64>,
}

impl<'a> Optimizer<'a> {
    pub fn new(
        algo: Algorithm,
        fun: &'a (dyn ObjFn<InfillObjData<f64>> + Sync),
        cons: &[&'a (dyn ObjFn<InfillObjData<f64>> + Sync)],
        user_data: &'a InfillObjData<f64>,
        bounds: &Array2<f64>,
    ) -> Self {
        Optimizer {
            algo,
            fun,
            cons: cons.to_vec(),
            cstr_tol: None,
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

    pub fn cstr_tol(&mut self, cstr_tol: Array1<f64>) -> &mut Self {
        self.cstr_tol = Some(cstr_tol);
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

    #[cfg(feature = "nlopt")]
    fn nlopt_minimize(&self, algo: nlopt::Algorithm, cstr_tol: Array1<f64>) -> (f64, Array1<f64>) {
        use nlopt::*;
        let mut optimizer = Nlopt::new(
            algo,
            self.bounds.nrows(),
            self.fun,
            Target::Minimize,
            self.user_data.clone(),
        );
        let lower = self.bounds.column(0).to_owned();
        optimizer
            .set_lower_bounds(lower.as_slice().unwrap())
            .unwrap();
        let upper = self.bounds.column(1).to_owned();
        optimizer
            .set_upper_bounds(upper.as_slice().unwrap())
            .unwrap();
        optimizer.set_maxeval(self.max_eval as u32).unwrap();
        optimizer
            .set_ftol_rel(self.ftol_rel.unwrap_or(0.0))
            .unwrap();
        optimizer
            .set_ftol_abs(self.ftol_abs.unwrap_or(0.0))
            .unwrap();
        self.cons.iter().enumerate().for_each(|(i, cstr)| {
            optimizer
                .add_inequality_constraint(
                    cstr,
                    self.user_data.clone(),
                    cstr_tol[i] / self.user_data.scale_cstr[i],
                )
                .unwrap();
        });

        let mut x_opt = self.xinit.clone().unwrap().to_vec();
        match optimizer.optimize(&mut x_opt) {
            Ok((_, opt)) => (opt, arr1(&x_opt)),
            Err((_err, _code)) => {
                // debug!("Nlopt Err: {:?} (y_opt={})", err, code);
                (f64::INFINITY, arr1(&x_opt))
            }
        }
    }

    pub fn minimize(&self) -> (f64, Array1<f64>) {
        let cstr_tol = self
            .cstr_tol
            .clone()
            .unwrap_or(Array1::zeros(self.cons.len()));
        match self.algo {
            Algorithm::Cobyla => {
                #[cfg(feature = "nlopt")]
                {
                    self.nlopt_minimize(nlopt::Algorithm::Cobyla, cstr_tol)
                }

                #[cfg(not(feature = "nlopt"))]
                {
                    let xinit = self.xinit.clone().unwrap().to_vec();
                    let bounds: Vec<_> = self
                        .bounds
                        .outer_iter()
                        .map(|row| (row[0], row[1]))
                        .collect();
                    let cstrs: Vec<_> = self
                        .cons
                        .iter()
                        .enumerate()
                        .map(|(i, f)| {
                            let cstr_tol = cstr_tol[i];
                            move |x: &[f64], u: &mut InfillObjData<f64>| {
                                -(*f)(x, None, u) - cstr_tol / u.scale_cstr[i]
                            }
                        })
                        .collect();
                    let res = cobyla::minimize(
                        |x: &[f64], u: &mut InfillObjData<f64>| (self.fun)(x, None, u),
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
            }
            Algorithm::Slsqp => {
                #[cfg(feature = "nlopt")]
                {
                    self.nlopt_minimize(nlopt::Algorithm::Slsqp, cstr_tol)
                }
                #[cfg(not(feature = "nlopt"))]
                {
                    let xinit = self.xinit.clone().unwrap().to_vec();
                    let bounds: Vec<_> = self
                        .bounds
                        .outer_iter()
                        .map(|row| (row[0], row[1]))
                        .collect();
                    let cstrs: Vec<_> = self
                        .cons
                        .iter()
                        .enumerate()
                        .map(|(i, f)| {
                            let cstr_tol = cstr_tol[i];
                            move |x: &[f64], g: Option<&mut [f64]>, u: &mut InfillObjData<f64>| {
                                (*f)(x, g, u) - cstr_tol / u.scale_cstr[i]
                            }
                        })
                        .collect();
                    let res = slsqp::minimize(
                        self.fun,
                        &xinit,
                        &bounds,
                        &cstrs,
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
