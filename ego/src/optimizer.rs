use crate::lhs_optimizer::LhsOptimizer;
use crate::types::{Algorithm, ObjFn, Optimizer};
use crate::ObjData;
use ndarray::{Array1, Array2};

use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

pub(crate) struct OptimizerBuilder<'a> {
    algo: Algorithm,
    fun: &'a (dyn ObjFn<ObjData<f64>> + Sync),
    cons: Vec<&'a (dyn ObjFn<ObjData<f64>> + Sync)>,
    bounds: Array2<f64>,
    user_data: &'a ObjData<f64>,
    max_eval: usize,
    xinit: Option<Array1<f64>>,
    ftol_abs: Option<f64>,
    ftol_rel: Option<f64>,
    xtol_abs: Option<f64>,
    xtol_rel: Option<f64>,
    seed: Option<u64>,
}

impl<'a> OptimizerBuilder<'a> {
    pub fn new(
        algo: Algorithm,
        fun: &'a (dyn ObjFn<ObjData<f64>> + Sync),
        cons: Vec<&'a (dyn ObjFn<ObjData<f64>> + Sync)>,
        user_data: &'a ObjData<f64>,
        bounds: &Array2<f64>,
    ) -> Self {
        OptimizerBuilder {
            algo,
            fun,
            cons,
            bounds: bounds.clone(),
            user_data,
            max_eval: 200,
            xinit: None,
            ftol_abs: None,
            ftol_rel: None,
            xtol_abs: None,
            xtol_rel: None,
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

    pub fn xtol_abs(&mut self, xtol_abs: f64) -> &mut Self {
        self.xtol_abs = Some(xtol_abs);
        self
    }

    pub fn xtol_rel(&mut self, xtol_rel: f64) -> &mut Self {
        self.xtol_rel = Some(xtol_rel);
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

    pub fn xinit(&mut self, xinit: Array1<f64>) -> &mut Self {
        self.xinit = Some(xinit);
        self
    }

    pub fn minimize(&self) -> Result<(Array1<f64>, f64), ()> {
        match self.algo {
            Algorithm::Cobyla => todo!(),
            Algorithm::Slsqp => todo!(),
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
