use crate::types::ObjData;
use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use ndarray::{Array1, Array2, Axis, Zip};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;

#[cfg(not(feature = "blas"))]
use linfa_linalg::norm::*;
#[cfg(feature = "blas")]
use ndarray_linalg::Norm;

use ndarray_stats::QuantileExt;
use nlopt::ObjFn;

pub(crate) struct LhsOptimizer<'a, R: Rng + Clone> {
    xlimits: Array2<f64>,
    n_start: usize,
    n_points: usize,
    cstr_tol: f64,
    obj: &'a dyn ObjFn<ObjData<f64>>,
    cstrs: Vec<&'a dyn ObjFn<ObjData<f64>>>,
    obj_data: ObjData<f64>,
    rng: R,
}

impl<'a> LhsOptimizer<'a, Xoshiro256Plus> {
    pub fn new(
        xlimits: &Array2<f64>,
        obj: &'a dyn ObjFn<ObjData<f64>>,
        cstrs: Vec<&'a dyn ObjFn<ObjData<f64>>>,
        obj_data: &ObjData<f64>,
    ) -> LhsOptimizer<'a, Xoshiro256Plus> {
        Self::new_with_rng(
            xlimits,
            obj,
            cstrs,
            obj_data,
            Xoshiro256Plus::from_entropy(),
        )
    }
}

impl<'a, R: Rng + Clone> LhsOptimizer<'a, R> {
    pub fn new_with_rng(
        xlimits: &Array2<f64>,
        obj: &'a dyn ObjFn<ObjData<f64>>,
        cstrs: Vec<&'a dyn ObjFn<ObjData<f64>>>,
        obj_data: &ObjData<f64>,
        rng: R,
    ) -> LhsOptimizer<'a, R> {
        LhsOptimizer {
            xlimits: xlimits.to_owned(),
            n_start: 20,
            n_points: 100,
            cstr_tol: 1e-6,
            obj,
            cstrs,
            obj_data: obj_data.clone(),
            rng,
        }
    }

    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> LhsOptimizer<'a, R2> {
        LhsOptimizer {
            xlimits: self.xlimits,
            n_start: self.n_start,
            n_points: self.n_points,
            cstr_tol: self.cstr_tol,
            obj: self.obj,
            cstrs: self.cstrs,
            obj_data: self.obj_data,
            rng,
        }
    }

    pub fn minimize(&self) -> Array1<f64> {
        let mut x_optim = vec![];

        // Make n_start optim
        for _ in 0..self.n_start {
            x_optim.push(self.find_lhs_min());
        }

        // Pick best
        if x_optim.iter().any(|opt| opt.0) {
            let values: Array1<_> = x_optim
                .iter()
                .filter(|opt| opt.0)
                .map(|opt| (opt.1.to_owned(), opt.2))
                .collect();
            let yvals: Array1<_> = values.iter().map(|val| val.1).collect();
            let index_min = yvals.argmin().unwrap();
            values[index_min].0.to_owned()
        } else {
            let l1_norms: Array1<_> = x_optim.iter().map(|opt| opt.3.norm_l1()).collect();
            let index_min = l1_norms.argmin().unwrap();
            x_optim[index_min].1.to_owned()
        }
    }

    fn find_lhs_min(&self) -> (bool, Array1<f64>, f64, Array1<f64>) {
        let n = self.n_points * self.xlimits.nrows();
        let doe = Lhs::new(&self.xlimits)
            .kind(LhsKind::Classic)
            .with_rng(self.rng.clone())
            .sample(n);

        let y: Array1<f64> = doe.map_axis(Axis(1), |x| {
            (self.obj)(&x.to_vec(), None, &mut self.obj_data.clone())
        });

        let n_cstr = self.cstrs.len();

        let mut cstrs_values = Array2::zeros((n, n_cstr));
        Zip::from(cstrs_values.rows_mut())
            .and(doe.rows())
            .for_each(|mut cstr_values, x| {
                for i in 0..self.cstrs.len() {
                    let cstr_val = (self.cstrs[i])(&x.to_vec(), None, &mut self.obj_data.clone());
                    cstr_values[i] = cstr_val;
                }
            });

        let valid = cstrs_values.map_axis(Axis(1), |cstrs_i| {
            cstrs_i.fold(true, |acc, &a| acc && a < self.cstr_tol)
        });

        if valid.iter().any(|b| *b) {
            let vals: Vec<_> = valid
                .iter()
                .enumerate()
                .filter_map(|(i, &b)| {
                    if b && !y[i].is_nan() {
                        Some((doe.row(i).to_owned(), y[i], cstrs_values.row(i)))
                    } else {
                        None
                    }
                })
                .collect();
            let values = Array1::from_vec(vals.iter().map(|(_, y, _)| *y).collect());
            let index_min = values
                .argmin()
                .unwrap_or_else(|err| panic!("Cannot find min in {}: {:?}", values, err));
            (
                true,
                vals[index_min].0.to_owned(),
                vals[index_min].1,
                vals[index_min].2.to_owned(),
            )
        } else {
            let l1_norms = cstrs_values.map_axis(Axis(1), |cstrs_i| cstrs_i.norm_l1());
            let index_min = l1_norms.argmin().unwrap();
            (
                false,
                doe.row(index_min).to_owned(),
                y[index_min],
                cstrs_values.row(index_min).to_owned(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_min_obj_only() {
        let obj = |x: &[f64], _grad: Option<&mut [f64]>, _params: &mut ObjData<f64>| -> f64 {
            x[0] * x[0]
        };
        let cstrs = vec![];

        let xlimits = array![[-1., 1.]];
        let obj_data = ObjData {
            scale_obj: 1.,
            scale_cstr: array![],
            scale_wb2: 1.,
        };

        let res = LhsOptimizer::new(&xlimits, &obj, cstrs, &obj_data)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .minimize();
        assert_abs_diff_eq!(res, array![0.], epsilon = 1e-1)
    }
}
