use crate::utils::{cdist, pdist};
use ndarray::{array, s, Array, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_rand::{
    rand::seq::SliceRandom, rand::thread_rng, rand::Rng, rand_distr::Uniform, RandomExt,
};
use ndarray_stats::QuantileExt;
use std::cmp;

struct LHS {
    xlimits: Array2<usize>,
}

impl LHS {
    pub fn new(xlimits: &ArrayBase<impl Data<Elem = usize>, Ix2>) -> Self {
        if xlimits.ncols() != 2 {
            panic!("xlimits must have 2 columns (lower, upper)");
        }
        LHS {
            xlimits: xlimits.to_owned(),
        }
    }

    pub fn build(self, ns: usize) -> Array2<f64> {
        let lhs0 = self.normalized_classic_lhs(ns);
        let nx = self.xlimits.ncols();

        let outer_loop = cmp::min((1.5 * nx as f64) as usize, 30);
        let inner_loop = cmp::min(20 * nx, 100);

        self.maximin_ese(&lhs0, outer_loop, inner_loop)
    }

    fn maximin_ese(&self, lhs: &Array2<f64>, outer_loop: usize, inner_loop: usize) -> Array2<f64> {
        // hard-coded params
        let j_range = 20;
        let p = 10.;
        let t0 = 0.005 * self._phip(&lhs, p);
        let tol = 1e-3;

        let mut t = t0;
        let mut lhs_own = lhs.to_owned();
        let mut lhs_best = Array2::zeros((lhs_own.nrows(), lhs_own.ncols()));
        lhs_best.assign(&lhs_own);
        let nx = lhs.ncols();
        let mut phip = self._phip(&lhs_best, p);
        let mut phip_best = phip;

        for _ in 0..outer_loop {
            let phip_old_best = phip_best;
            let mut n_acpt = 0.;
            let mut n_imp = 0.;

            for i in 0..inner_loop {
                let modulo = (i + 1) % nx;
                let mut l_x: Vec<Box<Array2<f64>>> = Vec::new();
                let mut l_phip: Vec<f64> = Vec::new();

                // Build j different plans with a single swap procedure
                // See description of phip_swap procedure
                for j in 0..j_range {
                    l_x.push(Box::new(lhs_own.to_owned()));
                    let php = self._phip_swap(&mut l_x[j], modulo, phip, p);
                    l_phip.push(php);
                }

                let lphip = Array::from_shape_vec(l_phip.len(), l_phip).unwrap();
                let k = lphip.argmin().unwrap();
                let mut phip_try = lphip[k];
                // Threshold of acceptance
                let mut rng = thread_rng();
                if phip_try - phip <= t * rng.gen::<f64>() {
                    phip = phip_try;
                    n_acpt = n_acpt + 1.;
                    lhs_own.assign(&(*l_x[k]));

                    // best plan retained
                    if phip < phip_best {
                        lhs_best.assign(&lhs_own);
                        phip_best = phip;
                        n_imp = n_imp + 1.;
                    }
                }
            }

            let p_accpt = n_acpt / inner_loop as f64; // probability of acceptance
            let p_imp = n_imp / inner_loop as f64; // probability of improvement

            if phip_best - phip < tol {
                if p_accpt >= 0.1 && p_imp < p_accpt {
                    t = 0.8 * t
                } else if p_accpt >= 0.1 && p_imp == p_accpt {
                } else {
                    t = t / 0.8
                }
            } else {
                if p_accpt <= 0.1 {
                    t = t / 0.7
                } else {
                    t = 0.9 * t
                }
            }
        }

        lhs_best
    }

    fn _phip(&self, lhs: &ArrayBase<impl Data<Elem = f64>, Ix2>, p: f64) -> f64 {
        let pd = pdist(lhs);
        f64::powf(pdist(lhs).mapv(|v| f64::powf(v, -p)).sum(), 1. / p)
    }

    pub fn _phip_swap(&self, x: &mut Array2<f64>, k: usize, phip: f64, p: f64) -> f64 {
        // Choose two random rows
        let mut rng = thread_rng();
        let mut i1 = rng.gen_range(0, x.nrows());
        let mut i2 = rng.gen_range(0, x.nrows());
        while i2 == i1 {
            i2 = rng.gen_range(0, x.nrows());
        }
        // Compute new phip
        let mut x_rest = Array2::zeros((x.nrows() - 2, x.ncols()));
        let mut row_i = 0;
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            if i != i1 && i != i2 {
                x_rest.slice_mut(s![row_i, ..]).assign(&row);
                row_i += 1;
            }
        }

        let dist1 = cdist(&x.slice(s![i1..i1 + 1, ..]), &x_rest);
        let dist2 = cdist(&x.slice(s![i2..i2 + 1, ..]), &x_rest);

        let m1 = (x_rest.slice(s![.., k]).to_owned() - x.slice(s![i1..i1 + 1, k])).map(|v| v * v);
        let m2 = (x_rest.slice(s![.., k]).to_owned() - x.slice(s![i2..i2 + 1, k])).map(|v| v * v);

        let mut d1 = dist1.mapv(|v| v * v) - &m1 + &m2;
        d1 = d1.mapv(|v| f64::sqrt(v));
        let mut d2 = dist2.mapv(|v| v * v) + &m1 - &m2;
        d2 = d2.mapv(|v| f64::sqrt(v));

        let mut res = (d1.mapv(|v| f64::powf(v, -p)) - dist1.mapv(|v| f64::powf(v, -p))).sum();
        res += (d2.mapv(|v| f64::powf(v, -p)) - dist2.mapv(|v| f64::powf(v, -p))).sum();
        let ope = f64::powf(phip, p) + res;
        res = f64::powf(f64::powf(phip, p) + res, 1. / p);

        // swap points
        x.swap([i1, k], [i2, k]);

        res
    }

    pub fn normalized_classic_lhs(&self, ns: usize) -> Array2<f64> {
        let nx = self.xlimits.ncols();
        let cut = Array::linspace(0., 1., ns + 1);

        let u = Array::random((ns, nx), Uniform::new(0., 1.));
        let a = cut.slice(s![..ns]).to_owned();
        let b = cut.slice(s![1..(ns + 1)]);
        let mut rdpoints = Array::zeros((ns, nx));
        for j in 0..nx {
            let c = &b - &a;
            let d = u.slice(s![.., j]).to_owned() * &c;
            let val = &d + &a;
            rdpoints.slice_mut(s![.., j]).assign(&val)
        }
        let mut lhs = Array::zeros((ns, nx));
        let mut rng = thread_rng();
        for j in 0..nx {
            rdpoints.as_slice_mut().unwrap().shuffle(&mut rng);
            let colj = rdpoints.slice(s![.., j]);
            lhs.slice_mut(s![.., j]).assign(&colj);
        }
        lhs
    }

    pub fn normalized_centered_lhs(&self, ns: usize) -> Array2<f64> {
        let nx = self.xlimits.ncols();
        let cut = Array::linspace(0., 1., ns + 1);

        let u = Array::random((ns, nx), Uniform::new(0., 1.));
        let a = cut.slice(s![..ns]).to_owned();
        let b = cut.slice(s![1..(ns + 1)]);
        let mut c = (a + b) / 2.;
        let mut lhs = Array::zeros(u.raw_dim());
        let mut rng = thread_rng();
        for j in 0..nx {
            c.as_slice_mut().unwrap().shuffle(&mut rng);
            lhs.slice_mut(s![.., j]).assign(&c);
        }
        lhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr2, array};

    #[test]
    fn test_lhs() {
        let xlimits = arr2(&[[5, 10], [0, 1]]);
        let lhs = LHS::new(&xlimits);
        let doe = lhs.build(10);
    }

    #[test]
    fn test_classic_lhs() {
        let xlimits = arr2(&[[5, 10], [0, 1]]);
        let lhs = LHS::new(&xlimits);
        let lhs0 = lhs.normalized_classic_lhs(10);
    }

    #[test]
    fn test_phip_swap() {
        let xlimits = arr2(&[[5, 10], [0, 1]]);
        let lhs = LHS::new(&xlimits);
        let k = 1;
        let phip = 7.290525742903316;
        let mut p0 = array![
            [0.45, 0.75],
            [0.75, 0.95],
            [0.05, 0.45],
            [0.55, 0.15000000000000002],
            [0.35000000000000003, 0.25],
            [0.95, 0.8500000000000001],
            [0.15000000000000002, 0.55],
            [0.25, 0.05],
            [0.8500000000000001, 0.35000000000000003],
            [0.6500000000000001, 0.6500000000000001]
        ];
        let p = 10.;
        let res = lhs._phip_swap(&mut p0, k, phip, p);
    }
}
