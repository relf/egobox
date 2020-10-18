use crate::utils::{cdist, pdist};
use ndarray::{array, s, Array, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_rand::{
    rand::seq::SliceRandom, rand::thread_rng, rand::Rng, rand_distr::Uniform, RandomExt,
};
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
        let lhs0 = self.normalized_centered_lhs(ns);
        let nx = self.xlimits.ncols();

        let outer_loop = cmp::min((1.5 * nx as f64) as usize, 30);
        let inner_loop = cmp::min(20 * nx, 100);

        self.maximin_ese(&lhs0, outer_loop, inner_loop)
    }

    fn maximin_ese(
        &self,
        lhs: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        outer_loop: usize,
        inner_loop: usize,
    ) -> Array2<f64> {
        let j_range = 20;
        let p = 10.;
        let t0 = 0.005 * self._phip(lhs, p);

        let t = t0;
        let lhs_ = lhs.clone();
        let lhs_best = lhs_;
        let nx = lhs.ncols();
        let phip = self._phip(lhs_best, p);
        let phip_best = phip;

        for z in 0..outer_loop {
            let phip_old_best = phip_best;
            let n_acpt = 0;
            let n_imp = 0;

            for i in 0..inner_loop {
                let modulo = (i + 1) % nx;
                let mut l_x: Vec<Array2<f64>> = Vec::new();
                let mut l_phip: Vec<f64> = Vec::new();

                // Build J different plans with a single exchange procedure
                // See description of PhiP_exchange procedure
                for j in 0..j_range {
                    l_x.push(lhs_.to_owned());
                    l_phip.push(self._phip_exchange(&l_x[j], modulo, phip, p));
                }

                let lphip = Array::from_shape_vec((1, l_phip.len()), l_phip);
                // k = np.argmin(lphip
                // PhiP_try = l_PhiP[k]

                // # Threshold of acceptance
                // if PhiP_try - PhiP_ <= T * np.random.rand(1)[0]:
                //     PhiP_ = PhiP_try
                //     n_acpt = n_acpt + 1
                //     X_ = l_X[k]

                //     # Best plan retained
                //     if PhiP_ < PhiP_best:
                //         X_best = X_
                //         PhiP_best = PhiP_
                //         n_imp = n_imp + 1

                // hist_PhiP.append(PhiP_best)
            }

            //     p_accpt = float(n_acpt) / inner_loop  # probability of acceptance
            //     p_imp = float(n_imp) / inner_loop  # probability of improvement

            //     hist_T.extend(inner_loop * [T])
            //     hist_proba.extend(inner_loop * [p_accpt])

            //     if PhiP_best - PhiP_oldbest < tol:
            //         # flag_imp = 1
            //         if p_accpt >= 0.1 and p_imp < p_accpt:
            //             T = 0.8 * T
            //         elif p_accpt >= 0.1 and p_imp == p_accpt:
            //             pass
            //         else:
            //             T = T / 0.8
            //     else:
            //         # flag_imp = 0
            //         if p_accpt <= 0.1:
            //             T = T / 0.7
            //         else:
            //             T = 0.9 * T

            // hist = {"PhiP": hist_PhiP, "T": hist_T, "proba": hist_proba}

            // if return_hist:
            //     return X_best, hist
            // else:
            //     return X_best
        }

        array![[0.]]
    }

    fn _phip(&self, lhs: &ArrayBase<impl Data<Elem = f64>, Ix2>, p: f64) -> f64 {
        f64::powf(pdist(lhs).mapv(|v| f64::powf(v, -p)).sum(), 1. / p)
    }

    pub fn _phip_exchange(
        &self,
        mut x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        k: usize,
        phip: f64,
        p: f64,
    ) -> f64 {
        let mut rng = thread_rng();
        let mut i1 = rng.gen_range(0, x.nrows());
        let mut i2 = rng.gen_range(0, x.nrows());
        while i2 == i1 {
            i2 = rng.gen_range(0, x.nrows());
        }

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
        res = f64::powf(f64::powf(phip, p) + res, 1. / p);

        res
    }

    fn normalized_centered_lhs(&self, ns: usize) -> Array2<f64> {
        let nx = self.xlimits.ncols();
        let cut = Array::linspace(0., 1., ns + 1);

        let u = Array::random((ns, nx), Uniform::new(0., 1.));
        let a = cut.slice(s![..ns]).to_owned();
        let b = cut.slice(s![1..(ns + 1)]);
        let mut c = (a + b) / 2.;
        let mut lhs = Array::zeros(u.raw_dim());
        let mut rng = thread_rng();
        for j in 0..nx {
            let cs = c.as_slice_mut().unwrap();
            cs.shuffle(&mut rng);
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
    fn test_phip_exchange() {
        let xlimits = arr2(&[[5, 10], [0, 1]]);
        let lhs = LHS::new(&xlimits);
        let k = 1;
        let phip = 7.290525742903316;
        let p0 = array![
            [0.09674143, 0.84426416],
            [0.4810236, 0.15567728],
            [0.90804981, 0.74706188],
            [0.24951498, 0.97382578],
            [0.54482131, 0.30729966]
        ];
        let p = 10.;
        let res = lhs._phip_exchange(&p0, k, phip, p);
        println!("res={}", res);
    }
}
