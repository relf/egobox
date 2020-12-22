use crate::utils::{cdist, pdist};
use ndarray::{s, Array, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_rand::{
    rand::seq::SliceRandom, rand::Rng, rand::SeedableRng, rand_distr::Uniform, RandomExt,
};
use ndarray_stats::QuantileExt;
use rand_isaac::Isaac64Rng;
use std::cmp;

pub enum LHSKind {
    Classic,
    Centered,
    Optimized,
}

pub struct LHS<R: Rng + Clone> {
    xlimits: Array2<f64>,
    kind: LHSKind,
    rng: R,
}

impl LHS<Isaac64Rng> {
    pub fn new(xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Self {
        Self::new_with_rng(xlimits, Isaac64Rng::seed_from_u64(42))
    }
}

impl<R: Rng + Clone> LHS<R> {
    pub fn new_with_rng(xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>, rng: R) -> Self {
        if xlimits.ncols() != 2 {
            panic!("xlimits must have 2 columns (lower, upper)");
        }
        LHS {
            xlimits: xlimits.to_owned(),
            kind: LHSKind::Optimized,
            rng,
        }
    }

    pub fn kind(mut self, kind: LHSKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> LHS<R2> {
        LHS {
            xlimits: self.xlimits,
            kind: self.kind,
            rng,
        }
    }

    pub fn sample(&self, ns: usize) -> Array2<f64> {
        let mut rng = self.rng.clone();
        let mut doe = match &self.kind {
            LHSKind::Classic => self._normalized_classic_lhs(ns, &mut rng),
            LHSKind::Centered => self._normalized_centered_lhs(ns, &mut rng),
            LHSKind::Optimized => {
                let doe = self._normalized_classic_lhs(ns, &mut rng);
                let nx = self.xlimits.nrows();
                let outer_loop = cmp::min((1.5 * nx as f64) as usize, 30);
                let inner_loop = cmp::min(20 * nx, 100);
                self._maximin_ese(&doe, outer_loop, inner_loop, &mut rng)
            }
        };
        let a = self.xlimits.column(1);
        let d = &self.xlimits.column(0).to_owned() - &a;
        doe = doe * d + a;
        doe
    }

    fn _maximin_ese(
        &self,
        lhs: &Array2<f64>,
        outer_loop: usize,
        inner_loop: usize,
        rng: &mut R,
    ) -> Array2<f64> {
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
                    let php = self._phip_swap(&mut l_x[j], modulo, phip, p, rng);
                    l_phip.push(php);
                }
                let lphip = Array::from_shape_vec(l_phip.len(), l_phip).unwrap();
                let k = lphip.argmin().unwrap();
                let phip_try = lphip[k];
                // Threshold of acceptance
                if phip_try - phip <= t * rng.gen::<f64>() {
                    phip = phip_try;
                    n_acpt += 1.;
                    //lhs_own.assign(&(*l_x[k]));
                    lhs_own = *l_x[k].to_owned();

                    // best plan retained
                    if phip < phip_best {
                        // lhs_best.assign(&lhs_own);
                        lhs_best = lhs_own.to_owned();
                        phip_best = phip;
                        n_imp += 1.;
                    }
                }
            }
            let p_accpt = n_acpt / (inner_loop as f64); // probability of acceptance
            let p_imp = n_imp / (inner_loop as f64); // probability of improvement

            if phip_best - phip < tol {
                if p_accpt >= 0.1 && p_imp < p_accpt {
                    t *= 0.8
                } else if p_accpt >= 0.1 && (p_imp - p_accpt).abs() < f64::EPSILON {
                } else {
                    t /= 0.8
                }
            } else if p_accpt <= 0.1 {
                t /= 0.7
            } else {
                t *= 0.9
            }
        }
        lhs_best
    }

    fn _phip(&self, lhs: &ArrayBase<impl Data<Elem = f64>, Ix2>, p: f64) -> f64 {
        f64::powf(pdist(lhs).mapv(|v| f64::powf(v, -p)).sum(), 1. / p)
    }

    fn _phip_swap(&self, x: &mut Array2<f64>, k: usize, phip: f64, p: f64, rng: &mut R) -> f64 {
        // Choose two random rows
        //let mut rng = thread_rng();
        let i1 = rng.gen_range(0, x.nrows());
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
        d1 = d1.mapv(f64::sqrt);
        let mut d2 = dist2.mapv(|v| v * v) + &m1 - &m2;
        d2 = d2.mapv(f64::sqrt);

        let mut res = (d1.mapv(|v| f64::powf(v, -p)) - dist1.mapv(|v| f64::powf(v, -p))).sum();
        res += (d2.mapv(|v| f64::powf(v, -p)) - dist2.mapv(|v| f64::powf(v, -p))).sum();
        res = f64::powf(f64::powf(phip, p) + res, 1. / p);

        // swap points
        x.swap([i1, k], [i2, k]);
        res
    }

    fn _normalized_classic_lhs(&self, ns: usize, rng: &mut R) -> Array2<f64> {
        let nx = self.xlimits.nrows();
        let cut = Array::linspace(0., 1., ns + 1);

        let u = Array::random_using((ns, nx), Uniform::new(0., 1.), rng);
        let a = cut.slice(s![..ns]).to_owned();
        let b = cut.slice(s![1..(ns + 1)]);
        let c = &b - &a;
        let mut rdpoints = Array::zeros((ns, nx));
        for j in 0..nx {
            let d = u.column(j).to_owned() * &c + &a;
            rdpoints.column_mut(j).assign(&d)
        }
        let mut lhs = Array::zeros((ns, nx));
        for j in 0..nx {
            let mut colj = rdpoints.slice(s![.., j]).to_owned();
            colj.as_slice_mut().unwrap().shuffle(rng);
            lhs.column_mut(j).assign(&colj);
        }
        lhs
    }

    pub fn _normalized_centered_lhs(&self, ns: usize, rng: &mut R) -> Array2<f64> {
        let nx = self.xlimits.nrows();
        let cut = Array::linspace(0., 1., ns + 1);

        let u = Array::random((ns, nx), Uniform::new(0., 1.));
        let a = cut.slice(s![..ns]).to_owned();
        let b = cut.slice(s![1..(ns + 1)]);
        let mut c = (a + b) / 2.;
        let mut lhs = Array::zeros(u.raw_dim());

        for j in 0..nx {
            c.as_slice_mut().unwrap().shuffle(rng);
            lhs.column_mut(j).assign(&c);
        }
        lhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr2, array};
    use std::time::Instant;

    #[test]
    fn test_lhs() {
        let xlimits = arr2(&[[5., 10.], [0., 1.]]);
        let expected = array![
            [9.529012065390592, 0.7501017996171474],
            [5.328628661104205, 0.263722994415054],
            [6.207914396957061, 0.8443641533423626],
            [8.271960129557709, 0.02969642071371703],
            [7.588272879378793, 0.4475080605957672]
        ];
        let actual = LHS::new(&xlimits).sample(5);
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }

    #[test]
    fn test_lhs_speed() {
        let start = Instant::now();
        let xlimits = arr2(&[[0., 1.], [0., 1.]]);
        let n = 10;
        let actual = LHS::new(&xlimits).sample(n);
        let duration = start.elapsed();
        println!("Time elapsed in optimized LHS is: {:?}", duration);
    }

    #[test]
    fn test_classic_lhs() {
        let xlimits = arr2(&[[5., 10.], [0., 1.]]);
        let expected = array![
            [9.529012065390592, 0.7501017996171474],
            [5.328628661104205, 0.263722994415054],
            [6.207914396957061, 0.4475080605957672],
            [8.271960129557709, 0.02969642071371703],
            [7.588272879378793, 0.8443641533423626]
        ];
        let actual = LHS::new(&xlimits).kind(LHSKind::Classic).sample(5);
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }

    #[test]
    fn test_centered_lhs() {
        let xlimits = arr2(&[[5., 10.], [0., 1.]]);
        let expected = array![[9.5, 0.3], [8.5, 0.5], [6.5, 0.1], [5.5, 0.7], [7.5, 0.9]];
        let actual = LHS::new(&xlimits)
            .with_rng(Isaac64Rng::seed_from_u64(0))
            .kind(LHSKind::Centered)
            .sample(5);
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }

    #[test]
    fn test_phip_swap() {
        let xlimits = arr2(&[[0., 1.], [0., 1.]]);
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
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let res = LHS::new(&xlimits)._phip_swap(&mut p0, k, phip, p, &mut rng);
    }
}
