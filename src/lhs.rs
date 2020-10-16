use crate::utils::pdist;
use ndarray::{s, Array, Array2, ArrayBase, Data, Ix2};
use ndarray_rand::{rand::seq::SliceRandom, rand::thread_rng, rand_distr::Uniform, RandomExt};
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

        let j = 20;
        let outer_loop = cmp::min((1.5 * nx as f64) as usize, 30);
        let inner_loop = cmp::min(20 * nx, 100);

        lhs0
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
}
