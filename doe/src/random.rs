use crate::SamplingMethod;
use ndarray::{Array, Array2, ArrayBase, Data, Ix2};
use ndarray_rand::{rand::Rng, rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand_isaac::Isaac64Rng;

/// The Random design consists in drawing samples randomly.
pub struct Random<R: Rng + Clone> {
    /// Sampling space definition as a (nx, 2) matrix
    /// The ith row is the [lower_bound, upper_bound] of xi, the ith component of x
    xlimits: Array2<f64>,
    /// Random generator used for reproducibility (not used in case of Centered LHS)
    rng: R,
}

impl Random<Isaac64Rng> {
    pub fn new(xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Self {
        Self::new_with_rng(xlimits, Isaac64Rng::seed_from_u64(42))
    }
}

impl<R: Rng + Clone> Random<R> {
    pub fn new_with_rng(xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>, rng: R) -> Self {
        if xlimits.ncols() != 2 {
            panic!("xlimits must have 2 columns (lower, upper)");
        }
        Random {
            xlimits: xlimits.to_owned(),
            rng,
        }
    }

    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> Random<R2> {
        Random {
            xlimits: self.xlimits,
            rng,
        }
    }
}

impl<R: Rng + Clone> SamplingMethod for Random<R> {
    fn sampling_space(&self) -> &Array2<f64> {
        &self.xlimits
    }

    fn normalized_sample(&self, ns: usize) -> Array2<f64> {
        let mut rng = self.rng.clone();
        let nx = self.xlimits.nrows();
        Array::random_using((ns, nx), Uniform::new(0., 1.), &mut rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr2, array};

    #[test]
    fn test_random() {
        let xlimits = arr2(&[[5., 10.], [0., 1.]]);
        let expected = array![
            [7.35493967304704, 0.778179233288187],
            [8.64019935221146, 0.24949100191426288],
            [7.058635603106036, 0.7624596970211635],
            [8.960428015214697, 0.68138502792473],
            [8.356856694478974, 0.8515178964314147],
            [6.559097971176039, 0.7398254113552798],
            [5.452725391445714, 0.7288312058240056],
            [5.2348966430803925, 0.5846614636962431],
            [8.02670850570956, 0.2310179777619814]
        ];
        let actual = Random::new(&xlimits).sample(9);
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }
}