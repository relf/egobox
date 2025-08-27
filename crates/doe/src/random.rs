use std::sync::{Arc, RwLock};

use crate::SamplingMethod;
use linfa::Float;
use ndarray::{Array, Array2, ArrayBase, Data, Ix2};
use ndarray_rand::{RandomExt, rand::Rng, rand::SeedableRng, rand_distr::Uniform};
use rand_xoshiro::Xoshiro256Plus;

#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

type RngRef<R> = Arc<RwLock<R>>;
/// The Random design consists in drawing samples randomly.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct Random<F: Float, R: Rng> {
    /// Sampling space definition as a (nx, 2) matrix
    /// The ith row is the [lower_bound, upper_bound] of xi, the ith component of x
    xlimits: Array2<F>,
    /// Random generator used for reproducibility
    rng: RngRef<R>,
}

impl<F: Float> Random<F, Xoshiro256Plus> {
    /// Constructor given a design space given a (nx, 2) matrix \[\[lower bound, upper bound\], ...\]
    ///
    /// ```
    /// use egobox_doe::Random;
    /// use ndarray::arr2;
    ///
    /// let doe = Random::new(&arr2(&[[0.0, 1.0], [5.0, 10.0]]));
    /// ```
    pub fn new(xlimits: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Self {
        Self::new_with_rng(xlimits, Xoshiro256Plus::from_entropy())
    }
}

impl<F: Float, R: Rng> Random<F, R> {
    /// Constructor given a design space given a (nx, 2) matrix \[\[lower bound, upper bound\], ...\]
    /// and a random generator for reproducibility
    ///
    /// **Panics** if xlimits number of columns is different from 2.
    pub fn new_with_rng(xlimits: &ArrayBase<impl Data<Elem = F>, Ix2>, rng: R) -> Self {
        if xlimits.ncols() != 2 {
            panic!("xlimits must have 2 columns (lower, upper)");
        }
        Random {
            xlimits: xlimits.to_owned(),
            rng: Arc::new(RwLock::new(rng)),
        }
    }

    /// Set random generator
    pub fn with_rng<R2: Rng>(self, rng: R2) -> Random<F, R2> {
        Random {
            xlimits: self.xlimits,
            rng: Arc::new(RwLock::new(rng)),
        }
    }
}

impl<F: Float, R: Rng> SamplingMethod<F> for Random<F, R> {
    fn sampling_space(&self) -> &Array2<F> {
        &self.xlimits
    }

    fn normalized_sample(&self, ns: usize) -> Array2<F> {
        let mut rng = self.rng.write().unwrap();
        let nx = self.xlimits.nrows();
        Array::random_using((ns, nx), Uniform::new(0., 1.), &mut *rng).mapv(|v| F::cast(v))
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
            [5.4287779764773045, 0.31041139572710486],
            [5.31284890781607, 0.306461322653673],
            [5.0002147942961885, 0.3030653113049855],
            [5.438048037018622, 0.2270337387265695],
            [9.31397733563812, 0.5232539513550647],
            [6.0549173955055435, 0.8198009346946455],
            [8.303444344933911, 0.8588635290560207],
            [5.721154177502889, 0.3516459308028457],
            [5.457086177138239, 0.11691074717669259]
        ];
        let actual = Random::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(9);
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }
}
