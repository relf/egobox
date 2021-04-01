use crate::SamplingMethod;
use ndarray::{s, Array, Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_stats::QuantileExt;

/// The FullFactorial design consists of all possible combinations
/// of levels for all components within the design space.
pub struct FullFactorial {
    /// Sampling space definition as a (nx, 2) matrix
    /// The ith row is the [lower_bound, upper_bound] of xi, the ith component of a sample x
    xlimits: Array2<f64>,
}

impl FullFactorial {
    pub fn new(xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Self {
        if xlimits.ncols() != 2 {
            panic!("xlimits must have 2 columns (lower, upper)");
        }
        FullFactorial {
            xlimits: xlimits.to_owned(),
        }
    }
}

impl SamplingMethod for FullFactorial {
    fn sampling_space(&self) -> &Array2<f64> {
        &self.xlimits
    }

    fn normalized_sample(&self, ns: usize) -> Array2<f64> {
        //! the number of level by components is choosen as evenly as possible
        //!
        let nx = self.xlimits.nrows();
        let weights: Array1<f64> = Array1::ones(nx) / (nx as f64);
        let mut num_list: Array1<usize> = Array::ones(nx);

        while num_list.fold(1, |acc, n| acc * n) < ns {
            let w: Array1<f64> = &num_list.mapv(|v| v as f64) / num_list.sum() as f64;
            let ind = (&weights - &w).argmax().unwrap();
            num_list[ind] += 1;
        }

        let nrows = num_list.fold(1, |acc, n| acc * n) as usize;
        let mut doe = Array2::<f64>::zeros((nrows, nx));

        let mut level_repeat = nrows;
        let mut range_repeat = 1;
        for j in 0..nx {
            let n = num_list[j];
            level_repeat /= n;
            let mut chunk = Array1::zeros(level_repeat * n);
            for i in 0..n {
                let fill = (i as f64) / (n - 1) as f64;
                chunk
                    .slice_mut(s![i * level_repeat..(i + 1) * level_repeat])
                    .assign(&Array1::from_elem(level_repeat, fill));
            }
            for k in 0..range_repeat {
                doe.slice_mut(s![n * level_repeat * k..n * level_repeat * (k + 1), j])
                    .assign(&chunk);
            }
            range_repeat *= n;
        }
        doe
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr2, array};

    #[test]
    fn test_ffact() {
        let xlimits = arr2(&[[5., 10.], [0., 1.]]);
        let expected = array![
            [5., 0.],
            [5., 0.5],
            [5., 1.],
            [7.5, 0.],
            [7.5, 0.5],
            [7.5, 1.],
            [10., 0.],
            [10., 0.5],
            [10., 1.],
        ];
        let actual = FullFactorial::new(&xlimits).sample(9);
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }
}
