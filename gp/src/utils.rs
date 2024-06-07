use linfa::Float;
use ndarray::{s, Array, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

/// A structure to store (n, xdim) matrix data and its mean and standard deviation vectors.
#[derive(Debug)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub(crate) struct NormalizedData<F: Float> {
    /// normalized data
    pub data: Array2<F>,
    /// mean vector computed from data
    pub mean: Array1<F>,
    /// standard deviation vector computed from data
    pub std: Array1<F>,
}

impl<F: Float> Clone for NormalizedData<F> {
    fn clone(&self) -> NormalizedData<F> {
        NormalizedData {
            data: self.data.to_owned(),
            mean: self.mean.to_owned(),
            std: self.std.to_owned(),
        }
    }
}

impl<F: Float> NormalizedData<F> {
    /// Constructor
    pub fn new(x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> NormalizedData<F> {
        let (data, mean, std) = normalize(x);
        NormalizedData {
            data: data.to_owned(),
            mean: mean.to_owned(),
            std: std.to_owned(),
        }
    }

    /// Dimension of data points
    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }
}

pub fn normalize<F: Float>(
    x: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> (Array2<F>, Array1<F>, Array1<F>) {
    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let mut x_std = x.std_axis(Axis(0), F::one());
    x_std.mapv_inplace(|v| if v == F::zero() { F::one() } else { v });
    let xnorm = (x - &x_mean) / &x_std;

    (xnorm, x_mean, x_std)
}

#[derive(Debug)]
pub struct DistanceMatrix<F: Float> {
    pub d: Array2<F>,
    pub d_indices: Array2<usize>,
    pub n_obs: usize,
    pub n_features: usize,
}

impl<F: Float> DistanceMatrix<F> {
    pub fn new(x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> DistanceMatrix<F> {
        let (d, d_indices) = Self::_cross_distances(x);
        let n_obs = x.nrows();
        let n_features = x.ncols();

        DistanceMatrix {
            d: d.to_owned(),
            d_indices: d_indices.to_owned(),
            n_obs,
            n_features,
        }
    }

    fn _cross_distances(x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> (Array2<F>, Array2<usize>) {
        let n_obs = x.nrows();
        let n_features = x.ncols();
        let n_non_zero_cross_dist = n_obs * (n_obs - 1) / 2;
        let mut indices = Array2::<usize>::zeros((n_non_zero_cross_dist, 2));
        let mut d = Array2::zeros((n_non_zero_cross_dist, n_features));
        let mut ll_1 = 0;
        for k in 0..(n_obs - 1) {
            let ll_0 = ll_1;
            ll_1 = ll_0 + n_obs - k - 1;
            indices
                .slice_mut(s![ll_0..ll_1, 0..1])
                .assign(&Array2::<usize>::from_elem((n_obs - k - 1, 1), k));
            let init_values = ((k + 1)..n_obs).collect();
            indices
                .slice_mut(s![ll_0..ll_1, 1..2])
                .assign(&Array2::from_shape_vec((n_obs - k - 1, 1), init_values).unwrap());

            let diff = &x
                .slice(s![k..(k + 1), ..])
                .broadcast((n_obs - k - 1, n_features))
                .unwrap()
                - &x.slice(s![k + 1..n_obs, ..]);
            d.slice_mut(s![ll_0..ll_1, ..]).assign(&diff);
        }
        d = d.mapv(|v| v.abs());

        (d, indices)
    }
}

/// Computes differences between each element of x and each element of y
/// resulting in a 2d array of shape (nrows(x) * nrows(y), ncols(x));
/// *Panics* if x and y have not the same column numbers
pub fn pairwise_differences<F: Float>(
    x: &ArrayBase<impl Data<Elem = F>, Ix2>,
    y: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Array2<F> {
    assert!(x.ncols() == y.ncols());
    let x3 = x.to_owned().insert_axis(Axis(1));
    let y3 = y.to_owned().insert_axis(Axis(0));
    let d = x3 - y3;
    let n = d.len();
    let res = Array::from_iter(d.iter().cloned());
    res.into_shape((n / x.ncols(), x.ncols())).unwrap()
}

/// Computes differences between x and each element of y
/// resulting in a 2d array of shape (nrows(y), ncols(x));
/// *Panics* if x and y have not the same number of components
pub fn differences<F: Float>(
    x: &ArrayBase<impl Data<Elem = F>, Ix1>,
    y: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Array2<F> {
    assert!(x.len() == y.ncols());
    x.to_owned() - y
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_pairwise_differences() {
        let x = array![[-0.9486833], [-0.82219219]];
        let y = array![
            [-1.26491106],
            [-0.63245553],
            [0.],
            [0.63245553],
            [1.26491106]
        ];
        assert_abs_diff_eq!(
            &array![
                [0.31622777],
                [-0.31622777],
                [-0.9486833],
                [-1.58113883],
                [-2.21359436],
                [0.44271887],
                [-0.18973666],
                [-0.82219219],
                [-1.45464772],
                [-2.08710326]
            ],
            &pairwise_differences(&x, &y),
            epsilon = 1e-6
        )
    }

    #[test]
    fn test_differences() {
        let x = array![-0.9486833];
        let y = array![
            [-1.26491106],
            [-0.63245553],
            [0.],
            [0.63245553],
            [1.26491106]
        ];
        assert_abs_diff_eq!(
            &array![
                [0.31622777],
                [-0.31622777],
                [-0.9486833],
                [-1.58113883],
                [-2.21359436],
            ],
            &differences(&x, &y),
            epsilon = 1e-6
        )
    }

    #[test]
    fn test_normalized_matrix() {
        let x = array![[1., 2.], [3., 4.]];
        let xnorm = NormalizedData::new(&x);
        assert_eq!(xnorm.ncols(), 2);
        assert_eq!(array![2., 3.], xnorm.mean);
        assert_eq!(array![f64::sqrt(2.), f64::sqrt(2.)], xnorm.std);
    }

    #[test]
    fn test_cross_distance_matrix() {
        let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
        let expected = (
            array![
                [0.7],
                [1.5],
                [2.5],
                [3.5],
                [0.8],
                [1.8],
                [2.8],
                [1.],
                [2.],
                [1.]
            ],
            array![
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [1, 2],
                [1, 3],
                [1, 4],
                [2, 3],
                [2, 4],
                [3, 4]
            ],
        );
        let dm = DistanceMatrix::new(&xt);
        assert_eq!(expected.0, dm.d);
        assert_eq!(expected.1, dm.d_indices);
    }
}
