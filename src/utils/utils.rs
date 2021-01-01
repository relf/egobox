use linfa::Float;
use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix1, Ix2};
use ndarray_linalg::{cholesky::*, Lapack, Scalar};
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use ndarray_rand::{rand::Rng, RandomExt};
use ndarray_stats::DeviationExt;

pub struct NormalizedMatrix<F: Float> {
    pub data: Array2<F>,
    pub mean: Array1<F>,
    pub std: Array1<F>,
}

impl<F: Float> Clone for NormalizedMatrix<F> {
    fn clone(&self) -> NormalizedMatrix<F> {
        NormalizedMatrix {
            data: self.data.to_owned(),
            mean: self.mean.to_owned(),
            std: self.std.to_owned(),
        }
    }
}

impl<F: Float> NormalizedMatrix<F> {
    pub fn new(x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> NormalizedMatrix<F> {
        let (data, mean, std) = normalize(x);
        NormalizedMatrix {
            data: data.to_owned(),
            mean: mean.to_owned(),
            std: std.to_owned(),
        }
    }

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
        let (d, d_indices) = Self::_l1_cross_distances(&x);
        let n_obs = x.nrows();
        let n_features = x.ncols();

        DistanceMatrix {
            d: d.to_owned(),
            d_indices: d_indices.to_owned(),
            n_obs,
            n_features,
        }
    }

    fn _l1_cross_distances(
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> (
        ArrayBase<impl Data<Elem = F>, Ix2>,
        ArrayBase<impl Data<Elem = usize>, Ix2>,
    ) {
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

pub fn pdist<F: Float>(x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array1<F> {
    let nrows = x.nrows();
    let size: usize = (nrows - 1) * nrows / 2;
    let mut res: Array1<F> = Array1::zeros(size);
    let mut k = 0;
    for i in 0..nrows {
        for j in (i + 1)..nrows {
            let a = x.slice(s![i, ..]);
            let b = x.slice(s![j, ..]);
            res[k] = F::from(a.l2_dist(&b).unwrap()).unwrap();
            k += 1;
        }
    }
    res
}

pub fn cdist<F: Float>(
    xa: &ArrayBase<impl Data<Elem = F>, Ix2>,
    xb: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Array2<F> {
    let ma = xa.nrows();
    let mb = xb.nrows();
    let na = xa.ncols();
    let nb = xb.ncols();
    if na != nb {
        panic!(
            "cdist: operands should have same nb of columns. Found {} and {}",
            na, nb
        );
    }
    let mut res = Array2::zeros((ma, mb));
    for i in 0..ma {
        for j in 0..mb {
            let a = xa.slice(s![i, ..]);
            let b = xb.slice(s![j, ..]);
            res[[i, j]] = F::from(a.l2_dist(&b).unwrap()).unwrap();
        }
    }

    res
}

pub struct MultivariateNormal<F: Float + Scalar + Lapack> {
    mean: Array1<F>,
    covariance_chol: Array2<F>,
}

impl<F: Float + Scalar + Lapack> MultivariateNormal<F> {
    pub fn new(mean: &ArrayView1<F>, covariance: &ArrayView2<F>) -> Self {
        let covariance_chol = covariance.cholesky(UPLO::Lower).unwrap();
        MultivariateNormal {
            mean: mean.to_owned(),
            covariance_chol,
        }
    }
}
impl<F: Float + Scalar + Lapack> Distribution<Array1<F>> for MultivariateNormal<F> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<F> {
        // standard normal distribution
        let res: Array1<f64> = Array1::random_using(self.mean.shape()[0], StandardNormal, rng);
        // use Cholesky decomposition to obtain a sample of our general multivariate normal
        self.mean.clone()
            + self
                .covariance_chol
                .view()
                .dot(&res.mapv(|v| F::from(v).unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, array};

    #[test]
    fn test_normalized_matrix() {
        let x = array![[1., 2.], [3., 4.]];
        let xnorm = NormalizedMatrix::new(&x);
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

    #[test]
    fn test_pdist() {
        let x = array![[1., 0., 0.], [0., 1., 0.], [0., 2., 0.], [3., 4., 5.]];
        #[allow(clippy::approx_constant)]
        let expected = array![1.41421356, 2.23606798, 6.70820393, 1., 6.55743852, 6.164414];
        let actual = pdist(&x);
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_cdist() {
        let a = array![
            [35.0456, -85.2672],
            [35.1174, -89.9711],
            [35.9728, -83.9422],
            [36.1667, -86.7833]
        ];
        let expected = array![
            [0., 4.7044, 1.6172, 1.8856],
            [4.7044, 0., 6.0893, 3.3561],
            [1.6172, 6.0893, 0., 2.8477],
            [1.8856, 3.3561, 2.8477, 0.]
        ];
        assert_abs_diff_eq!(cdist(&a, &a), expected, epsilon = 1e-4);
    }
}
