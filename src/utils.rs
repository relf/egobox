use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};

pub struct NormalizedMatrix {
    pub data: Array2<f64>,
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
}

impl NormalizedMatrix {
    pub fn new(x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> NormalizedMatrix {
        let (data, mean, std) = Self::normalize(x);
        NormalizedMatrix {
            data: data.to_owned(),
            mean: mean.to_owned(),
            std: std.to_owned(),
        }
    }

    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }

    fn normalize(
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> (
        ArrayBase<impl Data<Elem = f64>, Ix2>,
        ArrayBase<impl Data<Elem = f64>, Ix1>,
        ArrayBase<impl Data<Elem = f64>, Ix1>,
    ) {
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let x_std = x.std_axis(Axis(0), 1.);
        let xnorm = (x - &x_mean) / &x_std;

        (xnorm, x_mean, x_std)
    }
}

pub struct DistanceMatrix {
    pub d: Array2<f64>,
    pub d_indices: Array2<usize>,
    pub f: Array2<f64>,
    pub n_obs: usize,
    pub n_features: usize,
}

impl DistanceMatrix {
    pub fn new(x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> DistanceMatrix {
        let (d, d_indices) = Self::_l1_cross_distances(&x);
        let f = constant(&x);
        let n_obs = x.nrows();
        let n_features = x.ncols();

        DistanceMatrix {
            d: d.to_owned(),
            d_indices: d_indices.to_owned(),
            f: f.to_owned(),
            n_obs,
            n_features,
        }
    }

    fn _l1_cross_distances(
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> (
        ArrayBase<impl Data<Elem = f64>, Ix2>,
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
        d = d.mapv(f64::abs);

        (d, indices)
    }
}

pub fn constant(
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> ArrayBase<impl Data<Elem = f64>, Ix2> {
    let n_obs = x.shape()[0];
    Array2::<f64>::ones((n_obs, 1))
}

pub fn squared_exponential(
    theta: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    d: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> ArrayBase<impl Data<Elem = f64>, Ix2> {
    let (n_obs, n_features) = (d.nrows(), d.ncols());
    let mut r = Array2::zeros((n_obs, 1));

    let t = theta.view().into_shape((1, n_features)).unwrap();
    let d2 = d.mapv(|v| v * v);
    let m = (d2 * t).sum_axis(Axis(1)).mapv(|v| f64::exp(-v));
    r.slice_mut(s![.., 0]).assign(&m);
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::abs_diff_eq;
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
    fn test_squared_exponential() {
        let xt = array![[4.5], [1.2], [2.0], [3.0], [4.0]];
        let dm = DistanceMatrix::new(&xt);
        let res = squared_exponential(&arr1(&[0.1]), &dm.d);
        let expected = array![
            [0.336552878364737],
            [0.5352614285189903],
            [0.7985162187593771],
            [0.9753099120283326],
            [0.9380049995307295],
            [0.7232502423798424],
            [0.4565760496233148],
            [0.9048374180359595],
            [0.6703200460356393],
            [0.9048374180359595]
        ];
        assert!(abs_diff_eq!(res[[4, 0]], expected[[4, 0]], epsilon = 1e-6));
    }
}
