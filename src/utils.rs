use ndarray::{array, s, Array2, ArrayBase, Axis, Data, Ix1, Ix2};

pub fn normalize(
    x: &mut ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> (
    ArrayBase<impl Data<Elem = f64>, Ix1>,
    ArrayBase<impl Data<Elem = f64>, Ix1>,
) {
    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let x_std = x.std_axis(Axis(0), 1.);
    println!("x_std {:?}", x_std);

    (x_mean, x_std)
}

pub fn l1_cross_distances(
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> (
    ArrayBase<impl Data<Elem = f64>, Ix2>,
    ArrayBase<impl Data<Elem = usize>, Ix2>,
) {
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];
    let n_non_zero_cross_dist = n_samples * (n_samples - 1) / 2;
    let mut indices = Array2::<usize>::zeros((n_non_zero_cross_dist, 2));
    let mut d = Array2::zeros((n_non_zero_cross_dist, n_features));
    let mut ll_1 = 0;
    for k in 0..(n_samples - 1) {
        let ll_0 = ll_1;
        ll_1 = ll_0 + n_samples - k - 1;
        indices
            .slice_mut(s![ll_0..ll_1, 0..1])
            .assign(&Array2::<usize>::from_elem((n_samples - k - 1, 1), k));
        let init_values = ((k + 1)..n_samples).collect();
        indices
            .slice_mut(s![ll_0..ll_1, 1..2])
            .assign(&Array2::from_shape_vec((n_samples - k - 1, 1), init_values).unwrap());

        let diff = &x
            .slice(s![k..(k + 1), ..])
            .broadcast((n_samples - k - 1, n_features))
            .unwrap()
            - &x.slice(s![k + 1..n_samples, ..]);
        d.slice_mut(s![ll_0..ll_1, ..]).assign(&diff);
    }
    d = d.mapv(f64::abs);

    (d, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_normalize() {
        let mut x = array![[1., 2.], [3., 4.]];
        println!("{:?}", normalize(&mut x));
        let (mean, std) = normalize(&mut x);
        assert_eq!(array![2., 3.], mean);
        assert_eq!(array![f64::sqrt(2.), f64::sqrt(2.)], std);
    }

    fn test_l1_cross_distances() {
        let mut xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
        let mut yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];
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
        assert_eq!(expected.0, l1_cross_distances(&xt).0);
        assert_eq!(expected.1, l1_cross_distances(&xt).1);
    }
}
