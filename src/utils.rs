use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};

pub fn normalize(
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> (
    ArrayBase<impl Data<Elem = f64>, Ix2>,
    ArrayBase<impl Data<Elem = f64>, Ix1>,
    ArrayBase<impl Data<Elem = f64>, Ix1>,
) {
    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let x_std = x.std_axis(Axis(0), 1.);

    println!("{:?}", x);
    println!("{:?}", x_mean.broadcast(x.shape()).unwrap());

    let xnorm = (x - &x_mean) / &x_std;

    (xnorm, x_mean, x_std)
}

pub fn l1_cross_distances(
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> (
    ArrayBase<impl Data<Elem = f64>, Ix2>,
    ArrayBase<impl Data<Elem = usize>, Ix2>,
) {
    let n_obs = x.shape()[0];
    let n_features = x.shape()[1];
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

pub fn constant(
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> ArrayBase<impl Data<Elem = f64>, Ix2> {
    let n_obs = x.shape()[0];
    Array2::<f64>::zeros((n_obs, 1))
}

// pub fn squared_exponential(thetas: &ArrayBase<impl Data<Elem = f64>, Ix1>,
//                            d: &ArrayBase<impl Data<Elem = f64>, Ix2>)
// ) -> ArrayBase<impl Data<Elem = f64>, Ix2> {
//     let r = Array2::zeros((d.shape()[0], 1));
//     let n_features = d.shape()[1];

//     let mut i = 0;
//     let n_limit = 10000;

//     while i * nb_limit <= d.shape()[0] {
//         r.slice_mut(s![i * nb_limit .. (i + 1) * nb_limit, 0]).assign(exp(
//             -np.sum(
//                 theta.reshape(1, n_features) * d.slice(s![i * nb_limit .. (i + 1) * nb_limit, ..])  ,
//                 axis=1,
//             )
//         )
//         i += 1
//     }

//     r
// }

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_normalize() {
        let x = array![[1., 2.], [3., 4.]];
        println!("{:?}", normalize(&x));
        let (xnorm, mean, std) = normalize(&x);
        assert_eq!(xnorm.shape()[0], 2);
        assert_eq!(xnorm.shape()[1], 2);
        assert_eq!(array![2., 3.], mean);
        assert_eq!(array![f64::sqrt(2.), f64::sqrt(2.)], std);
    }

    #[test]
    fn test_l1_cross_distances() {
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
        let (actual0, actual1) = l1_cross_distances(&xt);
        assert_eq!(expected.0, actual0);
        assert_eq!(expected.1, actual1);
    }
}
