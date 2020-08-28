use ndarray::{s, Array2, ArrayBase, Axis, Data, Ix1, Ix2};

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

pub fn reduced_likelihood(
    thetas: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    d: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) {
    let res = f64::MIN;
    let nugget = 10. * f64::EPSILON;

    let r = squared_exponential(thetas, d);
    ()
}

pub fn squared_exponential(
    thetas: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    d: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> ArrayBase<impl Data<Elem = f64>, Ix2> {
    let (n_obs, n_features) = (d.shape()[0], d.shape()[1]);
    let mut r = Array2::zeros((n_obs, 1));

    let t = thetas.view().into_shape((1, n_features)).unwrap();
    let d2 = d.mapv(|v| v * v);
    let m = (d2 * &t).sum_axis(Axis(1)).mapv(|v| f64::exp(-v));
    r.slice_mut(s![.., 0]).assign(&m);
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::abs_diff_eq;
    use ndarray::{arr1, array};

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

    #[test]
    fn test_squared_exponential() {
        let xt = array![[4.5], [1.2], [2.0], [3.0], [4.0]];
        let (d, _) = l1_cross_distances(&xt);
        let res = squared_exponential(&arr1(&[0.1]), &d);
        println!("{:?}", res);
        // without squared fix as in SMT 0.5.2
        // let expected = array![
        //     [0.9323938199059483],
        //     [0.8607079764250578],
        //     [0.7788007830714049],
        //     [0.7046880897187134],
        //     [0.9231163463866358],
        //     [0.835270211411272],
        //     [0.7557837414557255],
        //     [0.9048374180359595],
        //     [0.8187307530779818],
        //     [0.9048374180359595]
        // ];
        // with squared fix
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
