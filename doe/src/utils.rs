use linfa::Float;
use ndarray::{s, Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_stats::DeviationExt;

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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

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
