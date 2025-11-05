use linfa::Float;
use ndarray::{Array, Array1, Array2, ArrayBase, Data, Ix2, Zip};
use ndarray_stats::DeviationExt;
use rayon::prelude::*;

/// Computes the pairwise distances between rows of a 2D-array using parallel processing
/// Warning : The result is expected to b=e used in a context where order does not matter
/// (e.g., get min distance) as the order of distances depends on the order of parallel execution
pub fn pdist<F: Float>(x: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>) -> Array1<F> {
    let nrows = x.nrows();

    // Parallelize the outer loop for better performance
    let pairs: Vec<_> = (0..nrows)
        .flat_map(|i| ((i + 1)..nrows).map(move |j| (i, j)))
        .collect();

    let distances: Vec<_> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let a = x.row(i);
            let b = x.row(j);
            F::cast(a.l2_dist(&b).unwrap())
        })
        .collect();

    Array::from_vec(distances)
}

/// Computes the pairwise distances between rows of two 2D arrays using parallel processing
/// The resulting array has shape (ma, mb) where ma is the number of rows in xa and mb is the number of rows in xb
pub fn cdist<F: Float>(
    xa: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
    xb: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
) -> Array2<F> {
    let ma = xa.nrows();
    let mb = xb.nrows();
    let na = xa.ncols();
    let nb = xb.ncols();
    if na != nb {
        panic!("cdist: operands should have same nb of columns. Found {na} and {nb}");
    }

    let mut res = Array2::zeros((ma, mb));
    Zip::from(res.rows_mut())
        .and(xa.rows())
        .par_for_each(|mut row_res, row_a| {
            for (j, row_b) in xb.rows().into_iter().enumerate() {
                row_res[j] = F::cast(row_a.l2_dist(&row_b).unwrap());
            }
        });

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
