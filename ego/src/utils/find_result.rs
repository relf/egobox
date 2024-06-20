use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip};
use ndarray_stats::QuantileExt;

use crate::utils::sort_axis::*;
use linfa::Float;
use ndarray::{array, s};
use std::iter::zip;

fn cstr_sum<F: Float>(y: &ArrayBase<impl Data<Elem = F>, Ix1>, cstr_tol: &Array1<F>) -> F {
    y.slice(s![1..])
        .iter()
        .enumerate()
        .filter(|(i, &c)| c > cstr_tol[*i])
        .fold(F::zero(), |acc, (i, &c)| acc + (c - cstr_tol[i]).abs())
}

pub fn cstr_min<F: Float>(
    (_, y1): (usize, &ArrayBase<impl Data<Elem = F>, Ix1>),
    (_, y2): (usize, &ArrayBase<impl Data<Elem = F>, Ix1>),
    cstr_tol: &Array1<F>,
) -> std::cmp::Ordering {
    if y1.len() > 1 {
        let sum_c1 = cstr_sum(y1, cstr_tol);
        let sum_c2 = cstr_sum(y2, cstr_tol);
        if sum_c1 > F::zero() && sum_c2 > F::zero() {
            sum_c1.partial_cmp(&sum_c2).unwrap()
        } else if sum_c1 == F::zero() && sum_c2 == F::zero() {
            y1[0].partial_cmp(&y2[0]).unwrap()
        } else if sum_c1 == F::zero() {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    } else {
        // unconstrained optimization
        y1[0].partial_cmp(&y2[0]).unwrap()
    }
}

/// This method find the best result in ydata wrt cstr_min partial order
/// given the current best index of the current best and the offset index
/// the starting index of the new data to compare  
///
/// This method supposes indexes consistency: current_index < offset_index < ydata.nrows()
pub fn find_best_result_index_from<F: Float>(
    current_index: usize, /* current best index */
    offset_index: usize,
    ydata: &ArrayBase<impl Data<Elem = F>, Ix2>, /* the whole data so far */
    cstr_tol: &Array1<F>,
) -> usize {
    let new_ydata = ydata.slice(s![offset_index.., ..]);

    let best = ydata.row(current_index);
    let min = new_ydata
        .outer_iter()
        .enumerate()
        .fold((usize::MAX, best), |a, b| {
            std::cmp::min_by(a, b, |(i, u), (j, v)| cstr_min((*i, u), (*j, v), cstr_tol))
        });
    match min {
        (usize::MAX, _) => current_index,
        (index, _) => offset_index + index,
    }
}

/// Find best (eg minimal) cost value (y_data\[0\]) with valid constraints (y_data\[1..\] < cstr_tol).
/// y_data containing ns samples [objective, cstr_1, ... cstr_nc] is given as a matrix (ns, nc + 1)  
pub fn find_best_result_index<F: Float>(
    y_data: &ArrayBase<impl Data<Elem = F>, Ix2>,
    cstr_tol: &Array1<F>,
) -> usize {
    if y_data.ncols() > 1 {
        // Compute sum of violated constraints
        let cstrs = y_data.slice(s![.., 1..]);
        let mut c_obj = Array2::zeros((y_data.nrows(), 2));

        Zip::from(c_obj.rows_mut())
            .and(cstrs.rows())
            .and(y_data.slice(s![.., 0]))
            .for_each(|mut c_obj_row, c_row, obj| {
                let c_sum = zip(c_row, cstr_tol)
                    .filter(|(c, ctol)| *c > ctol)
                    .fold(F::zero(), |acc, (c, ctol)| acc + (*c - *ctol).abs());
                c_obj_row.assign(&array![c_sum, *obj]);
            });
        let min_csum_index = c_obj.slice(s![.., 0]).argmin().ok();

        if let Some(min_index) = min_csum_index {
            if c_obj[[min_index, 0]] > F::zero() {
                // There is no feasible point take minimal cstr sum as the best one
                min_index
            } else {
                // There is at least one or several point feasible, take the minimal objective among them
                let mut index = 0;
                let mut y_best = F::infinity();
                Zip::indexed(c_obj.rows()).for_each(|i, c_o| {
                    if c_o[0] == F::zero() && c_o[1] < y_best {
                        y_best = c_o[1];
                        index = i;
                    }
                });
                index
            }
        } else {
            // Take min obj without looking at constraints
            let mut index = 0;

            // sort regardoing minimal objective
            let perm = y_data.sort_axis_by(Axis(0), |i, j| y_data[[i, 0]] < y_data[[j, 0]]);
            let y_sort = y_data.to_owned().permute_axis(Axis(0), &perm);

            // Take the first one which do not violate constraints
            for (i, row) in y_sort.axis_iter(Axis(0)).enumerate() {
                let success =
                    zip(row.slice(s![1..]), cstr_tol).fold(true, |acc, (c, tol)| acc && c < tol);

                if success {
                    index = i;
                    break;
                }
            }
            perm.indices[index]
        }
    } else {
        // unconstrained optimization
        let y_best = y_data.column(0).argmin().unwrap();
        y_best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_cstr_min() {
        // cstr respected, y1 > y2
        let (y1, y2) = (array![1.0, -0.15], array![-1.0, -0.01]);
        let cstr_tol = Array1::from_elem(1, 0.1);
        assert_eq!(
            std::cmp::Ordering::Greater,
            cstr_min((0, &y1), (1, &y2), &cstr_tol)
        );
        // cstr respected, y1 > y2
        let (y1, y2) = (array![-1.0, -0.01], array![-2.0, -0.2]);
        let cstr_tol = Array1::from_elem(1, 0.1);
        assert_eq!(
            std::cmp::Ordering::Greater,
            cstr_min((0, &y1), (1, &y2), &cstr_tol)
        );
        // cstr out of tolerance, y1 < y2
        let (y1, y2) = (array![-2.0, 1.], array![-1.0, 2.]);
        let cstr_tol = Array1::from_elem(1, 0.1);
        assert_eq!(
            std::cmp::Ordering::Less,
            cstr_min((0, &y1), (1, &y2), &cstr_tol)
        );
        // cstr1 out of tolerance, y1 > y2
        let (y1, y2) = (array![-2.0, 1.], array![-1.0, 0.01]);
        let cstr_tol = Array1::from_elem(1, 0.1);
        assert_eq!(
            std::cmp::Ordering::Greater,
            cstr_min((0, &y1), (1, &y2), &cstr_tol)
        );
    }

    #[test]
    fn test_find_best_obj() {
        // respect constraint (0, 1, 2) and minimize obj (1)
        let ydata = array![[1.0, -0.15], [-1.0, -0.01], [2.0, -0.2], [-3.0, 2.0]];
        let cstr_tol = Array1::from_elem(4, 0.1);
        assert_abs_diff_eq!(1, find_best_result_index(&ydata, &cstr_tol));

        // respect constraint (0, 1, 2) and minimize obj (2)
        let ydata = array![[1.0, -0.15], [-1.0, -0.01], [-2.0, -0.2], [-3.0, 2.0]];
        let cstr_tol = Array1::from_elem(4, 0.1);
        assert_abs_diff_eq!(2, find_best_result_index(&ydata, &cstr_tol));

        // all out of tolerance => minimize constraint overshoot sum (0)
        let ydata = array![[1.0, 0.15], [-1.0, 0.3], [2.0, 0.2], [-3.0, 2.0]];
        let cstr_tol = Array1::from_elem(4, 0.1);
        assert_abs_diff_eq!(0, find_best_result_index(&ydata, &cstr_tol));

        // all in tolerance => min obj
        let ydata = array![[1.0, 0.15], [-1.0, 0.3], [2.0, 0.2], [-3.0, 2.0]];
        let cstr_tol = Array1::from_elem(4, 3.0);
        assert_abs_diff_eq!(3, find_best_result_index(&ydata, &cstr_tol));

        // unconstrained => min obj
        let ydata = array![[1.0], [-1.0], [2.0], [-3.0]];
        let cstr_tol = Array1::from_elem(4, 0.1);
        assert_abs_diff_eq!(3, find_best_result_index(&ydata, &cstr_tol));
    }

    #[test]
    fn test_find_best_result() {
        let y_data = array![
            [-1.05044744051, -3.854157649246, 0.03068950747],
            [-2.74965213562, -1.955703115787, -0.70939921583],
            [-2.35364705246, 0.322760821911, -31.26001920874],
            [-4.30684045535, -0.188609601161, 0.12375208631],
            [-2.66585377971, -1.665992782883, -3.31489212502],
            [-5.76598597442, 1.767753631322, -0.23219495778],
            [-3.84677718652, -0.164470342807, -0.43935857142],
            [-4.23672675117, -2.343687786724, -0.86266607911],
            [-1.23999705899, -1.653209288978, -12.42363834689],
            [-5.81590801801, -11.725502513342, 2.72175031293],
            [-5.57379997815, -0.075893786744, 0.12260068082],
            [-5.26821022904, -0.093334332384, -0.29931405911],
            [-5.50558228637, -0.008847697249, 0.00015874647],
            [-5.50802373110, -2.479726473358e-5, 2.46930218281e-5],
            [-5.50802210236, -2.586721399788e-5, 2.28386911871e-5],
            [-5.50801726964, 2.607167473023e-7, 5.50684865174e-6],
            [-5.50801509642, 1.951629235996e-7, 2.48275059533e-6],
            [-5.50801399313, -6.707576982734e-8, 1.03991762046e-6]
        ];
        let cstr_tol = Array1::from_vec(vec![1e-6; 2]); // this is the default
        let index = find_best_result_index(&y_data, &cstr_tol);
        assert_eq!(11, index);
        let cstr_tol = Array1::from_vec(vec![2e-6; 2]);
        let index = find_best_result_index(&y_data, &cstr_tol);
        assert_eq!(17, index);
    }

    #[test]
    fn test_find_best_result2() {
        let y_data = array![
            [-1.05044744051, -3.854157649246, 0.03068950747],
            [-2.74965213562, -1.955703115787, -0.70939921583],
            [-2.35364705246, 0.322760821911, -31.26001920874],
            [-4.30684045535, -0.188609601161, 0.12375208631],
            [-2.66585377971, -1.665992782883, -3.31489212502],
            [-5.76598597442, 1.767753631322, -0.23219495778],
            [-3.84677718652, -0.164470342807, -0.43935857142],
            [-4.23672675117, -2.343687786724, -0.86266607911],
            [-1.23999705899, -1.653209288978, -12.42363834689],
            [-5.81590801801, -11.725502513342, 2.72175031293],
            [-5.57379997815, -0.075893786744, 0.12260068082],
            [-5.26821022904, -0.093334332384, -0.29931405911],
            [-5.50558228637, -0.008847697249, 0.00015874647],
            [-5.50802373110, -2.479726473358e-5, 2.46930218281e-5],
            [-5.50802210236, -2.586721399788e-5, 2.28386911871e-5],
            [-5.50801726964, 2.607167473023e-7, 5.50684865174e-6],
            [-5.50801509642, 1.951629235996e-7, 2.48275059533e-6],
            [-5.50801399313, -6.707576982734e-8, 1.03991762046e-6]
        ];
        let cstr_tol = Array1::from_vec(vec![2e-6; 2]);
        let index = find_best_result_index_from(11, 12, &y_data, &cstr_tol);
        assert_eq!(17, index);
        let cstr_tol = Array1::from_vec(vec![2e-6; 2]);
        let index = find_best_result_index(&y_data, &cstr_tol);
        assert_eq!(17, index);
    }
}
