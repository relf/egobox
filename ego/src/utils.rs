use crate::types::XType;
use egobox_moe::MixtureGpSurrogate;
use libm::erfc;
use ndarray::{concatenate, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2, Zip};
use ndarray_stats::{DeviationExt, QuantileExt};
use rayon::prelude::*;
const SQRT_2PI: f64 = 2.5066282746310007;

/// Computes scaling factors used to scale constraint functions values.
pub fn compute_cstr_scales(
    x: &ArrayView2<f64>,
    cstr_models: &[Box<dyn MixtureGpSurrogate>],
) -> Array1<f64> {
    let scales: Vec<f64> = cstr_models
        .par_iter()
        .map(|cstr_model| {
            let preds: Array1<f64> = cstr_model
                .predict_values(x)
                .unwrap()
                .into_iter()
                .filter(|v| !v.is_infinite()) // filter out infinite values
                .map(|v| v.abs())
                .collect();
            *preds.max().unwrap_or(&1.0)
        })
        .collect();
    Array1::from_shape_vec(cstr_models.len(), scales).unwrap()
}

/// Cumulative distribution function of Standard Normal at x
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Probability density function of Standard Normal at x
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / SQRT_2PI
}

// DOE handling functions
///////////////////////////////////////////////////////////////////////////////

/// Check if new point is not too close to previous ones `x_data`
pub fn is_update_ok(
    x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    x_new: &ArrayBase<impl Data<Elem = f64>, Ix1>,
) -> bool {
    for row in x_data.rows() {
        if row.l1_dist(x_new).unwrap() < 1e-6 {
            return false;
        }
    }
    true
}

/// Append `x_new` (resp. `y_new`) to `x_data` (resp. y_data) if `x_new` not too close to `x_data` points
/// Returns the index of appended points in `x_new`
pub fn update_data(
    x_data: &mut Array2<f64>,
    y_data: &mut Array2<f64>,
    x_new: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    y_new: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Vec<usize> {
    let mut appended = vec![];
    Zip::indexed(x_new.rows())
        .and(y_new.rows())
        .for_each(|idx, x, y| {
            let xdat = x.insert_axis(Axis(0));
            let ydat = y.insert_axis(Axis(0));
            if is_update_ok(x_data, &x) {
                *x_data = concatenate![Axis(0), x_data.view(), xdat];
                *y_data = concatenate![Axis(0), y_data.view(), ydat];
                appended.push(idx);
            }
        });
    appended
}

pub fn discrete(xtypes: &[XType]) -> bool {
    xtypes
        .iter()
        .any(|t| matches!(t, &XType::Int(_, _) | &XType::Ord(_) | &XType::Enum(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_is_update_ok() {
        let data = array![[0., 1.], [2., 3.]];
        assert!(is_update_ok(&data, &array![3., 4.]));
        assert!(!is_update_ok(&data, &array![1e-7, 1.]));
    }

    #[test]
    fn test_update_data() {
        let mut xdata = array![[0., 1.], [2., 3.]];
        let mut ydata = array![[3.], [4.]];
        assert_eq!(
            update_data(
                &mut xdata,
                &mut ydata,
                &array![[3., 4.], [1e-7, 1.]],
                &array![[6.], [7.]],
            ),
            &[0]
        );
        assert_eq!(&array![[0., 1.], [2., 3.], [3., 4.]], xdata);
        assert_eq!(&array![[3.], [4.], [6.]], ydata);
    }
}
