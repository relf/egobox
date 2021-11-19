use libm::erfc;
use moe::Moe;
use ndarray::{concatenate, Array1, Array2, ArrayBase, ArrayView, Axis, Data, Ix1, Ix2, Zip};
use ndarray_stats::{DeviationExt, QuantileExt};

// Infill strategy related function
////////////////////////////////////////////////////////////////////////////

const SQRT_2PI: f64 = 2.5066282746310007;

pub fn ei(x: &[f64], obj_model: &Moe, f_min: f64) -> f64 {
    let pt = ArrayView::from_shape((1, x.len()), x).unwrap().to_owned();
    if let Ok(p) = obj_model.predict_values(&pt) {
        if let Ok(s) = obj_model.predict_variances(&pt) {
            let pred = p[[0, 0]];
            let sigma = s[[0, 0]].sqrt();
            let args0 = (f_min - pred) / sigma;
            let args1 = (f_min - pred) * norm_cdf(args0);
            let args2 = sigma * norm_pdf(args0);
            args1 + args2
        } else {
            -f64::INFINITY
        }
    } else {
        -f64::INFINITY
    }
}

pub fn wb2s(x: &[f64], obj_model: &Moe, f_min: f64, scale: f64) -> f64 {
    let pt = ArrayView::from_shape((1, x.len()), x).unwrap().to_owned();
    let ei = ei(x, obj_model, f_min);
    scale * ei - obj_model.predict_values(&pt).unwrap()[[0, 0]]
}

pub fn compute_wb2s_scale(x: &Array2<f64>, obj_model: &Moe, f_min: f64) -> f64 {
    let ratio = 100.; // TODO: make it a parameter
    let ei_x = x.map_axis(Axis(1), |xi| {
        let ei = ei(xi.as_slice().unwrap(), obj_model, f_min);
        ei
    });
    let i_max = ei_x.argmax().unwrap();
    let pred_max = obj_model
        .predict_values(&x.row(i_max).insert_axis(Axis(0)).to_owned())
        .unwrap()[[0, 0]];
    let ei_max = ei_x[i_max];
    if ei_max > 0. {
        ratio * pred_max / ei_max
    } else {
        1.
    }
}

pub fn compute_obj_scale(x: &Array2<f64>, obj_model: &Moe) -> f64 {
    let preds = obj_model.predict_values(x).unwrap().mapv(|v| f64::abs(v));
    *preds.max().unwrap_or(&1.0)
}

pub fn compute_cstr_scales(x: &Array2<f64>, cstr_models: &Vec<Box<Moe>>) -> Array1<f64> {
    let scales: Vec<f64> = cstr_models
        .iter()
        .map(|cstr_model| {
            let preds = cstr_model.predict_values(x).unwrap().mapv(|v| f64::abs(v));
            *preds.max().unwrap_or(&1.0)
        })
        .collect();
    Array1::from_shape_vec(cstr_models.len(), scales).unwrap()
}

fn norm_cdf(x: f64) -> f64 {
    let norm = 0.5 * erfc(-x / std::f64::consts::SQRT_2);
    norm
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / SQRT_2PI
}

// DOE handling functions
///////////////////////////////////////////////////////////////////////////////
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

pub fn update_data(
    x_data: &mut Array2<f64>,
    y_data: &mut Array2<f64>,
    x_new: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    y_new: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> usize {
    let mut out_count = 0;
    Zip::from(x_new.rows()).and(y_new.rows()).for_each(|x, y| {
        let xdat = x.insert_axis(Axis(0));
        let ydat = y.insert_axis(Axis(0));
        if is_update_ok(x_data, &x) {
            *x_data = concatenate![Axis(0), x_data.view(), xdat];
            *y_data = concatenate![Axis(0), y_data.view(), ydat];
        } else {
            out_count += 1;
        }
    });
    out_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_is_update_ok() {
        let data = array![[0., 1.], [2., 3.]];
        assert_eq!(true, is_update_ok(&data, &array![3., 4.]));
        assert_eq!(false, is_update_ok(&data, &array![1e-7, 1.]));
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
            1
        );
        assert_eq!(&array![[0., 1.], [2., 3.], [3., 4.]], xdata);
        assert_eq!(&array![[3.], [4.], [6.]], ydata);
    }
}
