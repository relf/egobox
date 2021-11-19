use libm::erfc;
use moe::Moe;
use ndarray::{Array1, Array2, ArrayView, Axis};
use ndarray_stats::QuantileExt;

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
