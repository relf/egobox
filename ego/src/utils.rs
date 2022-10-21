use egobox_moe::ClusteredSurrogate;
use libm::erfc;
use ndarray::{
    concatenate, Array1, Array2, ArrayBase, ArrayView, ArrayView2, Axis, Data, Ix1, Ix2, Zip,
};
use ndarray_stats::{DeviationExt, QuantileExt};

// Infill strategy related function
////////////////////////////////////////////////////////////////////////////

const SQRT_2PI: f64 = 2.5066282746310007;

pub fn ei(x: &[f64], obj_model: &dyn ClusteredSurrogate, f_min: f64) -> f64 {
    let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
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

pub fn grad_ei(x: &[f64], obj_model: &dyn ClusteredSurrogate, f_min: f64) -> Array1<f64> {
    let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
    if let Ok(p) = obj_model.predict_values(&pt) {
        if let Ok(s) = obj_model.predict_variances(&pt) {
            let sigma = s[[0, 0]].sqrt();
            if sigma.abs() < 1e-12 {
                Array1::zeros(pt.len())
            } else {
                let pred = p[[0, 0]];
                let diff_y = f_min - pred;
                let arg = (f_min - pred) / sigma;
                let y_prime = obj_model.predict_derivatives(&pt).unwrap();
                let y_prime = y_prime.row(0);
                let sig_2_prime = obj_model.predict_variance_derivatives(&pt).unwrap();

                let sig_2_prime = sig_2_prime.row(0);
                let sig_prime = sig_2_prime.mapv(|v| v / (2. * sigma));
                let arg_prime = y_prime.mapv(|v| v / (-sigma))
                    - diff_y.to_owned() * sig_prime.mapv(|v| v / (sigma * sigma));
                let factor = sigma * (-arg / SQRT_2PI) * (-(arg * arg) / 2.).exp();

                let arg1 = y_prime.mapv(|v| v * (-norm_cdf(arg)));
                let arg2 = diff_y * norm_pdf(arg) * arg_prime.to_owned();
                let arg3 = sig_prime.to_owned() * norm_pdf(arg);
                let arg4 = factor * arg_prime;
                arg1 + arg2 + arg3 + arg4
            }
        } else {
            Array1::zeros(pt.len())
        }
    } else {
        Array1::zeros(pt.len())
    }
}

pub fn wb2s(x: &[f64], obj_model: &dyn ClusteredSurrogate, f_min: f64, scale: f64) -> f64 {
    let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
    let ei = ei(x, obj_model, f_min);
    scale * ei - obj_model.predict_values(&pt).unwrap()[[0, 0]]
}

pub fn grad_wbs2(
    x: &[f64],
    obj_model: &dyn ClusteredSurrogate,
    f_min: f64,
    scale: f64,
) -> Array1<f64> {
    let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
    let grad_ei = grad_ei(x, obj_model, f_min) * scale;
    grad_ei - obj_model.predict_derivatives(&pt).unwrap().row(0)
}

pub fn compute_wb2s_scale(
    x: &ArrayView2<f64>,
    obj_model: &dyn ClusteredSurrogate,
    f_min: f64,
) -> f64 {
    let ratio = 100.; // TODO: make it a parameter
    let ei_x = x.map_axis(Axis(1), |xi| {
        let ei = ei(xi.as_slice().unwrap(), obj_model, f_min);
        ei
    });
    let i_max = ei_x.argmax().unwrap();
    let pred_max = obj_model
        .predict_values(&x.row(i_max).insert_axis(Axis(0)))
        .unwrap()[[0, 0]];
    let ei_max = ei_x[i_max];
    if ei_max > 0. {
        ratio * pred_max / ei_max
    } else {
        1.
    }
}

pub fn compute_obj_scale(x: &ArrayView2<f64>, obj_model: &dyn ClusteredSurrogate) -> f64 {
    let preds = obj_model.predict_values(x).unwrap().mapv(f64::abs);
    *preds.max().unwrap_or(&1.0)
}

pub fn compute_cstr_scales(
    x: &ArrayView2<f64>,
    cstr_models: &[Box<dyn ClusteredSurrogate>],
) -> Array1<f64> {
    let scales: Vec<f64> = cstr_models
        .iter()
        .map(|cstr_model| {
            let preds = cstr_model.predict_values(x).unwrap().mapv(f64::abs);
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

#[cfg(test)]
mod tests {
    use super::*;
    use egobox_doe::SamplingMethod;
    use egobox_moe::*;
    use linfa::prelude::*;
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

    fn sphere(x: &Array2<f64>) -> Array2<f64> {
        let s = (x * x).sum_axis(Axis(1));
        s.insert_axis(Axis(1))
    }
    #[test]
    fn test_grad_wbs2() {
        let xt = egobox_doe::Lhs::new(&array![[-10., 10.], [-10., 10.]]).sample(10);
        let yt = sphere(&xt);
        let gp = Moe::params()
            .regression_spec(RegressionSpec::CONSTANT)
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
            .recombination(Recombination::Hard)
            .fit(&Dataset::new(xt, yt))
            .expect("GP fitting");
        let bgp = Box::new(gp) as Box<dyn ClusteredSurrogate>;

        let h = 1e-4;
        let x1 = 0.1;
        let x2 = 0.3;
        let xtest = vec![x1, x2];
        let xtest11 = vec![x1 + h, x2];
        let xtest12 = vec![x1 - h, x2];
        let xtest21 = vec![x1, x2 + h];
        let xtest22 = vec![x1, x2 - h];
        let fdiff1 = (wb2s(&xtest11, bgp.as_ref(), 0.1, 0.5)
            - wb2s(&xtest12, bgp.as_ref(), 0.1, 0.5))
            / (2. * h);
        let fdiff2 = (wb2s(&xtest21, bgp.as_ref(), 0.1, 0.5)
            - wb2s(&xtest22, bgp.as_ref(), 0.1, 0.5))
            / (2. * h);
        println!("fdiff({:?}) = [{}, {}]", xtest, fdiff1, fdiff2);
        println!(
            "grad_wbs2({:?}) = {:?}",
            xtest,
            grad_wbs2(&xtest21, bgp.as_ref(), 0.1, 0.5)
        );

        let h = 1e-4;
        let x1 = 0.5;
        let x2 = 0.7;
        let xtest = array![[x1, x2]];
        let basetest = xtest.to_owned();
        let xtest11 = array![[x1 + h, x2]];
        let xtest12 = array![[x1 - h, x2]];
        let xtest21 = array![[x1, x2 + h]];
        let xtest22 = array![[x1, x2 - h]];
        let fdiff1 = (bgp.predict_values(&xtest11.view()).unwrap()
            - bgp.predict_values(&xtest12.view()).unwrap())
            / (2. * h);
        let fdiff2 = (bgp.predict_values(&xtest21.view()).unwrap()
            - bgp.predict_values(&xtest22.view()).unwrap())
            / (2. * h);
        println!(
            "gp fdiff({}) = [[{}, {}]]",
            xtest,
            fdiff1[[0, 0]],
            fdiff2[[0, 0]]
        );
        println!(
            "GP predict derivatives({}) = {}",
            xtest,
            bgp.predict_derivatives(&basetest.view()).unwrap()
        );

        let h = 1e-4;
        let x1 = 5.0;
        let x2 = 1.0;
        let xtest = array![[x1, x2]];
        let basetest = xtest.to_owned();
        let xtest11 = array![[x1 + h, x2]];
        let xtest12 = array![[x1 - h, x2]];
        let xtest21 = array![[x1, x2 + h]];
        let xtest22 = array![[x1, x2 - h]];
        let fdiff1 = (bgp.predict_variances(&xtest11.view()).unwrap()
            - bgp.predict_variances(&xtest12.view()).unwrap())
            / (2. * h);
        let fdiff2 = (bgp.predict_variances(&xtest21.view()).unwrap()
            - bgp.predict_variances(&xtest22.view()).unwrap())
            / (2. * h);
        println!(
            "gp var fdiff({}) = [[{}, {}]]",
            xtest,
            fdiff1[[0, 0]],
            fdiff2[[0, 0]]
        );
        println!(
            "GP predict variances derivatives({}) = {}",
            xtest,
            bgp.predict_variance_derivatives(&basetest.view()).unwrap()
        );
    }
}
