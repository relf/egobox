use crate::criteria::InfillCriterion;
use crate::utils::{norm_cdf, norm_pdf};
use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, ArrayView};

use serde::{Deserialize, Serialize};

const SQRT_2PI: f64 = 2.5066282746310007;

#[derive(Clone, Serialize, Deserialize)]
pub struct ExpectedImprovement;

#[typetag::serde]
impl InfillCriterion for ExpectedImprovement {
    fn name(&self) -> &'static str {
        "EI"
    }

    /// Compute EI infill criterion at given `x` point using the surrogate model `obj_model`
    /// and the current minimum of the objective function.
    fn value(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        _scale: Option<f64>,
    ) -> f64 {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        if let Ok(p) = obj_model.predict(&pt) {
            if let Ok(s) = obj_model.predict_var(&pt) {
                if s[[0, 0]].abs() < 10000. * f64::EPSILON {
                    0.0
                } else {
                    let pred = p[0];
                    let sigma = s[[0, 0]].sqrt();
                    let args0 = (fmin - pred) / sigma;
                    let args1 = (fmin - pred) * norm_cdf(args0);
                    let args2 = sigma * norm_pdf(args0);
                    args1 + args2
                }
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Computes derivatives of EI infill criterion wrt to x components at given `x` point
    /// using the surrogate model `obj_model` and the current minimum of the objective function.
    fn grad(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        _scale: Option<f64>,
    ) -> Array1<f64> {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        if let Ok(p) = obj_model.predict(&pt) {
            if let Ok(s) = obj_model.predict_var(&pt) {
                let sigma = s[[0, 0]].sqrt();
                if sigma.abs() < 10000. * f64::EPSILON {
                    Array1::zeros(pt.len())
                } else {
                    let pred = p[0];
                    let diff_y = fmin - pred;
                    let arg = (fmin - pred) / sigma;
                    let y_prime = obj_model.predict_gradients(&pt).unwrap();
                    let y_prime = y_prime.row(0);
                    let sig_2_prime = obj_model.predict_var_gradients(&pt).unwrap();

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

    fn scaling(
        &self,
        _x: &ndarray::ArrayView2<f64>,
        _obj_model: &dyn MixtureGpSurrogate,
        _fmin: f64,
    ) -> f64 {
        1.0
    }
}

/// Expected Improvement infill criterion
pub const EI: ExpectedImprovement = ExpectedImprovement {};

#[derive(Clone, Serialize, Deserialize)]
pub struct LogExpectedImprovement;

#[typetag::serde]
impl InfillCriterion for LogExpectedImprovement {
    fn name(&self) -> &'static str {
        "LogEI"
    }

    /// Compute LogEI infill criterion at given `x` point using the surrogate model `obj_model`
    /// and the current minimum of the objective function.
    fn value(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        _scale: Option<f64>,
    ) -> f64 {
        let ei = EI.value(x, obj_model, fmin, _scale);
        (ei + 100. * f64::EPSILON).ln()
    }

    /// Computes derivatives of EI infill criterion wrt to x components at given `x` point
    /// using the surrogate model `obj_model` and the current minimum of the objective function.
    fn grad(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        _scale: Option<f64>,
    ) -> Array1<f64> {
        let ei = EI.value(x, obj_model, fmin, _scale);

        let grad_ei = EI.grad(x, obj_model, fmin, _scale);
        grad_ei / (ei + 100. * f64::EPSILON)
    }

    fn scaling(
        &self,
        _x: &ndarray::ArrayView2<f64>,
        _obj_model: &dyn MixtureGpSurrogate,
        _fmin: f64,
    ) -> f64 {
        1.0
    }
}

/// Log of Expected Improvement infill criterion
pub const LOG_EI: LogExpectedImprovement = LogExpectedImprovement {};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gpmix::mixint::{MixintContext, MoeBuilder},
        types::*,
    };
    use approx::assert_abs_diff_eq;
    // use egobox_moe::GpSurrogate;
    use finitediff::FiniteDiff;
    use linfa::Dataset;
    use ndarray::array;
    // use ndarray_npy::write_npy;

    #[test]
    fn test_ei_gradients() {
        let xtypes = vec![XType::Float(0., 25.)];

        let mixi = MixintContext::new(&xtypes);

        let surrogate_builder = MoeBuilder::new();
        let xt = array![[0.], [2.], [5.], [10.], [25.]];
        let yt = array![0., 0.2, -0.3, 0.5, -1.];
        let ds = Dataset::new(xt, yt);
        let mixi_moe = mixi
            .create_surrogate(&surrogate_builder, &ds)
            .expect("Mixint surrogate creation");

        let x = vec![3.];
        let grad = EI.grad(&x, &mixi_moe, 0., None);

        let f = |x: &Vec<f64>| -> f64 { EI.value(x, &mixi_moe, 0., None) };
        let grad_central = x.central_diff(&f);
        assert_abs_diff_eq!(grad[0], grad_central[0], epsilon = 1e-6);

        // let cx = Array1::linspace(0., 25., 100);
        // write_npy("ei_cx.npy", &cx).expect("save x");

        // let mut cy = Array1::zeros(cx.len());
        // Zip::from(&mut cy).and(&cx).for_each(|yi, xi| {
        //     *yi = EI.value(&[*xi], &mixi_moe, 0., None);
        // });
        // write_npy("ei_cy.npy", &cy).expect("save y");

        // let mut cdy = Array1::zeros(cx.len());
        // Zip::from(&mut cdy).and(&cx).for_each(|yi, xi| {
        //     *yi = EI.grad(&[*xi], &mixi_moe, 0., None)[0];
        // });
        // write_npy("ei_cdy.npy", &cdy).expect("save y");

        // let cytrue = mixi_moe
        //     .predict(&cx.insert_axis(Axis(1)).view())
        //     .expect("prediction");
        // write_npy("ei_cytrue.npy", &cytrue).expect("save cstr");

        // println!("thetas = {}", mixi_moe);
    }
}
