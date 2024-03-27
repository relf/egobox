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
    /// Compute EI infill criterion at given `x` point using the surrogate model `obj_model`
    /// and the current minimum of the objective function.
    fn value(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        f_min: f64,
        _scale: Option<f64>,
    ) -> f64 {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        if let Ok(p) = obj_model.predict(&pt) {
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

    /// Computes derivatives of EI infill criterion wrt to x components at given `x` point
    /// using the surrogate model `obj_model` and the current minimum of the objective function.
    fn grad(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        f_min: f64,
        _scale: Option<f64>,
    ) -> Array1<f64> {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        if let Ok(p) = obj_model.predict(&pt) {
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

    fn scaling(
        &self,
        _x: &ndarray::ArrayView2<f64>,
        _obj_model: &dyn MixtureGpSurrogate,
        _f_min: f64,
    ) -> f64 {
        1.0
    }
}

/// Expected Improvement infill criterion
pub const EI: ExpectedImprovement = ExpectedImprovement {};
