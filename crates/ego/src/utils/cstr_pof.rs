use std::iter::zip;

use crate::utils::{norm_cdf, norm_pdf};
use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, ArrayView};

/// Compute probability of feasibility using the given surrogate of
/// a constraint function wrt the tolerance (ie cstr <= cstr_tol)
pub fn pof(x: &[f64], cstr_model: &dyn MixtureGpSurrogate, cstr_tol: f64) -> f64 {
    let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
    if let Ok(p) = cstr_model.predict(&pt) {
        if let Ok(s) = cstr_model.predict_var(&pt) {
            if s[[0, 0]].abs() < 10000. * f64::EPSILON {
                0.0
            } else {
                let pred = p[0];
                let sigma = s[[0, 0]].sqrt();
                let args0 = (cstr_tol - pred) / sigma;
                norm_cdf(args0)
            }
        } else {
            0.0
        }
    } else {
        0.0
    }
}

/// Computes the derivative of the probability of feasibility of the given
/// constraint surrogate model.
pub fn pof_grad(x: &[f64], cstr_model: &dyn MixtureGpSurrogate, cstr_tol: f64) -> Array1<f64> {
    let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
    if let Ok(p) = cstr_model.predict(&pt) {
        if let Ok(s) = cstr_model.predict_var(&pt) {
            let sigma = s[[0, 0]].sqrt();
            if sigma.abs() < 10000. * f64::EPSILON {
                Array1::zeros(pt.len())
            } else {
                let pred = p[0];
                let arg = (cstr_tol - pred) / sigma;
                let y_prime = cstr_model.predict_gradients(&pt).unwrap();
                let y_prime = y_prime.row(0);
                let sig_2_prime = cstr_model.predict_var_gradients(&pt).unwrap();
                let sig_2_prime = sig_2_prime.row(0);
                let sig_prime = sig_2_prime.mapv(|v| v / (2. * sigma));
                let arg_prime =
                    y_prime.mapv(|v| v / (-sigma)) + sig_prime.mapv(|v| v * pred / (sigma * sigma));
                norm_pdf(arg) * arg_prime.to_owned()
            }
        } else {
            Array1::zeros(pt.len())
        }
    } else {
        Array1::zeros(pt.len())
    }
}

pub fn pofs(x: &[f64], cstr_models: &[Box<dyn MixtureGpSurrogate>], cstr_tols: &[f64]) -> f64 {
    zip(cstr_models, cstr_tols).fold(1., |acc, (cstr_model, cstr_tol)| {
        let pof = pof(x, &**cstr_model, *cstr_tol);
        acc * pof
    })
}

pub fn pofs_grad(
    x: &[f64],
    cstr_models: &[Box<dyn MixtureGpSurrogate>],
    cstr_tols: &[f64],
) -> Array1<f64> {
    zip(cstr_models, cstr_tols).fold(Array1::ones(x.len()), |acc, (cstr_model, cstr_tol)| {
        let pof = pof_grad(x, &**cstr_model, *cstr_tol);
        acc * pof
    })
}

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
    fn test_pof_gradients() {
        let xtypes = vec![XType::Cont(0., 25.)];

        let mixi = MixintContext::new(&xtypes);

        let surrogate_builder = MoeBuilder::new();
        let xt = array![[0.], [2.], [5.], [10.], [25.]];
        let yt = array![0., 0.2, -0.3, 0.5, -1.];
        let ds = Dataset::new(xt, yt);
        let mixi_moe = mixi
            .create_surrogate(&surrogate_builder, &ds)
            .expect("Mixint surrogate creation");

        let x = vec![0.3];
        let grad = pof_grad(&x, &mixi_moe, 0.);

        let f = |x: &Vec<f64>| -> f64 { pof(x, &mixi_moe, 0.) };
        let grad_central = x.central_diff(&f);

        assert_abs_diff_eq!(grad[0], grad_central[0], epsilon = 1e-6);

        // let cx = Array1::linspace(0., 25., 100);
        // write_npy("cei_cx.npy", &cx).expect("save x");

        // let mut cy = Array1::zeros(cx.len());
        // Zip::from(&mut cy).and(&cx).for_each(|yi, xi| {
        //     *yi = CEI.value(&[*xi], &mixi_moe, 0., None);
        // });
        // write_npy("cei_cy.npy", &cy).expect("save y");

        // let mut cdy = Array1::zeros(cx.len());
        // Zip::from(&mut cdy).and(&cx).for_each(|yi, xi| {
        //     *yi = CEI.grad(&[*xi], &mixi_moe, 0., None)[0];
        // });
        // println!("CDY = {}", cdy);
        // write_npy("cei_cdy.npy", &cdy).expect("save dy");

        // let cytrue = mixi_moe
        //     .predict(&cx.insert_axis(Axis(1)).view())
        //     .expect("prediction");
        // write_npy("cei_cytrue.npy", &cytrue).expect("save cstr");

        // println!("thetas = {}", mixi_moe);
    }
}
