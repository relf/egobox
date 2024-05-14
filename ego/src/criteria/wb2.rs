use crate::criteria::{ei::EI, InfillCriterion};

use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, ArrayView, ArrayView2, Axis};
use ndarray_stats::QuantileExt;

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct WB2Criterion(pub Option<f64>);

#[typetag::serde]
impl InfillCriterion for WB2Criterion {
    fn name(&self) -> &'static str {
        if self.0.is_some() {
            "WB2S"
        } else {
            "WB2"
        }
    }

    /// Compute WBS2 infill criterion at given `x` point using the surrogate model `obj_model`
    /// and the current minimum of the objective function.
    fn value(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        f_min: f64,
        scale: Option<f64>,
    ) -> f64 {
        let scale = scale.unwrap_or(1.0);
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        let ei = EI.value(x, obj_model, f_min, None);
        scale * ei - obj_model.predict(&pt).unwrap()[[0, 0]]
    }

    /// Computes derivatives of WS2 infill criterion wrt to x components at given `x` point
    /// using the surrogate model `obj_model` and the current minimum of the objective function.
    fn grad(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        f_min: f64,
        scale: Option<f64>,
    ) -> Array1<f64> {
        let scale = scale.unwrap_or(1.0);
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        let grad_ei = EI.grad(x, obj_model, f_min, None) * scale;
        grad_ei - obj_model.predict_gradients(&pt).unwrap().row(0)
    }

    fn scaling(
        &self,
        x: &ndarray::ArrayView2<f64>,
        obj_model: &dyn MixtureGpSurrogate,
        f_min: f64,
    ) -> f64 {
        if let Some(scale) = self.0 {
            scale
        } else {
            compute_wb2s_scale(x, obj_model, f_min)
        }
    }
}

/// Computes the scaling factor used to scale WB2 infill criteria.
pub(crate) fn compute_wb2s_scale(
    x: &ArrayView2<f64>,
    obj_model: &dyn MixtureGpSurrogate,
    f_min: f64,
) -> f64 {
    let ratio = 100.; // TODO: make it a parameter
    let ei_x = x.map_axis(Axis(1), |xi| {
        let xi = xi.as_standard_layout();
        let ei = EI.value(xi.as_slice().unwrap(), obj_model, f_min, None);
        ei
    });
    let i_max = ei_x.argmax().unwrap();
    let pred_max = obj_model
        .predict(&x.row(i_max).insert_axis(Axis(0)))
        .unwrap()[[0, 0]];
    let ei_max = ei_x[i_max];
    if ei_max.abs() > 100. * f64::EPSILON {
        ratio * pred_max / ei_max
    } else {
        1.
    }
}

/// WB2 infill criterion
pub const WB2: WB2Criterion = WB2Criterion(Some(1.0));
/// WB2 scaled infill criterion
pub const WB2S: WB2Criterion = WB2Criterion(None);

#[cfg(test)]
mod tests {
    use super::*;
    use egobox_doe::SamplingMethod;
    use egobox_moe::*;
    use linfa::prelude::*;
    use ndarray::{array, Array2};

    fn sphere(x: &Array2<f64>) -> Array2<f64> {
        let s = (x * x).sum_axis(Axis(1));
        s.insert_axis(Axis(1))
    }

    #[test]
    fn test_grad_wbs2() {
        let xt = egobox_doe::Lhs::new(&array![[-10., 10.], [-10., 10.]]).sample(10);
        let yt = sphere(&xt);
        let gp = GpMixture::params()
            .regression_spec(RegressionSpec::CONSTANT)
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
            .recombination(Recombination::Hard)
            .fit(&Dataset::new(xt, yt))
            .expect("GP fitting");
        let bgp = Box::new(gp) as Box<dyn MixtureGpSurrogate>;

        let h = 1e-4;
        let x1 = 0.1;
        let x2 = 0.3;
        let xtest = vec![x1, x2];
        let xtest11 = vec![x1 + h, x2];
        let xtest12 = vec![x1 - h, x2];
        let xtest21 = vec![x1, x2 + h];
        let xtest22 = vec![x1, x2 - h];
        let fdiff1 = (WB2S.value(&xtest11, bgp.as_ref(), 0.1, Some(0.5))
            - WB2S.value(&xtest12, bgp.as_ref(), 0.1, Some(0.5)))
            / (2. * h);
        let fdiff2 = (WB2S.value(&xtest21, bgp.as_ref(), 0.1, Some(0.5))
            - WB2S.value(&xtest22, bgp.as_ref(), 0.1, Some(0.5)))
            / (2. * h);
        println!("fdiff({xtest:?}) = [{fdiff1}, {fdiff2}]");
        println!(
            "grad_wbs2({:?}) = {:?}",
            xtest,
            WB2S.grad(&xtest21, bgp.as_ref(), 0.1, Some(0.5))
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
        let fdiff1 = (bgp.predict(&xtest11.view()).unwrap()
            - bgp.predict(&xtest12.view()).unwrap())
            / (2. * h);
        let fdiff2 = (bgp.predict(&xtest21.view()).unwrap()
            - bgp.predict(&xtest22.view()).unwrap())
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
            bgp.predict_gradients(&basetest.view()).unwrap()
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
        let fdiff1 = (bgp.predict_var(&xtest11.view()).unwrap()
            - bgp.predict_var(&xtest12.view()).unwrap())
            / (2. * h);
        let fdiff2 = (bgp.predict_var(&xtest21.view()).unwrap()
            - bgp.predict_var(&xtest22.view()).unwrap())
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
            bgp.predict_var_gradients(&basetest.view()).unwrap()
        );
    }
}
