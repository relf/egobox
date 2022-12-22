use egobox_ego::{ApproxValue, EgorBuilder, InfillStrategy};
use egobox_moe::{CorrelationSpec, RegressionSpec};
use ndarray::{array, Array2, ArrayView2, Zip};

fn ackley(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .par_for_each(|mut yi, xi| yi.assign(&array![argmin_testfunctions::ackley(&xi.to_vec(),)]));
    y
}

fn main() {
    let xlimits = array![[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]];
    let res = EgorBuilder::optimize(ackley)
        .min_within(&xlimits)
        .regression_spec(RegressionSpec::CONSTANT)
        .correlation_spec(CorrelationSpec::ABSOLUTEEXPONENTIAL)
        .infill_strategy(InfillStrategy::WB2S)
        .n_eval(200)
        .expect(Some(ApproxValue {
            value: 0.0,
            tolerance: 5e-1,
        }))
        .run()
        .expect("Minimize failure");
    println!("Ackley minimum y = {} at x = {}", res.y_opt, res.x_opt);
}
