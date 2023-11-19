use egobox_ego::{EgorBuilder, InfillStrategy};
use egobox_moe::{CorrelationSpec, RegressionSpec};
use ndarray::{array, Array2, ArrayView2, Zip};

/// Ackley test function: min f(x)=0 at x=(0, 0, 0)
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
        .configure(|config| {
            config
                .regression_spec(RegressionSpec::CONSTANT)
                .correlation_spec(CorrelationSpec::ABSOLUTEEXPONENTIAL)
                .infill_strategy(InfillStrategy::WB2S)
                .max_iters(200)
                .target(5e-1)
        })
        .min_within(&xlimits)
        .run()
        .expect("Minimize failure");
    println!("Ackley minimum y = {} at x = {}", res.y_opt, res.x_opt);
}
