use egobox_ego::{EgorBuilder, InfillOptimizer, InfillStrategy};
use egobox_moe::{CorrelationSpec, RegressionSpec};
use ndarray::{Array, Array2, ArrayView2, Zip, array};

/// Ackley test function: min f(x)=0 at x=(0, 0, 0)
fn ackley(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .par_for_each(|mut yi, xi| yi.assign(&array![argmin_testfunctions::ackley(&xi.to_vec(),)]));
    y
}

fn main() {
    let ndim = 16;
    let data = [-32.768, 32.768].repeat(16);
    let xlimits = Array::from_shape_vec((ndim, 2), data).unwrap();

    let res = EgorBuilder::optimize(ackley)
        .configure(|config| {
            config
                .configure_gp(|gp| {
                    gp.regression_spec(RegressionSpec::CONSTANT)
                        .correlation_spec(CorrelationSpec::ABSOLUTEEXPONENTIAL)
                })
                .infill_strategy(InfillStrategy::WB2S)
                .infill_optimizer(InfillOptimizer::Slsqp)
                .n_start(50)
                .max_iters(300)
        })
        .min_within(&xlimits)
        .run()
        .expect("Minimize failure");
    println!("Ackley minimum y = {} at x = {}", res.y_opt, res.x_opt);
}
