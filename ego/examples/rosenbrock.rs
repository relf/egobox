use egobox_ego::EgorBuilder;
use ndarray::{array, Array2, ArrayView2, Zip};

/// Rosenbrock test function: min f(x)=0 at x=(1, 1)
fn rosenbrock(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .par_for_each(|mut yi, xi| {
            yi.assign(&array![argmin_testfunctions::rosenbrock(
                &xi.to_vec(),
                1.,
                100.
            )])
        });
    y
}

fn main() {
    let xlimits = array![[-2., 2.], [-2., 2.]];
    let res = EgorBuilder::optimize(rosenbrock)
        .configure(|config| config.max_iters(100).target(1e-2))
        .min_within(&xlimits)
        .run()
        .expect("Minimize failure");
    println!("Rosenbrock minimum y = {} at x = {}", res.y_opt, res.x_opt);
}
