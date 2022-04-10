use egobox_ego::{ApproxValue, Egor, InfillStrategy};
use ndarray::{array, Array2, ArrayView2, Zip};

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
    let res = Egor::new(rosenbrock, &xlimits)
        .n_eval(50)
        .expect(Some(ApproxValue {
            value: 0.0,
            tolerance: 1e-2,
        }))
        .minimize()
        .expect("Minimize failure");
    println!("Rosenbrock minimum y = {} at x = {}", res.y_opt, res.x_opt);
}
