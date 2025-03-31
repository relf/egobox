use argmin_testfunctions::rosenbrock;
use egobox_doe::Lhs;
use egobox_doe::SamplingMethod;
use egobox_ego::EgorBuilder;
use ndarray::{array, Array2, ArrayView2, Zip};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

/// Rosenbrock test function: min f(x)=0 at x=(1, ..., 1)
fn rosenb(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec())]));
    y
}

fn main() {
    let outdir = "target/test_rosenbrock_hidim";
    let dim = 20;
    let xlimits = Array2::from_shape_vec((dim, 2), [-2.0, 2.0].repeat(dim)).unwrap();
    let init_doe = Lhs::new(&xlimits)
        .with_rng(Xoshiro256Plus::seed_from_u64(42))
        .sample(100);
    let max_iters = 200;
    let res = EgorBuilder::optimize(rosenb)
        .configure(|config| {
            config
                .doe(&init_doe)
                .max_iters(max_iters)
                .outdir(outdir)
                .seed(42)
            // .coego(true)
            // .trego(true)
        })
        .min_within(&xlimits)
        .run()
        .expect("Minimize failure");

    println!(
        "Rosenbrock optim result f(x) = {} at x = {}",
        res.y_opt, res.x_opt
    );
}
