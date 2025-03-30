use approx::assert_abs_diff_eq;
use argmin_testfunctions::rosenbrock;
use egobox_doe::Lhs;
use egobox_doe::SamplingMethod;
use egobox_ego::{EgorBuilder, DOE_FILE};
use ndarray::{array, Array1, Array2, ArrayView2, Zip};
use ndarray_npy::read_npy;
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

/// Rosenbrock test function: min f(x)=0 at x=(1, 1)
fn rosenb(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec())]));
    y
}

fn main() {
    let outdir = "target/test_coego";
    let dim = 50;
    let xlimits = Array2::from_shape_vec((dim, 2), [-2.0, 2.0].repeat(dim)).unwrap();
    let init_doe = Lhs::new(&xlimits)
        .with_rng(Xoshiro256Plus::seed_from_u64(42))
        .sample(100);
    let max_iters = 100;
    let res = EgorBuilder::optimize(rosenb)
        .configure(|config| {
            config
                .doe(&init_doe)
                .max_iters(max_iters)
                .outdir(outdir)
                .seed(42)
                .coego(true)
                .trego(true)
        })
        .min_within(&xlimits)
        .run()
        .expect("Minimize failure");
    let filepath = std::path::Path::new(&outdir).join(DOE_FILE);
    assert!(filepath.exists());
    let doe: Array2<f64> = read_npy(&filepath).expect("file read");
    assert!(doe.nrows() <= init_doe.nrows() + max_iters);
    assert!(doe.nrows() >= init_doe.nrows());

    println!("Rosenbrock optim result = {res:?}");
    let expected = Array1::<f64>::ones(dim);
    assert_abs_diff_eq!(expected, res.x_opt, epsilon = 5e-1);
}
