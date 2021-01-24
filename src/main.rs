// extern crate openblas_src;
extern crate intel_mkl_src;
use egobox::doe::{FullFactorial, LHSKind, LHS};
use egobox::gaussian_process::*;
use ndarray::{arr2, array, Array1, Array2, Zip};
use ndarray_npy::{read_npy, write_npy};
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use std::time::Instant;

fn main() {
    // use ndarray_npy::write_npy;

    let dims = vec![5, 10, 20, 60];
    let nts = vec![100, 300, 400, 800];

    // for i in 3..dims.len() {
    // for i in 0..dims.len() {
    for i in 3..4 {
        let dim = dims[i];
        let nt = nts[i];

        let d = Array1::linspace(1., dim as f64, dim).mapv(|v| v.sqrt());
        let griewak = |x: &Array1<f64>| -> f64 {
            x.mapv(|v| v * v).sum() / 4000. - (x / &d).mapv(|v| v.cos()).fold(1., |acc, x| acc * x)
                + 1.0
        };
        let prefix = "gp";
        let xfilename = format!("{}_xt_{}x{}.npy", prefix, nt, dim);
        let yfilename = format!("{}_yt_{}x{}.npy", prefix, nt, 1);

        let xt = match read_npy(&xfilename) {
            Ok(xt) => xt,
            Err(_) => {
                let lim = array![[-600., 600.]];
                let xlimits = lim.broadcast((dim, 2)).unwrap();
                let rng = Isaac64Rng::seed_from_u64(42);
                let xt = LHS::new(&xlimits).with_rng(rng).sample(nt);
                write_npy(&xfilename, xt.to_owned()).expect("cannot save xt");
                xt
            }
        };

        let yt = match read_npy(&yfilename) {
            Ok(yt) => yt,
            Err(_) => {
                let mut yv: Array1<f64> = Array1::zeros(xt.nrows());
                Zip::from(&mut yv).and(xt.genrows()).par_apply(|y, x| {
                    *y = griewak(&x.to_owned());
                });
                let yt = yv.into_shape((xt.nrows(), 1)).unwrap();
                write_npy(&yfilename, yt.to_owned()).expect("cannot save yt");
                yt
            }
        };
        let start2 = Instant::now();
        let gp = GaussianProcess::<ConstantMean, SquaredExponentialKernel>::params(
            ConstantMean::default(),
            SquaredExponentialKernel::default(),
        )
        //.with_kpls_dim(1)
        //.with_initial_theta(1.0)
        .fit(&xt, &yt)
        .expect("GP fit error");
        println!("Time fitting is: {:?}", start2.elapsed());

        let xtest = Array2::zeros((1, dim));
        let ytest = gp.predict_values(&xtest).expect("prediction error");
        println!("ytest = {}", ytest);
    }
}
