extern crate openblas_src;
use egobox::gaussian_process::*;

fn main() {
    use ndarray::{array, Array};
    // use ndarray_npy::write_npy;

    let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
    let yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];
    // write_npy("xtrain.npy", xt).expect("Failed to write .npy file");
    // write_npy("ytrain.npy", yt).expect("Failed to write .npy file");

    let gp = GaussianProcess::<ConstantMean, SquaredExponentialKernel>::params(
        ConstantMean::new(),
        SquaredExponentialKernel::new(),
    )
    .fit(&xt, &yt)
    .expect("GP fit");

    let num = 100;
    let x = Array::linspace(0.0, 4.0, num).into_shape((num, 1)).unwrap();
    let y = gp.predict_values(&x).expect("GP prediction");
    // write_npy("xvalid.npy", x).expect("Failed to write .npy file");
    // write_npy("yvalid.npy", y).expect("Failed to write .npy file");
}
