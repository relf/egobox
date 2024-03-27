use egobox_gp::{correlation_models::*, mean_models::*, GaussianProcess};
use linfa::prelude::*;
use ndarray::{arr2, concatenate, Array, Array2, Axis};

fn xsinx(x: &Array2<f64>) -> Array2<f64> {
    (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
}

fn main() {
    let xt = arr2(&[[0.0], [5.0], [10.0], [15.0], [18.0], [20.0], [25.0]]);
    let yt = xsinx(&xt);

    println!("Train kriging surrogate of 'xsinx' at {}", xt.column(0));
    let kriging = GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
        ConstantMean::default(),
        SquaredExponentialCorr::default(),
    )
    .fit(&Dataset::new(xt, yt))
    .expect("GP fitting");

    let xtest = Array::linspace(0., 25., 26).insert_axis(Axis(1));
    let ytest = xsinx(&xtest);
    // predict values
    let ypred = kriging.predict(&xtest).expect("Kriging prediction");
    // predict standard deviation
    let ysigma = kriging
        .predict_var(&xtest)
        .expect("Kriging prediction")
        .map(|v| v.sqrt());

    println!("Compute prediction errors (x, err(x))");
    println!("{}", concatenate![Axis(1), xtest, (ypred - ytest), ysigma]);
}
