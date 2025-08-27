use egobox_gp::Kriging;
use linfa::prelude::*;
use ndarray::{Array, Axis, arr1, arr2};

fn main() {
    let xtrain = arr2(&[[0.0], [1.0], [2.0], [3.0], [4.0]]);
    let ytrain = arr1(&[0.0, 1.0, 1.5, 0.9, 1.0]);

    let kriging = Kriging::params()
        .fit(&Dataset::new(xtrain, ytrain))
        .expect("Kriging fitting");

    let xtest = Array::linspace(0., 4., 100).insert_axis(Axis(1));
    let _ytest = kriging.predict(&xtest).expect("Kriging prediction");
}
