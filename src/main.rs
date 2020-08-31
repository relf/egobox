extern crate ndarray;
extern crate ndarray_linalg;
extern crate openblas_src; 
use crate::kriging;

fn main() {
    use ndarray::array;
    let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
    let yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];

    println!("{:?}", kriging::utils::normalize(&xt));
    // println!("{:?}", kriging::utils::l1_cross_distances(&xt));

    // let gp = GaussianProcess::default(xt, yt);

    // let xp = vec![1.5];
    // let yp = gp.predict(&xp);

    // println!("prediction: {}", yp);
}
