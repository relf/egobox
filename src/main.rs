fn main() {
    use ndarray::array;
    let mut xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
    let mut yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];

    // println!("{:?}", kriging::utils::normalize(&mut xt));
    println!("{:?}", kriging::utils::l1_cross_distances(&xt));

    // let gp = GaussianProcess::default(xt, yt);

    // let xp = vec![1.5];
    // let yp = gp.predict(&xp);

    // println!("prediction: {}", yp);
}
