use egobox_ego::EgorBuilder;
use ndarray::{Array2, ArrayView2, array};

fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
    (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
}

fn main() {
    let res = EgorBuilder::optimize(xsinx)
        .configure(|config| config.max_iters(20))
        .min_within(&array![[0., 25.]])
        .expect("Egor configured")
        .run()
        .expect("Minimization of xsinx");
    println!("Minimum xsinx(x) = {} at x = {}", res.y_opt, res.x_opt);
}
