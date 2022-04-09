use egobox_doe::{Lhs, SamplingMethod};
use egobox_moe::{Moe, MoePredict, Recombination};
use ndarray::{arr2, Array, Array2, Axis, Zip};
use std::error::Error;

fn f3parts(x: &Array2<f64>) -> Array2<f64> {
    let mut y = Array2::zeros(x.dim());
    Zip::from(&mut y).and(x).for_each(|yi, &xi| {
        if xi < 0.4 {
            *yi = xi * xi;
        } else if (0.4..0.8).contains(&xi) {
            *yi = 3. * xi + 1.;
        } else {
            *yi = f64::sin(10. * xi);
        }
    });
    y
}

fn main() -> Result<(), Box<dyn Error>> {
    let xtrain = Lhs::new(&arr2(&[[0., 1.]])).sample(50);
    let ytrain = f3parts(&xtrain);
    let moe1 = Moe::params(1).fit(&xtrain, &ytrain)?;
    let moe3 = Moe::params(3)
        .set_recombination(Recombination::Hard)
        .fit(&xtrain, &ytrain)?;

    let xtest = Array::linspace(0., 1., 101).insert_axis(Axis(1));
    let ytest = f3parts(&xtest);

    let ymoe1 = moe1.predict_values(&xtest)?;
    let ymoe3 = moe3.predict_values(&xtest)?;

    println!("Compute average prediction error");
    println!(
        "MoE with 1 cluster  = {}",
        (ymoe1 - &ytest)
            .map(|v| v.abs())
            .mean_axis(Axis(0))
            .unwrap()
    );
    println!(
        "MoE with 3 clusters = {}",
        (ymoe3 - &ytest)
            .map(|v| v.abs())
            .mean_axis(Axis(0))
            .unwrap()
    );

    Ok(())
}
