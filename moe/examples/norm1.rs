use egobox_doe::{Lhs, SamplingMethod};
use egobox_moe::{Moe, Recombination};
use linfa::{traits::Fit, Dataset};
use ndarray::{arr2, Array2, Axis};
use std::error::Error;

fn norm1(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.abs()).sum_axis(Axis(1)).insert_axis(Axis(1))
}

fn main() -> Result<(), Box<dyn Error>> {
    let xtrain = Lhs::new(&arr2(&[[-1., 1.], [-1., 1.]])).sample(200);
    let ytrain = norm1(&xtrain);
    let ds = Dataset::new(xtrain, ytrain);
    let moe1 = Moe::params(1).fit(&ds)?;
    let moe5 = Moe::params(6).recombination(Recombination::Hard).fit(&ds)?;

    let xtest = Lhs::new(&arr2(&[[-1., 1.], [-1., 1.]])).sample(50);
    let ytest = norm1(&xtest);

    let ymoe1 = moe1.predict_values(&xtest)?;
    let ymoe5 = moe5.predict_values(&xtest)?;

    println!("Compute average prediction error");
    println!(
        "MoE with 1 cluster  = {}",
        (ymoe1 - &ytest)
            .map(|v| v.abs())
            .mean_axis(Axis(0))
            .unwrap()
    );
    println!(
        "MoE with 6 clusters = {}",
        (ymoe5 - &ytest)
            .map(|v| v.abs())
            .mean_axis(Axis(0))
            .unwrap()
    );

    Ok(())
}
