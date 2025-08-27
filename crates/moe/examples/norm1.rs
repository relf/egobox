use egobox_doe::{Lhs, SamplingMethod};
use egobox_moe::{GpMixture, NbClusters, Recombination};
use linfa::{Dataset, traits::Fit};
use ndarray::{Array2, Axis, arr2};
use std::error::Error;

fn norm1(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.abs()).sum_axis(Axis(1)).insert_axis(Axis(1))
}

fn main() -> Result<(), Box<dyn Error>> {
    let xtrain = Lhs::new(&arr2(&[[-1., 1.], [-1., 1.]])).sample(200);
    let ytrain = norm1(&xtrain);
    let ds = Dataset::new(xtrain, ytrain.remove_axis(Axis(1)));
    let moe1 = GpMixture::params().fit(&ds)?;
    let moe5 = GpMixture::params()
        .n_clusters(NbClusters::fixed(6))
        .recombination(Recombination::Hard)
        .fit(&ds)?;

    let xtest = Lhs::new(&arr2(&[[-1., 1.], [-1., 1.]])).sample(50);
    let ytest = norm1(&xtest);

    let ymoe1 = moe1.predict(&xtest)?;
    let ymoe5 = moe5.predict(&xtest)?;

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
