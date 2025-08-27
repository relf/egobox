use egobox_doe::{Lhs, SamplingMethod};
use egobox_moe::{GpMixture, NbClusters, Recombination};
use linfa::prelude::{Dataset, Fit};
use ndarray::{Array, Array2, Axis, Zip, arr2};
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
    let ds = Dataset::new(xtrain, ytrain.remove_axis(Axis(1)));
    let moe1 = GpMixture::params().fit(&ds)?;
    let moe3 = GpMixture::params()
        .n_clusters(NbClusters::fixed(3))
        .recombination(Recombination::Hard)
        .fit(&ds)?;

    let xtest = Array::linspace(0., 1., 101).insert_axis(Axis(1));
    let ytest = f3parts(&xtest);

    let ymoe1 = moe1.predict(&xtest)?;
    let ymoe3 = moe3.predict(&xtest)?;

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
