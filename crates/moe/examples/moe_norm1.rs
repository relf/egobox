use csv::ReaderBuilder;
use egobox_doe::{FullFactorial, SamplingMethod};
use egobox_moe::{GpMixture, NbClusters};
use linfa::{Dataset, traits::Fit};
use ndarray::{Array2, Axis, arr2, s};
use ndarray_csv::Array2Reader;
use ndarray_npy::write_npy;
use std::error::Error;
use std::fs::File;

fn norm1(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.abs()).sum_axis(Axis(1)).insert_axis(Axis(1))
}

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("examples/norm1_D2_200.csv")?;
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .from_reader(file);

    let data_train: Array2<f64> = reader.deserialize_array2((200, 3))?;

    let xtrain = data_train.slice(s![.., ..2_usize]).to_owned();
    let ytrain = data_train.slice(s![.., 2_usize..]).to_owned();
    let ds = Dataset::new(xtrain, ytrain.remove_axis(Axis(1)));
    let moe = GpMixture::params()
        .n_clusters(NbClusters::fixed(4))
        .fit(&ds)?;

    let xlimits = arr2(&[[-1., 1.], [-1., 1.]]);
    let xtest = FullFactorial::new(&xlimits).sample(100);
    let ytest = moe.predict(&xtest)?;
    let ytrue = norm1(&xtest);

    // Save data as numpy arrays to plot with Python
    let example_dir = "target/examples";
    std::fs::create_dir_all(example_dir).ok();

    write_npy(format!("{example_dir}/moe_x_norm1.npy"), &xtest).expect("x not saved!");
    write_npy(format!("{example_dir}/moe_ypred_norm1.npy"), &ytest).expect("ypred not saved!");
    write_npy(format!("{example_dir}/moe_ytrue_norm1.npy"), &ytrue).expect("ytrue not saved!");

    Ok(())
}
