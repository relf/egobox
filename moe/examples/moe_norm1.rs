use csv::ReaderBuilder;
use doe::{FullFactorial, SamplingMethod};
use moe::Moe;
use ndarray::{arr2, s, Array2, Axis};
use ndarray_csv::Array2Reader;
use ndarray_npy::write_npy;
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    fn norm1(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.abs()).sum_axis(Axis(1)).insert_axis(Axis(1))
    }

    let file = File::open("D:/rlafage/workspace/egobox/moe/examples/norm1_D2_200.csv")?;
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .from_reader(file);

    let data_train: Array2<f64> = reader.deserialize_array2((200, 3))?;

    let xtrain = data_train.slice(s![.., ..2]);
    let ytrain = data_train.slice(s![.., 2..]);
    let moe = Moe::params(4).fit(&xtrain, &ytrain)?;

    let xlimits = arr2(&[[-1., 1.], [-1., 1.]]);
    let xtest = FullFactorial::new(&xlimits).sample(100);
    let ytest = moe.predict_values(&xtest)?;
    let ytrue = norm1(&xtest);

    write_npy("moe_x_norm1.npy", &xtest).expect("x not saved!");
    write_npy("moe_ypred_norm1.npy", &ytest).expect("ypred not saved!");
    write_npy("moe_ytrue_norm1.npy", &ytrue).expect("ytrue not saved!");

    Ok(())
}
