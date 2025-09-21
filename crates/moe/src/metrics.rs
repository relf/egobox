use linfa::dataset::Dataset;
use linfa::prelude::Records;
use linfa::{ParamGuard, traits::Fit};
use ndarray::{Array1, Array2};

use crate::GpSurrogate;

/// A trait for cross validation score
pub trait GpScore<ER, P, O>
where
    ER: std::error::Error + From<linfa::error::Error>,
    P: Fit<Array2<f64>, Array1<f64>, ER, Object = O> + ParamGuard,
    O: GpSurrogate,
{
    fn training_data(&self) -> &(Array2<f64>, Array1<f64>);

    fn params(&self) -> P;

    /// Compute quality metric Q2 with kfold cross validation
    fn q2_score(&self, kfold: usize) -> f64 {
        let (xt, yt) = self.training_data();
        let dataset = Dataset::new(xt.to_owned(), yt.to_owned());
        let yt_mean = yt.mean().unwrap();
        // Predictive Residual Sum of Squares
        let mut press = 0.;
        // Total Sum of Squares
        let mut tss = 0.;
        for (train, valid) in dataset.fold(kfold).into_iter() {
            let params = self.params();
            let model: O = params
                .fit(&train)
                .expect("cross-validation: sub model fitted");
            let pred = model.predict(&valid.records().view()).unwrap();
            press += (valid.targets() - pred).mapv(|v| v * v).sum();
            tss += (valid.targets() - yt_mean).mapv(|v| v * v).sum();
        }
        1. - press / tss
    }

    /// Q2 predictive coefficient with Leave-One-Out Cross-Validation
    fn looq2_score(&self) -> f64 {
        self.q2_score(self.training_data().0.nrows())
    }

    /// Predictive variance adequacy
    fn pva_score(&self, kfold: usize) -> f64 {
        let (xt, yt) = self.training_data();
        let dataset = Dataset::new(xt.to_owned(), yt.to_owned());
        // Total Sum of Squares
        let mut varss = 0.;
        // Number of fold
        let mut n = 0usize;
        for (train, valid) in dataset.fold(kfold).into_iter() {
            let params = self.params();
            let model: O = params
                .fit(&train)
                .expect("cross-validation: sub model fitted");
            let pred = model.predict(&valid.records().view()).unwrap();
            let var = model.predict_var(&valid.records().view()).unwrap();
            varss += ((valid.targets() - &pred).mapv(|v| v * v) / var).sum();
            n += valid.nsamples();
        }
        (varss / n as f64).ln().abs()
    }

    /// Q2 predictive coefficient with Leave-One-Out Cross-Validation
    fn loopva_score(&self) -> f64 {
        self.pva_score(self.training_data().0.nrows())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::GpMixtureParams;
    use approx::assert_abs_diff_eq;
    use egobox_doe::{Lhs, SamplingMethod};
    use ndarray::{Array1, array};
    use ndarray_rand::rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    fn x_squared(x: &Array2<f64>) -> Array1<f64> {
        x.mapv(|v| v * v).sum_axis(ndarray::Axis(1))
    }

    #[test]
    fn test_gpqa_griewank() {
        let dims = [2];
        let nts = [20];
        let lim = array![[-10., 10.]];

        (0..dims.len()).for_each(|i| {
            let dim = dims[i];
            let nt = nts[i];
            let xlimits = lim.broadcast((dim, 2)).unwrap();

            let rng = Xoshiro256Plus::seed_from_u64(42);
            let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
            let yt = x_squared(&xt);

            let moe = GpMixtureParams::default()
                .fit(&Dataset::new(xt, yt))
                .expect("GP fit error");

            assert_abs_diff_eq!(moe.q2_score(10), 1., epsilon = 1e-3);
            assert_abs_diff_eq!(moe.looq2_score(), 1., epsilon = 1e-3);
            assert_abs_diff_eq!(moe.pva_score(10), 0., epsilon = 2e-1);
            assert_abs_diff_eq!(moe.loopva_score(), 0., epsilon = 2e-1);
        });
    }
}
