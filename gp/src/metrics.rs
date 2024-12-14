use linfa::dataset::Dataset;
use linfa::{
    traits::{Fit, Predict, PredictInplace},
    Float, ParamGuard,
};
use ndarray::{Array1, Array2, ArrayBase, Ix2, OwnedRepr};

use crate::{
    correlation_models, mean_models, GaussianProcess, GpError, GpParams, SgpParams,
    SparseGaussianProcess,
};

/// A trait for cross validation score
pub trait CrossValScore<F, ER, P, O>
where
    F: Float,
    ER: std::error::Error + From<linfa::error::Error>,
    P: Fit<Array2<F>, Array1<F>, ER, Object = O> + ParamGuard,
    O: PredictInplace<ArrayBase<OwnedRepr<F>, Ix2>, Array1<F>>,
{
    fn training_data(&self) -> &(Array2<F>, Array1<F>);

    fn params(&self) -> P;

    /// Compute quality metric based on cross validation
    fn cv_score(&self, fold: usize) -> F {
        let (xt, yt) = self.training_data();
        let dataset = Dataset::new(xt.to_owned(), yt.to_owned());
        let mut error = F::zero();
        for (train, valid) in dataset.fold(fold).into_iter() {
            let params = self.params();
            let model: O = params
                .fit(&train)
                .expect("cross-validation: sub model fitted");
            let pred = model.predict(valid.records());
            error += (valid.targets() - pred).mapv(|v| v * v).sum();
        }
        (error / F::cast(fold)).sqrt() / yt.mean().unwrap()
    }

    /// Leave one out cross validation
    fn loocv_score(&self) -> F {
        self.cv_score(self.training_data().0.nrows())
    }
}

impl<F, Mean, Corr> CrossValScore<F, GpError, GpParams<F, Mean, Corr>, Self>
    for GaussianProcess<F, Mean, Corr>
where
    F: Float,
    Mean: mean_models::RegressionModel<F>,
    Corr: correlation_models::CorrelationModel<F>,
{
    fn training_data(&self) -> &(Array2<F>, Array1<F>) {
        &self.training_data
    }

    fn params(&self) -> GpParams<F, Mean, Corr> {
        GpParams::from(self.params.clone())
    }
}

impl<F, Corr> CrossValScore<F, GpError, SgpParams<F, Corr>, Self> for SparseGaussianProcess<F, Corr>
where
    F: Float,
    Corr: correlation_models::CorrelationModel<F>,
{
    fn training_data(&self) -> &(Array2<F>, Array1<F>) {
        &self.training_data
    }

    fn params(&self) -> SgpParams<F, Corr> {
        SgpParams::from(self.params.clone())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Inducings, SparseKriging};
    use approx::assert_abs_diff_eq;
    use egobox_doe::{Lhs, SamplingMethod};
    use ndarray::{array, Array, Array1, Axis, Data, Ix2, Zip};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::{Normal, Uniform};
    use ndarray_rand::RandomExt;
    use rand_xoshiro::Xoshiro256Plus;

    fn griewank(x: &Array2<f64>) -> Array1<f64> {
        let dim = x.ncols();
        let d = Array1::linspace(1., dim as f64, dim).mapv(|v| v.sqrt());
        let mut y = Array1::zeros((x.nrows(),));
        Zip::from(&mut y).and(x.rows()).for_each(|y, x| {
            let s = x.mapv(|v| v * v).sum() / 4000.;
            let p = (x.to_owned() / &d)
                .mapv(|v| v.cos())
                .fold(1., |acc, x| acc * x);
            *y = s - p + 1.;
        });
        y
    }

    #[test]
    fn test_cv_gp_griewank() {
        let dims = [5]; // , 10, 60];
        let nts = [100]; // , 300, 500];
        let lim = array![[-600., 600.]];

        (0..dims.len()).for_each(|i| {
            let dim = dims[i];
            let nt = nts[i];
            let xlimits = lim.broadcast((dim, 2)).unwrap();

            let rng = Xoshiro256Plus::seed_from_u64(42);
            let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
            let yt = griewank(&xt);

            let gp = GaussianProcess::<
                f64,
                mean_models::ConstantMean,
                correlation_models::SquaredExponentialCorr,
            >::params(
                mean_models::ConstantMean::default(),
                correlation_models::SquaredExponentialCorr::default(),
            )
            .kpls_dim(Some(3))
            .fit(&Dataset::new(xt, yt))
            .expect("GP fit error");

            assert_abs_diff_eq!(gp.loocv_score(), 0., epsilon = 1e-2);
            assert_abs_diff_eq!(gp.cv_score(10), 0., epsilon = 1e-2);
        });
    }

    const PI: f64 = std::f64::consts::PI;

    fn f_obj(x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        x.mapv(|v| (3. * PI * v).sin() + 0.3 * (9. * PI * v).cos() + 0.5 * (7. * PI * v).sin())
    }

    fn make_test_data(
        nt: usize,
        eta2: f64,
        rng: &mut Xoshiro256Plus,
    ) -> (Array2<f64>, Array1<f64>) {
        let normal = Normal::new(0., eta2.sqrt()).unwrap();
        let gaussian_noise = Array::<f64, _>::random_using((nt, 1), normal, rng);
        let xt = 2. * Array::<f64, _>::random_using((nt, 1), Uniform::new(0., 1.), rng) - 1.;
        let yt = (f_obj(&xt) + gaussian_noise).remove_axis(Axis(1));
        (xt, yt)
    }

    #[test]
    fn test_cv_sgp() {
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        // Generate training data
        let nt = 200;
        // Variance of the gaussian noise on our training data
        let eta2: f64 = 0.01;
        let (xt, yt) = make_test_data(nt, eta2, &mut rng);
        let n_inducings = 30;
        let sgp = SparseKriging::params(Inducings::Randomized(n_inducings))
            .seed(Some(42))
            .fit(&Dataset::new(xt.clone(), yt.clone()))
            .expect("GP fitted");

        assert_abs_diff_eq!(sgp.loocv_score(), 22.36, epsilon = 5e-1);
        assert_abs_diff_eq!(sgp.cv_score(10), 64.97, epsilon = 5e-1);
    }
}
