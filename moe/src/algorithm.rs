use super::gaussian_mixture::GaussianMixture;
use crate::errors::Result;
use crate::{MoeParams, Recombination};
use gp::{ConstantMean, GaussianProcess, SquaredExponentialKernel};
use linfa::{traits::Fit, traits::Predict, Dataset, DatasetBase};
use linfa_clustering::GaussianMixtureModel;
use ndarray::{s, stack, Array, Array1, Array2, ArrayBase, Axis, Data, Ix2, Zip};
use ndarray_rand::rand::Rng;
use rand_isaac::Isaac64Rng;

impl<R: Rng + Clone> MoeParams<R> {
    pub fn fit(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Moe> {
        let nx = xt.ncols();
        let data = stack(Axis(1), &[xt.view(), yt.view()]).unwrap();
        let mut xtrain = data.slice(s![.., ..nx]).to_owned();
        let mut ytrain = data.slice(s![.., nx..nx + 1]).to_owned();
        if self.is_heaviside_optimization_enabled() {
            // Separate training data and test data for heaviside optim
            let (data_test, data_train) = Self::_extract_part(&data, 10);
            xtrain = data_train.slice(s![.., ..nx]).to_owned();
            ytrain = data_train.slice(s![.., nx..nx + 1]).to_owned();
        }

        // Cluster inputs
        let dataset = Dataset::from(data);
        let gmm = GaussianMixtureModel::params(self.n_clusters())
            .with_n_runs(20)
            .with_reg_covariance(1e-6)
            .with_rng(self.rng())
            .fit(&dataset)
            .expect("Training data clustering");

        // Fit GPs on clustered data
        let dataset_clustering = gmm.predict(dataset);
        let clusters = self.sort_by_cluster(dataset_clustering);
        let mut gps = Vec::new();
        for cluster in clusters {
            let xtrain = cluster.slice(s![.., ..nx]);
            let ytrain = cluster.slice(s![.., nx..nx + 1]);
            gps.push(
                GaussianProcess::<ConstantMean, SquaredExponentialKernel>::params(
                    ConstantMean::default(),
                    SquaredExponentialKernel::default(),
                )
                .set_kpls_dim(self.kpls_dim())
                .fit(&xtrain, &ytrain)
                .expect("GP fit error"),
            );
        }

        // GMX for prediction
        let weights = gmm.weights().to_owned();
        let means = gmm.means().slice(s![.., ..nx]).to_owned();
        let covariances = gmm.covariances().slice(s![.., ..nx, ..nx]).to_owned();
        let gmx = GaussianMixture::new(weights, means, covariances)?
            .with_heaviside_factor(self.heaviside_factor());
        Ok(Moe {
            recombination: self.recombination(),
            heaviside_factor: self.heaviside_factor(),
            gps,
            gmx,
        })
    }

    fn is_heaviside_optimization_enabled(&self) -> bool {
        self.recombination() == Recombination::Smooth && self.n_clusters() > 1
    }

    fn sort_by_cluster(
        &self,
        dataset: DatasetBase<Array2<f64>, Array1<usize>>,
    ) -> Vec<Array2<f64>> {
        let mut res: Vec<Array2<f64>> = Vec::new();
        let ndim = dataset.records.ncols();
        for n in 0..self.n_clusters() {
            let cluster_data_indices: Array1<usize> = dataset
                .targets
                .iter()
                .enumerate()
                .filter_map(|(k, i)| if *i == n { Some(k) } else { None })
                .collect();
            let nsamples = cluster_data_indices.len();
            let mut subset = Array2::zeros((nsamples, ndim));
            Zip::from(subset.genrows_mut())
                .and(&cluster_data_indices)
                .apply(|mut r, &k| {
                    r.assign(&dataset.records.row(k));
                });
            res.push(subset);
        }
        res
    }

    fn _extract_part(
        data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        quantile: usize,
    ) -> (Array2<f64>, Array2<f64>) {
        let nsamples = data.nrows();
        let indices = Array::range(0., nsamples as f32, quantile as f32).mapv(|v| v as usize);
        let data_test = data.select(Axis(0), indices.as_slice().unwrap());
        let indices2: Vec<usize> = (0..nsamples).filter(|i| i % quantile == 0).collect();
        let data_train = data.select(Axis(0), &indices2);
        (data_test, data_train)
    }
}

pub struct Moe {
    recombination: Recombination,
    heaviside_factor: f64,
    gps: Vec<GaussianProcess<ConstantMean, SquaredExponentialKernel>>,
    gmx: GaussianMixture<f64>,
}

impl Moe {
    pub fn params(n_clusters: usize) -> MoeParams<Isaac64Rng> {
        MoeParams::new(n_clusters)
    }

    pub fn nb_clusters(&self) -> usize {
        self.gps.len()
    }

    pub fn recombination(&self) -> Recombination {
        self.recombination
    }

    pub fn heaviside_factor(&self) -> f64 {
        self.heaviside_factor
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        match self.recombination {
            Recombination::Hard => self._predict_hard(x),
            Recombination::Smooth => self._predict_smooth(x),
        }
    }

    pub fn _predict_smooth(&self, observations: &Array2<f64>) -> Result<Array2<f64>> {
        let probas = self.gmx.predict_probas(observations);
        let mut preds = Array1::<f64>::zeros(observations.nrows());

        Zip::from(&mut preds)
            .and(observations.genrows())
            .and(probas.genrows())
            .par_apply(|y, x, p| {
                let obs = x.clone().insert_axis(Axis(0));
                let subpreds: Array1<f64> = self
                    .gps
                    .iter()
                    .map(|gp| gp.predict_values(&obs).unwrap()[[0, 0]])
                    .collect();
                *y = (subpreds * p).sum();
            });
        Ok(preds.insert_axis(Axis(1)))
    }

    pub fn _predict_hard(&self, observations: &Array2<f64>) -> Result<Array2<f64>> {
        let clustering = self.gmx.predict(observations);
        let mut preds = Array2::<f64>::zeros((observations.nrows(), 1));
        Zip::from(preds.genrows_mut())
            .and(observations.genrows())
            .and(&clustering)
            .par_apply(|mut y, x, &c| {
                y.assign(
                    &self.gps[c]
                        .predict_values(&x.insert_axis(Axis(0)))
                        .unwrap()
                        .row(0),
                );
            });
        Ok(preds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2, Zip};
    use ndarray_npy::write_npy;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand_isaac::Isaac64Rng;

    fn function_test_1d(x: &Array2<f64>) -> Array2<f64> {
        let mut y = Array2::zeros(x.dim());
        Zip::from(&mut y).and(x).apply(|yi, &xi| {
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

    #[test]
    fn test_moe_hard() {
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = function_test_1d(&xt);
        let moe = Moe::params(3)
            .with_rng(rng)
            .fit(&xt, &yt)
            .expect("MOE fitted");
        let obs = Array::linspace(0., 1., 100).insert_axis(Axis(1));
        let preds = moe.predict(&obs).expect("MOE prediction");
        assert_abs_diff_eq!(
            0.39 * 0.39, // 0.1521
            moe.predict(&array![[0.39]]).unwrap()[[0, 0]],
            epsilon = 1e-4
        );
        write_npy("obs.npy", obs).expect("obs saved");
        write_npy("preds.npy", preds).expect("preds saved");
    }

    #[test]
    fn test_moe_smooth() {
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = function_test_1d(&xt);
        let moe = Moe::params(3)
            .set_recombination(Recombination::Smooth)
            .set_heaviside_factor(0.5)
            .with_rng(rng)
            .fit(&xt, &yt)
            .expect("MOE fitted");
        let obs = Array::linspace(0., 1., 100).insert_axis(Axis(1));
        let preds = moe.predict(&obs).expect("MOE prediction");
        write_npy("obs_smooth.npy", obs).expect("obs saved");
        write_npy("preds_smooth.npy", preds).expect("preds saved");
        assert_abs_diff_eq!(
            0.859021,
            moe.predict(&array![[0.39]]).unwrap()[[0, 0]],
            epsilon = 1e-6
        );
    }
}
