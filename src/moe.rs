use crate::errors::Result;
use crate::gaussian_mixture::GaussianMixture;
use crate::gaussian_process::GaussianProcess;
use crate::utils::{ConstantMean, MultivariateNormal};
use linfa::{traits::Fit, traits::Predict, Dataset, Float};
use linfa_clustering::GaussianMixtureModel;
use ndarray::{arr1, s, stack, Array, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip};
use ndarray_linalg::{Lapack, Scalar};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;

enum Recombination {
    Hard,
    Smooth,
}

struct MoeHyperParams<R: Rng + Clone> {
    n_clusters: usize,
    recombination: Recombination,
    heaviside_factor: Option<f64>,
    rng: R,
}

impl MoeHyperParams<Isaac64Rng> {
    pub fn new(n_clusters: usize) -> MoeHyperParams<Isaac64Rng> {
        Self::new_with_rng(n_clusters, Isaac64Rng::seed_from_u64(42))
    }

    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> MoeHyperParams<R2> {
        MoeHyperParams {
            n_clusters: self.n_clusters,
            recombination: self.recombination,
            heaviside_factor: self.heaviside_factor,
            rng,
        }
    }
}

impl<R: Rng + Clone> MoeHyperParams<R> {
    pub fn new_with_rng(n_clusters: usize, rng: R) -> MoeHyperParams<R> {
        MoeHyperParams {
            n_clusters,
            recombination: Recombination::Hard,
            heaviside_factor: None,
            rng,
        }
    }

    fn fit(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<MixtureOfExperts> {
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
        let gmm = GaussianMixtureModel::params(self.n_clusters)
            .with_n_runs(20)
            .with_reg_covariance(1e-6)
            .with_rng(self.rng.clone())
            .fit(&dataset)
            .expect("Training data clustering");

        // Fit GPs on clustered data
        // let dists = self._create_cluster_distributions(&gmm);
        let dataset_clustering = gmm.predict(dataset);
        let clusters = self.sort_by_cluster(dataset_clustering);
        let mut gps = Vec::new();
        for cluster in clusters {
            let xtrain = cluster.slice(s![.., ..nx]);
            let ytrain = cluster.slice(s![.., nx..nx + 1]);
            gps.push(
                GaussianProcess::<ConstantMean>::params(ConstantMean::new())
                    .fit(&xtrain, &ytrain)
                    .expect("GP fit error"),
            );
        }

        // GMX for prediction
        let weights = gmm.weights().to_owned();
        let means = gmm.means().slice(s![.., ..nx]).to_owned();
        let covariances = gmm.covariances().slice(s![.., ..nx, ..nx]).to_owned();
        let gmx = GaussianMixture::new(weights, means, covariances)?;
        Ok(MixtureOfExperts { gps, gmx })
    }

    fn is_heaviside_optimization_enabled(&self) -> bool {
        match self.heaviside_factor {
            Some(_) => self.n_clusters > 1,
            None => false,
        }
    }

    fn sort_by_cluster(&self, dataset: Dataset<Array2<f64>, Array1<usize>>) -> Vec<Array2<f64>> {
        let mut res: Vec<Array2<f64>> = Vec::new();
        let ndim = dataset.records.ncols();
        for n in 0..self.n_clusters {
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

    fn _create_cluster_distributions(
        &self,
        gmm: &GaussianMixtureModel<f64>,
    ) -> Vec<MultivariateNormal> {
        let means = gmm.means();
        let h = match self.heaviside_factor {
            Some(factor) => factor,
            None => 1.0,
        };
        let cov = gmm.covariances().mapv(|v| v * h);
        let mut dists = Vec::new();
        for k in 0..self.n_clusters {
            let meansk = means.slice(s![k, ..]);
            let covk = cov.slice(s![k, .., ..]);
            let mvn = MultivariateNormal::new(&meansk, &covk);
            dists.push(mvn);
        }
        dists
    }

    fn _extract_part(
        data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        quantile: usize,
    ) -> (Array2<f64>, Array2<f64>) {
        let nsamples = data.nrows();
        let ndim = data.ncols();
        let indices = Array::range(0., nsamples as f32, quantile as f32).mapv(|v| v as usize);
        let data_test = data.select(Axis(0), indices.as_slice().unwrap());
        let indices2: Vec<usize> = (0..nsamples)
            .filter_map(|i| if i % quantile == 0 { None } else { Some(i) })
            .collect();
        let data_train = data.select(Axis(0), &indices2);
        (data_test, data_train)
    }
}

struct MixtureOfExperts {
    gps: Vec<GaussianProcess<ConstantMean>>,
    gmx: GaussianMixture<f64>,
}

impl MixtureOfExperts {
    pub fn params(n_clusters: usize) -> MoeHyperParams<Isaac64Rng> {
        MoeHyperParams::new(n_clusters)
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self._predict_hard(x)
    }

    pub fn _predict_hard(&self, observations: &Array2<f64>) -> Result<Array2<f64>> {
        let clustering = self.gmx.predict(observations);
        println!("clustering records={:?}", &clustering);
        let mut pred = Array2::<f64>::zeros((observations.nrows(), 1));
        Zip::from(pred.genrows_mut())
            .and(observations.genrows())
            .and(&clustering)
            .apply(|mut y, x, &c| {
                y.assign(
                    &self.gps[c]
                        .predict_values(&x.insert_axis(Axis(1)))
                        .unwrap()
                        .row(0),
                );
            });
        Ok(pred)
    }
}

#[cfg(test)]
mod tests {
    extern crate openblas_src;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr2, array, Array2, Zip};
    use ndarray_npy::write_npy;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn function_test_1d(x: &Array2<f64>) -> Array2<f64> {
        let mut y = Array2::zeros(x.dim());
        Zip::from(&mut y).and(x).apply(|yi, &xi| {
            if xi < 0.4 {
                *yi = xi * xi;
            } else if xi >= 0.4 && xi < 0.8 {
                *yi = 3. * xi + 1.;
            } else {
                *yi = f64::sin(10. * xi);
            }
        });
        y
    }

    #[test]
    fn test_moe() {
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        println!("{:?}", xt);
        let yt = function_test_1d(&xt);
        let moe = MoeHyperParams::new(3)
            .with_rng(rng)
            .fit(&xt, &yt)
            .expect("MOE fitted");
        let obs = Array::linspace(0., 1., 100).insert_axis(Axis(1));
        let preds = moe.predict(&obs).expect("MOE prediction");
        write_npy("obs.npy", obs).expect("obs saved");
        write_npy("preds.npy", preds).expect("pred saved");
    }
}
