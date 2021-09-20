use super::gaussian_mixture::GaussianMixture;
use crate::errors::MoeError;
use crate::errors::Result;
use crate::{MoeParams, Recombination};
use gp::{ConstantMean, GaussianProcess, SquaredExponentialKernel};
use linfa::{
    dataset::WithLapack, dataset::WithoutLapack, traits::Fit, traits::Predict, Dataset,
    DatasetBase, Float,
};
use linfa_clustering::GaussianMixtureModel;

use ndarray::{concatenate, s, Array, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip};
use ndarray_linalg::{Lapack, Scalar};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;

// impl<F: Float, R: Rng + SeedableRng + Clone, D: Data<Elem = F>, T>
//     Fit<ArrayBase<D, Ix2>, T, GmmError> for MoeParams<F, R>
// {

impl<F: linfa_pls::Float, R: Rng + SeedableRng + Clone> MoeParams<F, R> {
    pub fn fit(
        &self,
        xt: &ArrayBase<impl Data<Elem = F>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Result<Moe<F>> {
        let nx = xt.ncols();
        let data = concatenate(Axis(1), &[xt.view(), yt.view()]).unwrap();

        let dataset = Dataset::from(data.to_owned());

        let gmm = GaussianMixtureModel::params(self.n_clusters())
            .with_n_runs(20)
            //.with_reg_covariance(1e-6)
            .with_rng(self.rng())
            .fit(&dataset)
            .expect("Training data clustering");

        // Clustering using GMM
        let dataset_clustering = match self.gmm() {
            Some(gmm) => gmm.predict(&data),
            None => gmm.predict(&data),
        };

        // GMX for prediction
        let weights = gmm.weights().to_owned();
        let means = gmm.means().slice(s![.., ..nx]).to_owned();
        let covariances = gmm.covariances().slice(s![.., ..nx, ..nx]).to_owned();
        let gmx = GaussianMixture::new(weights, means, covariances)?
            .with_heaviside_factor(self.heaviside_factor());

        let dataset_clustering = gmx.predict(xt);
        let clusters = sort_by_cluster(self.n_clusters(), &data, &dataset_clustering, self.rng());

        check_number_of_points(&clusters, data.ncols())?;

        // Fit GPs on clustered data
        let mut gps = Vec::new();
        for cluster in clusters {
            let xtrain = cluster.slice(s![.., ..nx]);
            let ytrain = cluster.slice(s![.., nx..]);

            gps.push(
                GaussianProcess::<F, ConstantMean, SquaredExponentialKernel>::params(
                    ConstantMean::default(),
                    SquaredExponentialKernel::default(),
                )
                .set_kpls_dim(self.kpls_dim())
                .fit(&xtrain, &ytrain)
                .expect("GP fit error"),
            );
        }

        Ok(Moe {
            recombination: self.recombination(),
            heaviside_factor: self.heaviside_factor(),
            gps,
            gmx,
        })
    }

    pub fn is_heaviside_optimization_enabled(&self) -> bool {
        self.recombination() == Recombination::Smooth && self.n_clusters() > 1
    }
}

fn check_number_of_points<F>(clusters: &Vec<Array2<F>>, dim: usize) -> Result<()> {
    let min_number_point = factorial(dim + 2) / (factorial(dim) * factorial(2));
    for cluster in clusters {
        if cluster.len() < min_number_point {
            return Err(MoeError::GpError(format!(
                "Not enough points in training set. Need {} points, got {}",
                min_number_point,
                cluster.len()
            )));
        }
    }
    Ok(())
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

pub fn sort_by_cluster<F: linfa_pls::Float, R: Rng + SeedableRng + Clone>(
    n_clusters: usize,
    data: &ArrayBase<impl Data<Elem = F>, Ix2>,
    dataset_clustering: &Array1<usize>,
    rng: R,
) -> Vec<Array2<F>> {
    let mut res: Vec<Array2<F>> = Vec::new();
    let ndim = data.ncols();
    for n in 0..n_clusters {
        let cluster_data_indices: Array1<usize> = dataset_clustering
            .iter()
            .enumerate()
            .filter_map(|(k, i)| if *i == n { Some(k) } else { None })
            .collect();
        let nsamples = cluster_data_indices.len();
        let mut subset = Array2::zeros((nsamples, ndim));
        Zip::from(subset.genrows_mut())
            .and(&cluster_data_indices)
            .apply(|mut r, &k| {
                r.assign(&data.row(k));
            });
        res.push(subset);
    }
    res
}

pub struct Moe<F: linfa_pls::Float> {
    recombination: Recombination,
    heaviside_factor: F,
    gps: Vec<GaussianProcess<F, ConstantMean, SquaredExponentialKernel>>,
    gmx: GaussianMixture<F>,
}

impl<F: linfa_pls::Float> Moe<F> {
    pub fn params(n_clusters: usize) -> MoeParams<F, Isaac64Rng> {
        MoeParams::new(n_clusters)
    }

    pub fn nb_clusters(&self) -> usize {
        self.gps.len()
    }

    pub fn recombination(&self) -> Recombination {
        self.recombination
    }

    pub fn heaviside_factor(&self) -> F {
        self.heaviside_factor
    }

    pub fn predict(&self, x: &Array2<F>) -> Result<Array2<F>> {
        match self.recombination {
            Recombination::Hard => self.predict_hard(x),
            Recombination::Smooth => self.predict_smooth(x),
        }
    }

    pub fn predict_smooth(&self, observations: &Array2<F>) -> Result<Array2<F>> {
        let probas = self.gmx.predict_probas(observations);
        let mut preds = Array1::<F>::zeros(observations.nrows());

        Zip::from(&mut preds)
            .and(observations.genrows())
            .and(probas.genrows())
            .par_apply(|y, x, p| {
                let obs = x.clone().insert_axis(Axis(0));
                let subpreds: Array1<F> = self
                    .gps
                    .iter()
                    .map(|gp| gp.predict_values(&obs).unwrap()[[0, 0]])
                    .collect();
                *y = (subpreds * p).sum();
            });
        Ok(preds.insert_axis(Axis(1)))
    }

    pub fn predict_hard(&self, observations: &Array2<F>) -> Result<Array2<F>> {
        let clustering = self.gmx.predict(observations);
        let mut preds = Array2::<F>::zeros((observations.nrows(), 1));
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

pub fn extract_part<F: linfa_pls::Float>(
    data: &ArrayBase<impl Data<Elem = F>, Ix2>,
    quantile: usize,
) -> (Array2<F>, Array2<F>) {
    let nsamples = data.nrows();
    let indices = Array::range(0., nsamples as f32, quantile as f32).mapv(|v| v as usize);
    let data_test = data.select(Axis(0), indices.as_slice().unwrap());
    let indices2: Vec<usize> = (0..nsamples).filter(|i| i % quantile == 0).collect();
    let data_train = data.select(Axis(0), &indices2);
    (data_test, data_train)
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
            .set_recombination(Recombination::Hard)
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
        write_npy("obs.npy", &obs).expect("obs saved");
        write_npy("preds.npy", &preds).expect("preds saved");
    }

    #[test]
    fn test_moe_smooth() {
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = function_test_1d(&xt);
        let moe = Moe::params(3)
            .set_heaviside_factor(0.5)
            .with_rng(rng)
            .fit(&xt, &yt)
            .expect("MOE fitted");
        let obs = Array::linspace(0., 1., 100).insert_axis(Axis(1));
        let preds = moe.predict(&obs).expect("MOE prediction");
        write_npy("obs_smooth.npy", &obs).expect("obs saved");
        write_npy("preds_smooth.npy", &preds).expect("preds saved");
        assert_abs_diff_eq!(
            0.859021,
            moe.predict(&array![[0.39]]).unwrap()[[0, 0]],
            epsilon = 1e-6
        );
    }
}
