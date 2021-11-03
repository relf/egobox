use super::gaussian_mixture::GaussianMixture;
use crate::errors::MoeError;
use crate::errors::Result;
use crate::expert::*;
use crate::{CorrelationSpec, MoeParams, Recombination, RegressionSpec};
use env_logger;
use log::debug;

use gp::{correlation_models::*, mean_models::*, Float, GaussianProcess};
use linfa::dataset::Records;
use linfa::{traits::Fit, traits::Predict, Dataset};
use linfa_clustering::GaussianMixtureModel;
use paste::paste;
use std::cmp::Ordering;
use std::ops::Sub;

use ndarray::{concatenate, s, Array1, Array2, ArrayBase, Axis, Data, Ix2, Zip};
use ndarray_linalg::norm::Norm;
use ndarray_npy::write_npy;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_stats::QuantileExt;
use rand_isaac::Isaac64Rng;

macro_rules! check_allowed {
    ($spec:ident, $model_kind:ident, $model:ident, $list:ident) => {
        paste! {
            if $spec.contains([< $model_kind Spec>]::[< $model:upper >]) {
                $list.push(stringify!($model));
            }
        }
    };
}

impl<R: Rng + SeedableRng + Clone> MoeParams<f64, R> {
    pub fn fit(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Moe> {
        let _opt = env_logger::try_init().ok();
        let nx = xt.ncols();
        let data = concatenate(Axis(1), &[xt.view(), yt.view()]).unwrap();
        let training = data.to_owned();
        // let (test, training) = extract_part(&data, 5); // 10% of data for validation
        // println!(
        //     "training size = {}, test size = {}",
        //     training.nrows(),
        //     test.nrows(),
        // );
        // write_npy("xt_hard.npy", &training.slice(s![.., ..nx])).expect("obs saved");
        // write_npy("yt_hard.npy", &training.slice(s![.., nx..])).expect("preds saved");

        // let xtest = test.slice(s![.., ..nx]).to_owned();
        // let ytest = test.slice(s![.., nx..]).to_owned();
        let dataset = Dataset::from(training.to_owned());

        let gmm = GaussianMixtureModel::params(self.n_clusters())
            .n_runs(20)
            //.reg_covariance(1e-6)
            .with_rng(self.rng())
            .fit(&dataset)
            .expect("Training data clustering");

        // GMX for prediction
        let weights = gmm.weights().to_owned();
        let means = gmm.means().slice(s![.., ..nx]).to_owned();
        let covariances = gmm.covariances().slice(s![.., ..nx, ..nx]).to_owned();
        let factor = match self.recombination() {
            Recombination::Smooth(Some(f)) => f,
            Recombination::Smooth(None) => 1.,
            Recombination::Hard => 1.,
        };
        let gmx = GaussianMixture::new(weights, means, covariances)?.with_heaviside_factor(factor);

        let dataset_clustering = gmx.predict(xt);
        let clusters = sort_by_cluster(self.n_clusters(), &data, &dataset_clustering);

        check_number_of_points(&clusters, data.ncols())?;

        // Fit GPs on clustered data
        let mut experts = Vec::new();
        for cluster in clusters {
            if cluster.nrows() <= 5 {
                return Err(MoeError::ClusteringError(format!(
                    "Not enough points in cluster, requires at least 5, got {}",
                    cluster.nrows()
                )));
            }
            let xtrain = cluster.slice(s![.., ..nx]);
            let ytrain = cluster.slice(s![.., nx..]);

            let expert = self.find_best_expert(nx, &cluster)?;
            experts.push(expert.fit(&xtrain.view(), &ytrain.view())?);
        }

        let recombination = match self.recombination() {
            Recombination::Smooth(_) => Recombination::Smooth(Some(factor)),
            Recombination::Hard => self.recombination(),
        };

        Ok(Moe {
            recombination,
            experts,
            gmx,
        })
    }

    pub fn find_best_expert(&self, nx: usize, data: &Array2<f64>) -> Result<Box<dyn ExpertParams>> {
        let xtrain = data.slice(s![.., ..nx]);
        let ytrain = data.slice(s![.., nx..]);
        let mut dataset = Dataset::from((xtrain.to_owned(), ytrain.to_owned()));
        let regression_spec = self.regression_spec();
        let mut allowed_means = vec![];
        check_allowed!(regression_spec, Regression, Constant, allowed_means);
        check_allowed!(regression_spec, Regression, Linear, allowed_means);
        check_allowed!(regression_spec, Regression, Quadratic, allowed_means);
        let correlation_spec = self.correlation_spec();
        let mut allowed_corrs = vec![];
        check_allowed!(
            correlation_spec,
            Correlation,
            SquaredExponential,
            allowed_corrs
        );
        check_allowed!(
            correlation_spec,
            Correlation,
            AbsoluteExponential,
            allowed_corrs
        );
        check_allowed!(correlation_spec, Correlation, Matern32, allowed_corrs);
        check_allowed!(correlation_spec, Correlation, Matern52, allowed_corrs);

        let mut map_accuracy = Vec::new();
        compute_accuracies!(allowed_means, allowed_corrs, dataset, map_accuracy);
        let errs: Vec<f64> = map_accuracy.iter().map(|(_, err)| *err).collect();
        println!("Accuracies {:?}", map_accuracy);
        let argmin = errs
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap();
        let best = &map_accuracy[argmin].0;
        match best.as_str() {
            "Constant_SquaredExponential" => make_expert_params!(Constant, SquaredExponential),
            "Constant_AbsoluteExponential" => make_expert_params!(Constant, AbsoluteExponential),
            "Constant_Matern32" => make_expert_params!(Constant, Matern32),
            "Constant_Matern52" => make_expert_params!(Constant, Matern52),
            "Linear_SquaredExponential" => make_expert_params!(Linear, SquaredExponential),
            "Linear_AbsoluteExponential" => make_expert_params!(Linear, AbsoluteExponential),
            "Linear_Matern32" => make_expert_params!(Linear, Matern32),
            "Linear_Matern52" => make_expert_params!(Linear, Matern52),
            "Quadratic_SquaredExponential" => make_expert_params!(Quadratic, SquaredExponential),
            "Quadratic_AbsoluteExponential" => make_expert_params!(Quadratic, AbsoluteExponential),
            "Quadratic_Matern32" => make_expert_params!(Quadratic, Matern32),
            "Quadratic_Matern52" => make_expert_params!(Quadratic, Matern52),
            _ => Err(MoeError::ExpertError(format!("Unknown expert {}", best))),
        }
    }

    pub fn optimize_heaviside_factor(
        &self,
        experts: &Vec<Box<dyn Expert>>,
        gmx: &GaussianMixture<f64>,
        xtest: &Array2<f64>,
        ytest: &Array2<f64>,
    ) -> f64 {
        if self.recombination() == Recombination::Hard || self.n_clusters() == 1 {
            1.
        } else {
            let scale_factors = Array1::linspace(0.1, 2.1, 20);
            let errors = scale_factors.map(move |&factor| {
                let gmx2 = gmx.clone();
                let gmx2 = gmx2.with_heaviside_factor(factor);
                let pred = predict_smooth(&experts, &gmx2, &xtest).unwrap();
                pred.sub(ytest).mapv(|x| x * x).sum().sqrt() / xtest.mapv(|x| x * x).sum().sqrt()
            });

            let min_error_index = errors.argmin().unwrap();
            let scale_factor = if *errors.max().unwrap() < 1e-6 {
                1.
            } else {
                scale_factors[min_error_index]
            };
            scale_factor
        }
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

pub fn sort_by_cluster<F: Float>(
    n_clusters: usize,
    data: &ArrayBase<impl Data<Elem = F>, Ix2>,
    dataset_clustering: &Array1<usize>,
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
        Zip::from(subset.rows_mut())
            .and(&cluster_data_indices)
            .for_each(|mut r, &k| {
                r.assign(&data.row(k));
            });
        res.push(subset);
    }
    res
}

pub fn predict_smooth(
    experts: &Vec<Box<dyn Expert>>,
    gmx: &GaussianMixture<f64>,
    observations: &Array2<f64>,
) -> Result<Array2<f64>> {
    let probas = gmx.predict_probas(observations);
    let mut preds = Array1::<f64>::zeros(observations.nrows());

    Zip::from(&mut preds)
        .and(observations.rows())
        .and(probas.rows())
        .for_each(|y, x, p| {
            let obs = x.clone().insert_axis(Axis(0));
            let subpreds: Array1<f64> = experts
                .iter()
                .map(|gp| gp.predict_values(&obs).unwrap()[[0, 0]])
                .collect();
            *y = (subpreds * p).sum();
        });
    Ok(preds.insert_axis(Axis(1)))
}

pub struct Moe {
    recombination: Recombination<f64>,
    experts: Vec<Box<dyn Expert>>,
    gmx: GaussianMixture<f64>,
}

impl std::fmt::Display for Moe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let recomb = match self.recombination() {
            Recombination::Hard => "Hard".to_string(),
            Recombination::Smooth(Some(f)) => format!("Smooth({})", f),
            Recombination::Smooth(None) => "Smooth".to_string(),
        };
        let experts = self
            .experts
            .iter()
            .map(|expert| expert.to_string())
            .reduce(|acc, s| acc + ", " + &s)
            .unwrap();
        write!(f, "Mixture[{}], ({})", &recomb, &experts)
    }
}

impl Moe {
    pub fn params(n_clusters: usize) -> MoeParams<f64, Isaac64Rng> {
        MoeParams::new(n_clusters)
    }

    pub fn n_clusters(&self) -> usize {
        self.experts.len()
    }

    pub fn recombination(&self) -> Recombination<f64> {
        self.recombination
    }

    pub fn set_recombination(mut self, recombination: Recombination<f64>) -> Self {
        self.recombination = match recombination {
            Recombination::Hard => recombination,
            Recombination::Smooth(None) => Recombination::Smooth(Some(1.)),
            Recombination::Smooth(Some(_)) => recombination,
        };
        self
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        match self.recombination {
            Recombination::Hard => self.predict_hard(x),
            Recombination::Smooth(_) => self.predict_smooth(x),
        }
    }

    pub fn save_expert_predict(&self, x: &Array2<f64>) -> () {
        self.experts.iter().enumerate().for_each(|(i, expert)| {
            let preds = expert.predict_values(&x.view()).unwrap();
            write_npy(format!("preds_expert_{}.npy", i), &preds).expect("expert pred saved");
        });
    }

    pub fn predict_smooth(&self, observations: &Array2<f64>) -> Result<Array2<f64>> {
        let probas = self.gmx.predict_probas(observations);
        let mut preds = Array1::<f64>::zeros(observations.nrows());

        Zip::from(&mut preds)
            .and(observations.rows())
            .and(probas.rows())
            .for_each(|y, x, p| {
                let obs = x.clone().insert_axis(Axis(0));
                let subpreds: Array1<f64> = self
                    .experts
                    .iter()
                    .map(|gp| gp.predict_values(&obs).unwrap()[[0, 0]])
                    .collect();
                *y = (subpreds * p).sum();
            });
        Ok(preds.insert_axis(Axis(1)))
    }

    pub fn predict_hard(&self, observations: &Array2<f64>) -> Result<Array2<f64>> {
        let clustering = self.gmx.predict(observations);
        println!("{:?}", clustering);
        let mut preds = Array2::zeros((observations.nrows(), 1));
        Zip::from(preds.rows_mut())
            .and(observations.rows())
            .and(&clustering)
            .for_each(|mut y, x, &c| {
                y.assign(
                    &self.experts[c]
                        .predict_values(&x.insert_axis(Axis(0)))
                        .unwrap()
                        .row(0),
                );
            });
        Ok(preds)
    }
}

pub fn extract_part<F: Float>(
    data: &ArrayBase<impl Data<Elem = F>, Ix2>,
    quantile: usize,
) -> (Array2<F>, Array2<F>) {
    let nsamples = data.nrows();
    let indices = Array1::range(0., nsamples as f32, quantile as f32).mapv(|v| v as usize);
    let data_test = data.select(Axis(0), indices.as_slice().unwrap());
    let indices2: Vec<usize> = (0..nsamples).filter(|i| i % quantile != 0).collect();
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
        let obs = Array1::linspace(0., 1., 100).insert_axis(Axis(1));
        moe.save_expert_predict(&obs);
        let preds = moe.predict(&obs).expect("MOE prediction");
        write_npy("obs_hard.npy", &obs).expect("obs saved");
        write_npy("preds_hard.npy", &preds).expect("preds saved");
        assert_abs_diff_eq!(
            0.39 * 0.39,
            moe.predict(&array![[0.39]]).unwrap()[[0, 0]],
            epsilon = 1e-4
        );
        assert_abs_diff_eq!(
            f64::sin(10. * 0.82),
            moe.predict(&array![[0.82]]).unwrap()[[0, 0]],
            epsilon = 1e-4
        );
    }

    #[test]
    fn test_moe_smooth() {
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = function_test_1d(&xt);
        let moe = Moe::params(3)
            .set_recombination(Recombination::Smooth(Some(0.5)))
            .with_rng(rng)
            .fit(&xt, &yt)
            .expect("MOE fitted");
        let obs = Array1::linspace(0., 1., 100).insert_axis(Axis(1));
        let preds = moe.predict(&obs).expect("MOE prediction");
        write_npy("obs_smooth.npy", &obs).expect("obs saved");
        write_npy("preds_smooth.npy", &preds).expect("preds saved");
        assert_abs_diff_eq!(
            0.859021,
            moe.predict(&array![[0.39]]).unwrap()[[0, 0]],
            epsilon = 1e-3
        );
    }

    fn xsinx(x: &[f64]) -> f64 {
        (x[0] - 3.5) * f64::sin((x[0] - 3.5) / std::f64::consts::PI)
    }

    #[test]
    fn test_find_best_expert() {
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let xt = Array2::random_using((10, 1), Uniform::new(0., 1.), &mut rng);
        let yt = xt.mapv(|x| xsinx(&[x]));
        let data = concatenate(Axis(1), &[xt.view(), yt.view()]).unwrap();
        let moe = Moe::params(1).with_rng(rng);
        let _best_expert = &moe.find_best_expert(1, &data);
    }

    #[test]
    fn test_find_best_heaviside_factor() {
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = function_test_1d(&xt);
        let _moe = Moe::params(3)
            .with_rng(rng)
            .fit(&xt, &yt)
            .expect("MOE fitted");
    }
}
