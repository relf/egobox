use super::gaussian_mixture::GaussianMixture;
use crate::clustering::find_best_number_of_clusters;
use crate::errors::MoeError;
use crate::errors::Result;
use crate::expertise_macros::*;
use crate::parameters::{
    CorrelationSpec, MoeParams, MoeValidParams, Recombination, RegressionSpec,
};
use crate::surrogates::*;
use egobox_gp::{correlation_models::*, mean_models::*, GaussianProcess};
use linfa::dataset::Records;
use linfa::traits::{Fit, Predict};
use linfa::{Dataset, DatasetBase, Float, ParamGuard};
use linfa_clustering::GaussianMixtureModel;
use log::{debug, info, trace};
use paste::paste;
use std::cmp::Ordering;
use std::ops::Sub;

#[cfg(not(feature = "blas"))]
use linfa_linalg::norm::*;
use ndarray::{concatenate, s, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix2, Zip};

#[cfg(feature = "blas")]
use ndarray_linalg::Norm;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_stats::QuantileExt;
use rand_isaac::Isaac64Rng;

#[cfg(feature = "persistent")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "persistent")]
use std::fs;
#[cfg(feature = "persistent")]
use std::io::Write;

macro_rules! check_allowed {
    ($spec:ident, $model_kind:ident, $model:ident, $list:ident) => {
        paste! {
            if $spec.contains([< $model_kind Spec>]::[< $model:upper >]) {
                $list.push(stringify!($model));
            }
        }
    };
}

impl<R: Rng + SeedableRng + Clone> Fit<Array2<f64>, Array2<f64>, MoeError>
    for MoeValidParams<f64, R>
{
    type Object = Moe;

    /// Fit Moe parameters using maximum likelihood
    ///
    /// # Errors
    ///
    /// * [MoeError::ClusteringError]: if there is not enough points regarding the clusters,
    /// * [MoeError::GpError]: if gaussian process fitting fails
    ///
    fn fit(&self, dataset: &DatasetBase<Array2<f64>, Array2<f64>>) -> Result<Self::Object> {
        let x = dataset.records();
        let y = dataset.targets();
        self.train(x, y)
    }
}

impl<R: Rng + SeedableRng + Clone> MoeValidParams<f64, R> {
    fn train(&self, xt: &Array2<f64>, yt: &Array2<f64>) -> Result<Moe> {
        let _opt = env_logger::try_init().ok();
        let nx = xt.ncols();
        let data = concatenate(Axis(1), &[xt.view(), yt.view()]).unwrap();
        let training;
        let (mut xtest, mut ytest) = (None, None);

        let (n_clusters, recomb) = if self.n_clusters() == 0 {
            // automatic mode
            let max_nb_clusters = xt.nrows() / 10 + 1;
            find_best_number_of_clusters(
                xt,
                yt,
                max_nb_clusters,
                self.kpls_dim(),
                self.regression_spec(),
                self.correlation_spec(),
                self.rng(),
            )
        } else {
            (self.n_clusters(), self.recombination())
        };
        if self.n_clusters() == 0 {
            debug!("Automatic settings {} {:?}", n_clusters, recomb);
        }
        if let Recombination::Smooth(None) = recomb {
            // Extract 5% of data for validation
            // TODO: Use cross-validation ? Performances
            let (test, training_data) = extract_part(&data, 5);
            // write_npy("xt_hard.npy", &training.slice(s![.., ..nx])).expect("obs saved");
            // write_npy("yt_hard.npy", &training.slice(s![.., nx..])).expect("preds saved");

            xtest = Some(test.slice(s![.., ..nx]).to_owned());
            ytest = Some(test.slice(s![.., nx..]).to_owned());
            training = training_data;
        } else {
            training = data.to_owned();
        }
        let dataset = Dataset::from(training);

        let gmm = GaussianMixtureModel::params(n_clusters)
            .n_runs(20)
            .with_rng(self.rng())
            .fit(&dataset)
            .expect("Training data clustering");

        // GMX for prediction
        let weights = gmm.weights().to_owned();
        let means = gmm.means().slice(s![.., ..nx]).to_owned();
        let covariances = gmm.covariances().slice(s![.., ..nx, ..nx]).to_owned();
        let factor = match recomb {
            Recombination::Smooth(Some(f)) => f,
            Recombination::Smooth(None) => 1.,
            Recombination::Hard => 1.,
        };
        let gmx = GaussianMixture::new(weights, means, covariances)?.with_heaviside_factor(factor);

        let dataset_clustering = gmx.predict(xt);
        let clusters = sort_by_cluster(n_clusters, &data, &dataset_clustering);

        check_number_of_points(&clusters, xt.ncols())?;

        // Fit GPs on clustered data
        let mut experts = Vec::new();
        let nb_clusters = clusters.len();
        for cluster in clusters {
            if nb_clusters > 1 && cluster.nrows() < 3 {
                return Err(MoeError::ClusteringError(format!(
                    "Not enough points in cluster, requires at least 3, got {}",
                    cluster.nrows()
                )));
            }
            let expert = self.find_best_expert(nx, &cluster)?;
            experts.push(expert);
        }

        if recomb == Recombination::Smooth(None) {
            let factor =
                self.optimize_heaviside_factor(&experts, &gmx, &xtest.unwrap(), &ytest.unwrap());
            MoeParams::from(self.clone())
                .n_clusters(n_clusters)
                .recombination(Recombination::Smooth(Some(factor)))
                .check()?
                .train(xt, yt)
        } else {
            Ok(Moe {
                recombination: recomb,
                experts,
                gmx,
            })
        }
    }

    /// Select the surrogate which gives the smallest prediction error on the given data
    /// The error is computed using cross-validation
    fn find_best_expert(&self, nx: usize, data: &Array2<f64>) -> Result<Box<dyn Surrogate>> {
        let xtrain = data.slice(s![.., ..nx]).to_owned();
        let ytrain = data.slice(s![.., nx..]).to_owned();
        let mut dataset = Dataset::from((xtrain.clone(), ytrain.clone()));
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

        let best = if allowed_means.len() == 1 && allowed_corrs.len() == 1 {
            (format!("{}_{}", allowed_means[0], allowed_corrs[0]), None) // shortcut
        } else {
            let mut map_error = Vec::new();
            compute_errors!(self, allowed_means, allowed_corrs, dataset, map_error);
            let errs: Vec<f64> = map_error.iter().map(|(_, err)| *err).collect();
            debug!("Accuracies {:?}", map_error);
            let argmin = errs
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap();
            (map_error[argmin].0.clone(), Some(map_error[argmin].1))
        };
        let best_expert_params: std::result::Result<Box<dyn SurrogateParams>, MoeError> = match best
            .0
            .as_str()
        {
            "Constant_SquaredExponential" => {
                Ok(make_surrogate_params!(Constant, SquaredExponential))
            }
            "Constant_AbsoluteExponential" => {
                Ok(make_surrogate_params!(Constant, AbsoluteExponential))
            }
            "Constant_Matern32" => Ok(make_surrogate_params!(Constant, Matern32)),
            "Constant_Matern52" => Ok(make_surrogate_params!(Constant, Matern52)),
            "Linear_SquaredExponential" => Ok(make_surrogate_params!(Linear, SquaredExponential)),
            "Linear_AbsoluteExponential" => Ok(make_surrogate_params!(Linear, AbsoluteExponential)),
            "Linear_Matern32" => Ok(make_surrogate_params!(Linear, Matern32)),
            "Linear_Matern52" => Ok(make_surrogate_params!(Linear, Matern52)),
            "Quadratic_SquaredExponential" => {
                Ok(make_surrogate_params!(Quadratic, SquaredExponential))
            }
            "Quadratic_AbsoluteExponential" => {
                Ok(make_surrogate_params!(Quadratic, AbsoluteExponential))
            }
            "Quadratic_Matern32" => Ok(make_surrogate_params!(Quadratic, Matern32)),
            "Quadratic_Matern52" => Ok(make_surrogate_params!(Quadratic, Matern52)),
            _ => return Err(MoeError::ExpertError(format!("Unknown expert {}", best.0))),
        };
        let mut expert_params = best_expert_params?;
        expert_params.kpls_dim(self.kpls_dim());
        let expert = expert_params.fit(&xtrain, &ytrain);
        info!(
            "Best expert {} accuracy={}",
            best.0,
            best.1
                .map_or_else(|| String::from("<Not Computed>"), |v| format!("{}", v))
        );
        expert.map_err(MoeError::from)
    }

    /// Take the best heaviside factor from 0.1 to 2.1 (step 0.1).
    /// Mixture (`gmx` and experts`) is already trained only the continuous recombination is changed
    /// and the factor giving the smallest prediction error on the given test data  
    /// Used only in case of smooth recombination
    fn optimize_heaviside_factor(
        &self,
        experts: &[Box<dyn Surrogate>],
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
                let pred = predict_values_smooth(experts, &gmx2, xtest).unwrap();
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

fn check_number_of_points<F>(clusters: &[Array2<F>], dim: usize) -> Result<()> {
    if clusters.len() > 1 {
        let min_number_point = factorial(dim + 2) / (factorial(dim) * factorial(2));
        for cluster in clusters {
            if cluster.len() < min_number_point {
                return Err(MoeError::ClusteringError(format!(
                    "Not enough points in training set. Need {} points, got {}",
                    min_number_point,
                    cluster.len()
                )));
            }
        }
    }
    Ok(())
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// Return a vector of clustered data set given the `data_clustering` indices which contraints
/// for each `data` rows the cluster number.     
pub(crate) fn sort_by_cluster<F: Float>(
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

/// Predict outputs at given points with `experts` and gaussian mixture `gmx`.
/// `gmx` is used to get the probability of the observation to belongs to one cluster
/// or another (ie responsabilities). Those responsabilities are used to combine
/// output values predict by each cluster experts.
fn predict_values_smooth(
    experts: &[Box<dyn Surrogate>],
    gmx: &GaussianMixture<f64>,
    points: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Result<Array2<f64>> {
    let probas = gmx.predict_probas(points);
    let mut preds = Array1::<f64>::zeros(points.nrows());

    Zip::from(&mut preds)
        .and(points.rows())
        .and(probas.rows())
        .for_each(|y, x, p| {
            let obs = x.insert_axis(Axis(0));
            let subpreds: Array1<f64> = experts
                .iter()
                .map(|gp| gp.predict_values(&obs).unwrap()[[0, 0]])
                .collect();
            *y = (subpreds * p).sum();
        });
    Ok(preds.insert_axis(Axis(1)))
}

/// Mixture of gaussian process experts
#[cfg_attr(feature = "persistent", derive(Serialize, Deserialize))]
pub struct Moe {
    /// The mode of recombination to get the output prediction from experts prediction
    recombination: Recombination<f64>,
    /// The list of the best experts trained on each cluster
    experts: Vec<Box<dyn Surrogate>>,
    /// The gaussian mixture allowing to predict cluster responsabilities for a given point
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

#[cfg_attr(feature = "persistent", typetag::serde)]
impl Surrogate for Moe {
    fn predict_values(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        match self.recombination {
            Recombination::Hard => self.predict_values_hard(x),
            Recombination::Smooth(_) => self.predict_values_smooth(x),
        }
    }

    fn predict_variances(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        match self.recombination {
            Recombination::Hard => self.predict_variances_hard(x),
            Recombination::Smooth(_) => self.predict_variances_smooth(x),
        }
    }

    /// Save Moe model in given file.
    #[cfg(feature = "persistent")]
    fn save(&self, path: &str) -> Result<()> {
        let mut file = fs::File::create(path).unwrap();
        let bytes = match serde_json::to_string(self) {
            Ok(b) => b,
            Err(err) => return Err(MoeError::SaveError(err)),
        };
        file.write_all(bytes.as_bytes())?;
        Ok(())
    }
}

impl Moe {
    /// Constructor of mixture of experts parameters
    pub fn params() -> MoeParams<f64, Isaac64Rng> {
        MoeParams::new()
    }

    /// Number of clusters
    pub fn n_clusters(&self) -> usize {
        self.experts.len()
    }

    /// Recombination mode
    pub fn recombination(&self) -> Recombination<f64> {
        self.recombination
    }

    /// Sets recombination mode
    pub fn set_recombination(mut self, recombination: Recombination<f64>) -> Self {
        self.recombination = match recombination {
            Recombination::Hard => recombination,
            Recombination::Smooth(None) => Recombination::Smooth(Some(1.)),
            Recombination::Smooth(Some(_)) => recombination,
        };
        self
    }

    /// Predict outputs at a set of points `x` specified as (n, xdim) matrix.
    /// Gaussian Mixture is used to get the probability of the point to belongs to one cluster
    /// or another (ie responsabilities). Those responsabilities are used to combine
    /// output values predict by each cluster experts.
    pub fn predict_values_smooth(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        predict_values_smooth(&self.experts, &self.gmx, x)
    }

    /// Predict variances at a set of points `x` specified as (n, xdim) matrix.
    /// Gaussian Mixture is used to get the probability of the point to belongs to one cluster
    /// or another (ie responsabilities). Those responsabilities are used to combine
    /// variances predict by each cluster experts.
    pub fn predict_variances_smooth(
        &self,
        observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let probas = self.gmx.predict_probas(observations);
        let mut preds = Array1::<f64>::zeros(observations.nrows());

        Zip::from(&mut preds)
            .and(observations.rows())
            .and(probas.rows())
            .for_each(|y, x, p| {
                let obs = x.insert_axis(Axis(0));
                let subpreds: Array1<f64> = self
                    .experts
                    .iter()
                    .map(|gp| gp.predict_variances(&obs).unwrap()[[0, 0]])
                    .collect();
                *y = (subpreds * p * p).sum();
            });
        Ok(preds.insert_axis(Axis(1)))
    }

    /// Predict outputs at a set of points `x` specified as (n, xdim) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// Then the expert of the cluster is used to predict output value.
    pub fn predict_values_hard(
        &self,
        observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let clustering = self.gmx.predict(observations);
        debug!("Clustering {:?}", clustering);
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

    /// Predict variance at a set of points `x` specified as (n, xdim) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// Then the expert of the cluster is used to predict output value.
    pub fn predict_variances_hard(
        &self,
        observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let clustering = self.gmx.predict(observations);
        debug!("Clustering {:?}", clustering);
        let mut variances = Array2::zeros((observations.nrows(), 1));
        Zip::from(variances.rows_mut())
            .and(observations.rows())
            .and(&clustering)
            .for_each(|mut y, x, &c| {
                y.assign(
                    &self.experts[c]
                        .predict_variances(&x.insert_axis(Axis(0)))
                        .unwrap()
                        .row(0),
                );
            });
        Ok(variances)
    }

    pub fn predict_values(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        <Moe as Surrogate>::predict_values(self, &x.view())
    }

    pub fn predict_variances(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        <Moe as Surrogate>::predict_variances(self, &x.view())
    }

    #[cfg(feature = "persistent")]
    /// Load Moe from given json file.
    pub fn load(path: &str) -> Result<Box<Moe>> {
        let data = fs::read_to_string(path)?;
        let moe: Moe = serde_json::from_str(&data).unwrap();
        Ok(Box::new(moe))
    }
}

/// Take one out of `quantile` in a set of data rows.
/// Returns the selectionned part and the remaining data.
fn extract_part<F: Float>(
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
        let moe = Moe::params()
            .n_clusters(3)
            .recombination(Recombination::Hard)
            .with_rng(rng)
            .fit(&Dataset::new(xt, yt))
            .expect("MOE fitted");
        let obs = Array1::linspace(0., 1., 100).insert_axis(Axis(1));
        let preds = moe.predict_values(&obs).expect("MOE prediction");
        write_npy("obs_hard.npy", &obs).expect("obs saved");
        write_npy("preds_hard.npy", &preds).expect("preds saved");
        assert_abs_diff_eq!(
            0.39 * 0.39,
            moe.predict_values(&array![[0.39]]).unwrap()[[0, 0]],
            epsilon = 1e-4
        );
        assert_abs_diff_eq!(
            f64::sin(10. * 0.82),
            moe.predict_values(&array![[0.82]]).unwrap()[[0, 0]],
            epsilon = 1e-4
        );
    }

    #[test]
    fn test_moe_smooth() {
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = function_test_1d(&xt);
        let ds = Dataset::new(xt, yt);
        let moe = Moe::params()
            .n_clusters(3)
            .recombination(Recombination::Smooth(Some(0.5)))
            .with_rng(rng.clone())
            .fit(&ds)
            .expect("MOE fitted");
        let obs = Array1::linspace(0., 1., 100).insert_axis(Axis(1));
        let preds = moe.predict_values(&obs).expect("MOE prediction");
        println!("Smooth moe {}", moe);
        assert_abs_diff_eq!(
            0.37579, // true value = 0.37*0.37 = 0.1369
            moe.predict_values(&array![[0.37]]).unwrap()[[0, 0]],
            epsilon = 1e-3
        );
        let moe = Moe::params()
            .n_clusters(3)
            .recombination(Recombination::Smooth(None))
            .with_rng(rng)
            .fit(&ds)
            .expect("MOE fitted");
        println!("Smooth moe {}", moe);
        write_npy("obs_smooth.npy", &obs).expect("obs saved");
        write_npy("preds_smooth.npy", &preds).expect("preds saved");
        assert_abs_diff_eq!(
            0.37 * 0.37, // true value of the function
            moe.predict_values(&array![[0.37]]).unwrap()[[0, 0]],
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_moe_auto() {
        // env_logger::init();
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let xt = Array2::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
        let yt = function_test_1d(&xt);
        let ds = Dataset::new(xt, yt);
        let moe = Moe::params()
            .n_clusters(0)
            .with_rng(rng.clone())
            .fit(&ds)
            .expect("MOE fitted");
        println!(
            "Moe auto: nb clusters={}, recomb={:?}",
            moe.n_clusters(),
            moe.recombination()
        );
        assert_abs_diff_eq!(
            0.37 * 0.37, // true value of the function
            moe.predict_values(&array![[0.37]]).unwrap()[[0, 0]],
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_moe_variances_smooth() {
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = function_test_1d(&xt);
        let moe = Moe::params()
            .n_clusters(3)
            .recombination(Recombination::Smooth(None))
            .regression_spec(RegressionSpec::CONSTANT)
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
            .with_rng(rng.clone())
            .fit(&Dataset::new(xt, yt))
            .expect("MOE fitted");
        let obs = Array1::linspace(0., 1., 100).insert_axis(Axis(1));
        let variances = moe
            .predict_variances(&obs)
            .expect("MOE variances prediction");
        assert_abs_diff_eq!(*variances.max().unwrap(), 0., epsilon = 1e-10);
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
        let moe = Moe::params().with_rng(rng).check_unwrap();
        let best_expert = &moe.find_best_expert(1, &data).unwrap();
        println!("Best expert {}", best_expert);
    }

    #[test]
    fn test_find_best_heaviside_factor() {
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = function_test_1d(&xt);
        let _moe = Moe::params()
            .n_clusters(3)
            .with_rng(rng)
            .fit(&Dataset::new(xt, yt))
            .expect("MOE fitted");
    }

    #[cfg(feature = "persistent")]
    #[test]
    fn test_save_load_moe() {
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = function_test_1d(&xt);
        let ds = Dataset::new(xt, yt);
        let moe = Moe::params()
            .n_clusters(3)
            .with_rng(rng)
            .fit(&ds)
            .expect("MOE fitted");
        let xtest = array![[0.6]];
        let y_expected = moe.predict_values(&xtest).unwrap();
        moe.save("saved_moe.json").expect("MoE saving");
        let new_moe = Moe::load("saved_moe.json").expect("MoE loading");
        assert_abs_diff_eq!(
            y_expected,
            new_moe.predict_values(&xtest).unwrap(),
            epsilon = 1e-6
        );
    }
}
