use super::gaussian_mixture::GaussianMixture;
use crate::clustering::{find_best_number_of_clusters, sort_by_cluster};
use crate::errors::MoeError;
use crate::errors::Result;
use crate::expertise_macros::*;
use crate::sgp_parameters::{SparseGpMixParams, SparseGpMixValidParams};
use crate::surrogates::*;
use crate::types::*;
use egobox_gp::{
    correlation_models::*, mean_models::*, GaussianProcess, Inducings, SparseGaussianProcess,
};
use linfa::dataset::Records;
use linfa::traits::{Fit, Predict, PredictInplace};
use linfa::{Dataset, DatasetBase, Float, ParamGuard};
use linfa_clustering::GaussianMixtureModel;
use log::{debug, info, trace};
use paste::paste;
use std::cmp::Ordering;
use std::ops::Sub;

#[cfg(not(feature = "blas"))]
use linfa_linalg::norm::*;
use ndarray::{
    concatenate, s, Array1, Array2, Array3, ArrayBase, ArrayView2, Axis, Data, Ix2, Zip,
};

#[cfg(feature = "blas")]
use ndarray_linalg::Norm;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_stats::QuantileExt;
use rand_xoshiro::Xoshiro256Plus;

#[cfg(feature = "serializable")]
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

impl<D: Data<Elem = f64>, R: Rng + SeedableRng + Clone>
    Fit<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>, MoeError> for SparseGpMixValidParams<f64, R>
{
    type Object = SparseGpMixture;

    /// Fit Sgp parameters using maximum likelihood
    ///
    /// # Errors
    ///
    /// * [MoeError::ClusteringError]: if there is not enough points regarding the clusters,
    /// * [MoeError::GpError]: if gaussian process fitting fails
    ///
    fn fit(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
    ) -> Result<Self::Object> {
        let x = dataset.records();
        let y = dataset.targets();
        self.train(x, y)
    }
}

impl<R: Rng + SeedableRng + Clone> SparseGpMixValidParams<f64, R> {
    pub fn train(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<SparseGpMixture> {
        trace!("Sgp training...");
        let _opt = env_logger::try_init().ok();
        let nx = xt.ncols();
        let data = concatenate(Axis(1), &[xt.view(), yt.view()]).unwrap();

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

        let training = if recomb == Recombination::Smooth(None) && self.n_clusters() > 1 {
            // Extract 5% of data for validation
            // TODO: Use cross-validation ? Performances
            let (_, training_data) = extract_part(&data, 5);
            training_data
        } else {
            data.to_owned()
        };
        let dataset = Dataset::from(training);

        let gmx = if self.gmx().is_some() {
            *self.gmx().as_ref().unwrap().clone()
        } else {
            trace!("GMM training...");
            let gmm = GaussianMixtureModel::params(n_clusters)
                .n_runs(20)
                .with_rng(self.rng())
                .fit(&dataset)?;

            // GMX for prediction
            let weights = gmm.weights().to_owned();
            let means = gmm.means().slice(s![.., ..nx]).to_owned();
            let covariances = gmm.covariances().slice(s![.., ..nx, ..nx]).to_owned();
            let factor = match recomb {
                Recombination::Smooth(Some(f)) => f,
                Recombination::Smooth(None) => 1.,
                Recombination::Hard => 1.,
            };
            GaussianMixture::new(weights, means, covariances)?.heaviside_factor(factor)
        };

        trace!("Train on clusters...");
        let clustering = Clustering::new(gmx, recomb);
        self.train_on_clusters(&xt.view(), &yt.view(), &clustering)
    }

    /// Using the current state of the clustering, select and train the experts
    /// Returns the fitted mixture of experts model
    pub fn train_on_clusters(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        clustering: &Clustering,
    ) -> Result<SparseGpMixture> {
        let gmx = clustering.gmx();
        let recomb = clustering.recombination();
        let nx = xt.ncols();
        let data = concatenate(Axis(1), &[xt.view(), yt.view()]).unwrap();

        let dataset_clustering = gmx.predict(xt);
        let clusters = sort_by_cluster(gmx.n_clusters(), &data, &dataset_clustering);

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

        if recomb == Recombination::Smooth(None) && self.n_clusters() > 1 {
            // Extract 5% of data for validation
            // TODO: Use cross-validation ? Performances
            let (test, _) = extract_part(&data, 5);
            let xtest = test.slice(s![.., ..nx]).to_owned();
            let ytest = test.slice(s![.., nx..]).to_owned();
            let factor = self.optimize_heaviside_factor(&experts, gmx, &xtest, &ytest);
            info!("Retrain mixture with optimized heaviside factor={}", factor);
            let sgp = SparseGpMixParams::from(self.clone())
                .n_clusters(gmx.n_clusters())
                .recombination(Recombination::Smooth(Some(factor)))
                .check()?
                .train(xt, yt)?; // needs to train the gaussian mixture on all data (xt, yt) as it was
                                 // previously trained on data excluding test data (see train method)
            Ok(sgp)
        } else {
            Ok(SparseGpMixture {
                recombination: recomb,
                experts,
                gmx: gmx.clone(),
                output_dim: yt.ncols(),
            })
        }
    }

    /// Select the surrogate which gives the smallest prediction error on the given data
    /// The error is computed using cross-validation
    fn find_best_expert(
        &self,
        nx: usize,
        data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Box<dyn SgpSurrogate>> {
        let xtrain = data.slice(s![.., ..nx]).to_owned();
        let ytrain = data.slice(s![.., nx..]).to_owned();
        let mut dataset = Dataset::from((xtrain.clone(), ytrain.clone()));
        let regression_spec = RegressionSpec::CONSTANT;
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

        debug!("Find best expert");
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
        debug!("after Find best expert");
        let inducings = self.inducings().clone();
        let best_expert_params: std::result::Result<Box<dyn SgpSurrogateParams>, MoeError> =
            match best.0.as_str() {
                "Constant_SquaredExponential" => {
                    Ok(make_sgp_surrogate_params!(SquaredExponential, inducings))
                }
                "Constant_AbsoluteExponential" => {
                    Ok(make_sgp_surrogate_params!(AbsoluteExponential, inducings))
                }
                "Constant_Matern32" => Ok(make_sgp_surrogate_params!(Matern32, inducings)),
                "Constant_Matern52" => Ok(make_sgp_surrogate_params!(Matern52, inducings)),
                _ => return Err(MoeError::ExpertError(format!("Unknown expert {}", best.0))),
            };
        let mut expert_params = best_expert_params?;
        let seed = self.rng().gen();
        expert_params.kpls_dim(self.kpls_dim());
        expert_params.seed(seed);
        let expert = expert_params.train(&xtrain.view(), &ytrain.view());
        if let Some(v) = best.1 {
            info!("Best expert {} accuracy={}", best.0, v);
        }
        expert.map_err(MoeError::from)
    }

    /// Take the best heaviside factor from 0.1 to 2.1 (step 0.1).
    /// Mixture (`gmx` and experts`) is already trained only the continuous recombination is changed
    /// and the factor giving the smallest prediction error on the given test data  
    /// Used only in case of smooth recombination
    fn optimize_heaviside_factor(
        &self,
        experts: &[Box<dyn SgpSurrogate>],
        gmx: &GaussianMixture<f64>,
        xtest: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        ytest: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> f64 {
        if self.recombination() == Recombination::Hard || self.n_clusters() == 1 {
            1.
        } else {
            let scale_factors = Array1::linspace(0.1, 2.1, 20);
            let errors = scale_factors.map(move |&factor| {
                let gmx2 = gmx.clone();
                let gmx2 = gmx2.heaviside_factor(factor);
                let pred = predict_values_smooth(experts, &gmx2, xtest).unwrap();
                pred.sub(ytest).mapv(|x| x * x).sum().sqrt() / xtest.mapv(|x| x * x).sum().sqrt()
            });

            let min_error_index = errors.argmin().unwrap();
            if *errors.max().unwrap() < 1e-6 {
                1.
            } else {
                scale_factors[min_error_index]
            }
        }
    }
}

fn check_number_of_points<F>(
    clusters: &[ArrayBase<impl Data<Elem = F>, Ix2>],
    dim: usize,
) -> Result<()> {
    if clusters.len() > 1 {
        let min_number_point = (dim + 1) * (dim + 2) / 2;
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

/// Predict outputs at given points with `experts` and gaussian mixture `gmx`.
/// `gmx` is used to get the probability of x to belongs to one cluster
/// or another (ie responsabilities). Those responsabilities are used to combine
/// output values predict by each cluster experts.
fn predict_values_smooth(
    experts: &[Box<dyn SgpSurrogate>],
    gmx: &GaussianMixture<f64>,
    points: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Result<Array2<f64>> {
    let probas = gmx.predict_probas(points);
    let mut preds = Array1::<f64>::zeros(points.nrows());

    Zip::from(&mut preds)
        .and(points.rows())
        .and(probas.rows())
        .for_each(|y, x, p| {
            let x = x.insert_axis(Axis(0));
            let preds: Array1<f64> = experts
                .iter()
                .map(|gp| gp.predict_values(&x).unwrap()[[0, 0]])
                .collect();
            *y = (preds * p).sum();
        });
    Ok(preds.insert_axis(Axis(1)))
}

/// Mixture of gaussian process experts
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct SparseGpMixture {
    /// The mode of recombination to get the output prediction from experts prediction
    recombination: Recombination<f64>,
    /// The list of the best experts trained on each cluster
    experts: Vec<Box<dyn SgpSurrogate>>,
    /// The gaussian mixture allowing to predict cluster responsabilities for a given point
    gmx: GaussianMixture<f64>,
    /// The dimension of the predicted output
    output_dim: usize,
}

impl std::fmt::Display for SparseGpMixture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let recomb = match self.recombination() {
            Recombination::Hard => "Hard".to_string(),
            Recombination::Smooth(Some(f)) => format!("Smooth({f})"),
            Recombination::Smooth(None) => "Smooth".to_string(),
        };
        let experts = self
            .experts
            .iter()
            .map(|expert| expert.to_string())
            .reduce(|acc, s| acc + ", " + &s)
            .unwrap();
        write!(f, "Mixture[{}]({})", &recomb, &experts)
    }
}

impl Clustered for SparseGpMixture {
    /// Number of clusters
    fn n_clusters(&self) -> usize {
        self.gmx.n_clusters()
    }

    /// Clustering Recombination
    fn recombination(&self) -> Recombination<f64> {
        self.recombination()
    }

    /// Convert to clustering
    fn to_clustering(&self) -> Clustering {
        Clustering {
            recombination: self.recombination(),
            gmx: self.gmx.clone(),
        }
    }
}

#[cfg_attr(feature = "serializable", typetag::serde)]
impl GpSurrogate for SparseGpMixture {
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
    /// Save Sgp model in given file.
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

impl ClusteredGpSurrogate for SparseGpMixture {}

impl SparseGpMixture {
    /// Constructor of mixture of experts parameters
    pub fn params(inducings: Inducings<f64>) -> SparseGpMixParams<f64, Xoshiro256Plus> {
        SparseGpMixParams::new(inducings)
    }

    /// Recombination mode
    pub fn recombination(&self) -> Recombination<f64> {
        self.recombination
    }

    pub fn experts(&self) -> &[Box<dyn SgpSurrogate>] {
        &self.experts
    }

    /// Clustering Recombination
    pub fn noise_variance(&self) -> Vec<f64> {
        self.experts.iter().map(|e| e.noise_variance()).collect()
    }

    /// Clustering Recombination
    pub fn variance(&self) -> Vec<f64> {
        self.experts.iter().map(|e| e.variance()).collect()
    }

    /// Gaussian mixture
    pub fn gmx(&self) -> &GaussianMixture<f64> {
        &self.gmx
    }

    /// Retrieve output dimensions from
    pub fn output_dim(&self) -> usize {
        self.output_dim
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

    pub fn set_gmx(&mut self, weights: Array1<f64>, means: Array2<f64>, covariances: Array3<f64>) {
        self.gmx = GaussianMixture::new(weights, means, covariances).unwrap();
    }

    /// Predict outputs at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the probability of the point to belongs to one cluster
    /// or another (ie responsabilities).     
    /// The smooth recombination of each cluster expert responsabilty is used to get the result.
    pub fn predict_values_smooth(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        predict_values_smooth(&self.experts, &self.gmx, x)
    }

    /// Predict variances at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the probability of the point to belongs to one cluster
    /// or another (ie responsabilities).
    /// The smooth recombination of each cluster expert responsabilty is used to get the result.
    pub fn predict_variances_smooth(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let probas = self.gmx.predict_probas(x);
        let mut preds = Array1::<f64>::zeros(x.nrows());

        Zip::from(&mut preds)
            .and(x.rows())
            .and(probas.rows())
            .for_each(|y, x, p| {
                let x = x.insert_axis(Axis(0));
                let preds: Array1<f64> = self
                    .experts
                    .iter()
                    .map(|gp| gp.predict_variances(&x).unwrap()[[0, 0]])
                    .collect();
                *y = (preds * p * p).sum();
            });
        Ok(preds.insert_axis(Axis(1)))
    }

    /// Predict outputs at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// Then the expert of the cluster is used to predict the output value.
    /// Returns the ouputs as a (n, 1) column vector
    pub fn predict_values_hard(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let clustering = self.gmx.predict(x);
        trace!("Clustering {:?}", clustering);
        let mut preds = Array2::zeros((x.nrows(), 1));
        Zip::from(preds.rows_mut())
            .and(x.rows())
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

    /// Predict variance at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// The expert of the cluster is used to predict variance value.
    /// Returns the variances as a (n, 1) column vector
    pub fn predict_variances_hard(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let clustering = self.gmx.predict(x);
        trace!("Clustering {:?}", clustering);
        let mut variances = Array2::zeros((x.nrows(), 1));
        Zip::from(variances.rows_mut())
            .and(x.rows())
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

    pub fn predict_values(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Result<Array2<f64>> {
        <SparseGpMixture as GpSurrogate>::predict_values(self, &x.view())
    }

    pub fn predict_variances(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        <SparseGpMixture as GpSurrogate>::predict_variances(self, &x.view())
    }

    #[cfg(feature = "persistent")]
    /// Load Sgp from given json file.
    pub fn load(path: &str) -> Result<Box<SparseGpMixture>> {
        let data = fs::read_to_string(path)?;
        let sgp: SparseGpMixture = serde_json::from_str(&data).unwrap();
        Ok(Box::new(sgp))
    }
}

/// Take one out of `quantile` in a set of data rows
/// Returns the selected part and the remaining data.
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

impl<D: Data<Elem = f64>> PredictInplace<ArrayBase<D, Ix2>, Array2<f64>> for SparseGpMixture {
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array2<f64>) {
        assert_eq!(
            x.nrows(),
            y.nrows(),
            "The number of data points must match the number of output targets."
        );

        let values = self.predict_values(x).expect("Sgp prediction");
        *y = values;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array2<f64> {
        Array2::zeros((x.nrows(), self.output_dim()))
    }
}

/// Adaptator to implement `linfa::Predict` for variance prediction
pub struct SgpVariancePredictor<'a>(&'a SparseGpMixture);
impl<'a, D: Data<Elem = f64>> PredictInplace<ArrayBase<D, Ix2>, Array2<f64>>
    for SgpVariancePredictor<'a>
{
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array2<f64>) {
        assert_eq!(
            x.nrows(),
            y.nrows(),
            "The number of data points must match the number of output targets."
        );

        let values = self
            .0
            .predict_variances(x)
            .expect("Sgp variances prediction");
        *y = values;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array2<f64> {
        Array2::zeros((x.nrows(), self.0.output_dim()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use ndarray::Array;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::{Normal, Uniform};
    use ndarray_rand::RandomExt;
    use rand_xoshiro::Xoshiro256Plus;

    const PI: f64 = std::f64::consts::PI;

    fn f_obj(x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        x.mapv(|v| (3. * PI * v).sin() + 0.3 * (9. * PI * v).cos() + 0.5 * (7. * PI * v).sin())
    }

    fn make_test_data(
        nt: usize,
        eta2: f64,
        rng: &mut Xoshiro256Plus,
    ) -> (Array2<f64>, Array2<f64>) {
        let normal = Normal::new(0., eta2.sqrt()).unwrap();
        let gaussian_noise = Array::<f64, _>::random_using((nt, 1), normal, rng);
        let xt = 2. * Array::<f64, _>::random_using((nt, 1), Uniform::new(0., 1.), rng) - 1.;
        let yt = f_obj(&xt) + gaussian_noise;
        (xt, yt)
    }

    #[test]
    fn test_sgp_default() {
        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        // Generate training data
        let nt = 200;
        // Variance of the gaussian noise on our training data
        let eta2: f64 = 0.01;
        let (xt, yt) = make_test_data(nt, eta2, &mut rng);

        let xplot = Array::linspace(-0.5, 0.5, 50).insert_axis(Axis(1));
        let n_inducings = 30;

        let sgp = SparseGpMixture::params(Inducings::Randomized(n_inducings))
            .with_rng(rng)
            .fit(&Dataset::new(xt.clone(), yt.clone()))
            .expect("GP fitted");

        println!("noise variance={:?}", sgp.experts()[0].noise_variance());

        let sgp_vals = sgp.predict_values(&xplot).unwrap();
        let yplot = f_obj(&xplot);
        let errvals = (yplot - &sgp_vals).mapv(|v| v.abs());
        assert_abs_diff_eq!(errvals, Array2::zeros((xplot.nrows(), 1)), epsilon = 1.0);
        let sgp_vars = sgp.predict_variances(&xplot).unwrap();
        let errvars = (&sgp_vars - Array2::from_elem((xplot.nrows(), 1), 0.01)).mapv(|v| v.abs());
        assert_abs_diff_eq!(errvars, Array2::zeros((xplot.nrows(), 1)), epsilon = 0.05);
    }
}
