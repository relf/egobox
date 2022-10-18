//use super::gaussian_mixture::GaussianMixture;
use super::multivariate_normal::MultivariateNormal;
use crate::clustering::{find_best_number_of_clusters, sort_by_cluster, Clustered, Clustering};
use crate::errors::MoeError;
use crate::errors::Result;
use crate::expertise_macros::*;
use crate::parameters::{
    CorrelationSpec, MoeParams, MoeValidParams, Recombination, RegressionSpec,
};
use crate::surrogates::*;
use egobox_gp::{correlation_models::*, mean_models::*, GaussianProcess};
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

impl<D: Data<Elem = f64>, R: Rng + SeedableRng + Clone>
    Fit<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>, MoeError> for MoeValidParams<f64, R>
{
    type Object = Moe;

    /// Fit Moe parameters using maximum likelihood
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

impl<R: Rng + SeedableRng + Clone> MoeValidParams<f64, R> {
    pub fn train(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Moe> {
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
            MultivariateNormal::new(weights, means, covariances)?.heaviside_factor(factor)
        };

        let clustering = Clustering::new(gmx, recomb);
        self.train_on_clusters(&xt.view(), &yt.view(), &clustering)
    }

    pub fn train_on_clusters(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        clustering: &Clustering,
    ) -> Result<Moe> {
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
            let xtest = Some(test.slice(s![.., ..nx]).to_owned());
            let ytest = Some(test.slice(s![.., nx..]).to_owned());
            let factor =
                self.optimize_heaviside_factor(&experts, gmx, &xtest.unwrap(), &ytest.unwrap());
            info!("Retrain mixture with optimized heaviside factor={}", factor);
            let moe = MoeParams::from(self.clone())
                .n_clusters(gmx.n_clusters())
                .recombination(Recombination::Smooth(Some(factor)))
                .check()?
                .train(xt, yt)?; // needs to train the gaussian mixture on all data (xt, yt) as it was
                                 // previously trained on data excluding test data (see train method)
            Ok(moe)
        } else {
            Ok(Moe {
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
    ) -> Result<Box<dyn Surrogate>> {
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
        experts: &[Box<dyn Surrogate>],
        gmx: &MultivariateNormal<f64>,
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

/// Predict outputs at given points with `experts` and gaussian mixture `gmx`.
/// `gmx` is used to get the probability of the observation to belongs to one cluster
/// or another (ie responsabilities). Those responsabilities are used to combine
/// output values predict by each cluster experts.
fn predict_values_smooth(
    experts: &[Box<dyn Surrogate>],
    gmx: &MultivariateNormal<f64>,
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
    gmx: MultivariateNormal<f64>,
    /// The dimension of the predicted output
    output_dim: usize,
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

impl Clustered for Moe {
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

    fn predict_jacobian(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        match self.recombination {
            Recombination::Hard => self.predict_jacobian_hard(x),
            Recombination::Smooth(_) => self.predict_jacobian_smooth(x),
        }
    }

    fn predict_variance_jacobian(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        match self.recombination {
            Recombination::Hard => self.predict_variance_jacobian_hard(x),
            Recombination::Smooth(_) => self.predict_variance_jacobian_smooth(x),
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

pub trait ClusteredSurrogate: Clustered + Surrogate {}

impl ClusteredSurrogate for Moe {}

impl Moe {
    /// Constructor of mixture of experts parameters
    pub fn params() -> MoeParams<f64, Isaac64Rng> {
        MoeParams::new()
    }

    /// Recombination mode
    pub fn recombination(&self) -> Recombination<f64> {
        self.recombination
    }

    /// Recombination mode
    pub fn gmx(&self) -> &MultivariateNormal<f64> {
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
        self.gmx = MultivariateNormal::new(weights, means, covariances).unwrap();
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
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let probas = self.gmx.predict_probas(x);
        let mut preds = Array1::<f64>::zeros(x.nrows());

        Zip::from(&mut preds)
            .and(x.rows())
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

    pub fn predict_jacobian_smooth(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let probas = self.gmx.predict_probas(x);
        let der_probas = self.gmx.predict_probas_jacobian(x);
        let mut jac = Array2::<f64>::zeros((x.nrows(), x.ncols()));

        Zip::from(jac.rows_mut())
            .and(x.rows())
            .and(probas.rows())
            .and(der_probas.outer_iter())
            .for_each(|mut y, x, p, pprime| {
                let obs = x.insert_axis(Axis(0));
                let subpreds: Array1<f64> = self
                    .experts
                    .iter()
                    .map(|gp| gp.predict_values(&obs).unwrap()[[0, 0]])
                    .collect();
                let jacs: Vec<Array1<f64>> = self
                    .experts
                    .iter()
                    .map(|gp| gp.predict_jacobian(&obs).unwrap().row(0).to_owned())
                    .collect();

                let subpreds = subpreds.insert_axis(Axis(1));
                let mut subpreds_jac = Array2::zeros((self.experts.len(), x.len()));
                Zip::indexed(subpreds_jac.rows_mut()).for_each(|i, mut jc| jc.assign(&jacs[i]));

                let mut term1 = Array2::zeros((self.experts.len(), x.len()));
                Zip::from(term1.rows_mut())
                    .and(&p)
                    .and(subpreds_jac.rows())
                    .for_each(|mut t, p, der| t.assign(&(der.to_owned().mapv(|v| v * p))));
                let term1 = term1.sum_axis(Axis(0));

                let term2 = pprime.to_owned() * subpreds;
                let term2 = term2.sum_axis(Axis(0));

                y.assign(&(term1 + term2));
            });
        Ok(jac)
    }

    pub fn predict_variance_jacobian_smooth(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        self.predict_variance_jacobian_hard(x)
    }

    /// Predict outputs at a set of points `x` specified as (n, xdim) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// Then the expert of the cluster is used to predict output value.
    pub fn predict_values_hard(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let clustering = self.gmx.predict(x);
        debug!("Clustering {:?}", clustering);
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

    /// Predict variance at a set of points `x` specified as (n, xdim) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// Then the expert of the cluster is used to predict output value.
    pub fn predict_variances_hard(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let clustering = self.gmx.predict(x);
        debug!("Clustering {:?}", clustering);
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

    pub fn predict_jacobian_hard(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let mut jac = Array2::<f64>::zeros((x.nrows(), x.ncols()));
        let clustering = self.gmx.predict(x);
        Zip::from(jac.rows_mut())
            .and(x.rows())
            .and(&clustering)
            .for_each(|mut jac_i, xi, &c| {
                let x = xi.to_owned().insert_axis(Axis(0));
                let x_jac: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
                    self.experts[c].predict_jacobian(&x.view()).unwrap();
                jac_i.assign(&x_jac.column(0))
            });
        Ok(jac)
    }

    pub fn predict_variance_jacobian_hard(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let mut varjac = Array2::<f64>::zeros((x.nrows(), x.ncols()));
        let clustering = self.gmx.predict(x);
        Zip::from(varjac.rows_mut())
            .and(x.rows())
            .and(&clustering)
            .for_each(|mut varjac_i, xi, &c| {
                let x = xi.to_owned().insert_axis(Axis(0));
                let x_varjac: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> = self
                    .experts[c]
                    .predict_variance_jacobian(&x.view())
                    .unwrap();
                varjac_i.assign(&x_varjac.row(0))
            });
        Ok(varjac)
    }

    pub fn predict_values(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Result<Array2<f64>> {
        <Moe as Surrogate>::predict_values(self, &x.view())
    }

    pub fn predict_variances(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        <Moe as Surrogate>::predict_variances(self, &x.view())
    }

    pub fn predict_jacobian(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        <Moe as Surrogate>::predict_jacobian(self, &x.view())
    }

    pub fn predict_variance_jacobian(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        <Moe as Surrogate>::predict_variance_jacobian(self, &x.view())
    }

    #[cfg(feature = "persistent")]
    /// Load Moe from given json file.
    pub fn load(path: &str) -> Result<Box<Moe>> {
        let data = fs::read_to_string(path)?;
        let moe: Moe = serde_json::from_str(&data).unwrap();
        Ok(Box::new(moe))
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

impl<D: Data<Elem = f64>> PredictInplace<ArrayBase<D, Ix2>, Array2<f64>> for Moe {
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array2<f64>) {
        assert_eq!(
            x.nrows(),
            y.nrows(),
            "The number of data points must match the number of output targets."
        );

        let values = self.predict_values(x).expect("MoE prediction");
        *y = values;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array2<f64> {
        Array2::zeros((x.nrows(), self.output_dim()))
    }
}

struct MoeVariancePredictor<'a>(&'a Moe);
impl<'a, D: Data<Elem = f64>> PredictInplace<ArrayBase<D, Ix2>, Array2<f64>>
    for MoeVariancePredictor<'a>
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
            .expect("MoE variances prediction");
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
    use egobox_doe::{Lhs, SamplingMethod};
    use ndarray::{array, Array2, Zip};
    use ndarray_npy::write_npy;
    use ndarray_rand::rand;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand_isaac::Isaac64Rng;

    fn f_test_1d(x: &Array2<f64>) -> Array2<f64> {
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

    fn df_test_1d(x: &Array2<f64>) -> Array2<f64> {
        let mut y = Array2::zeros(x.dim());
        Zip::from(&mut y).and(x).for_each(|yi, &xi| {
            if xi < 0.4 {
                *yi = 2. * xi;
            } else if (0.4..0.8).contains(&xi) {
                *yi = 3.;
            } else {
                *yi = 10. * f64::cos(10. * xi);
            }
        });
        y
    }

    #[test]
    fn test_moe_hard() {
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = f_test_1d(&xt);
        let moe = Moe::params()
            .n_clusters(3)
            .regression_spec(RegressionSpec::CONSTANT)
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
            .recombination(Recombination::Hard)
            .with_rng(rng)
            .fit(&Dataset::new(xt, yt))
            .expect("MOE fitted");
        let obs = Array1::linspace(0., 1., 30).insert_axis(Axis(1));
        let preds = moe.predict_values(&obs).expect("MOE prediction");
        let dpreds = moe.predict_jacobian(&obs).expect("MOE jac prediction");
        println!("dpred = {}", dpreds);
        let test_dir = "target/tests";
        std::fs::create_dir_all(test_dir).ok();
        write_npy(format!("{}/obs_hard.npy", test_dir), &obs).expect("obs saved");
        write_npy(format!("{}/preds_hard.npy", test_dir), &preds).expect("preds saved");
        write_npy(format!("{}/dpreds_hard.npy", test_dir), &dpreds).expect("dpreds saved");
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
        let yt = f_test_1d(&xt);
        let ds = Dataset::new(xt, yt);
        let moe = Moe::params()
            .n_clusters(3)
            .recombination(Recombination::Smooth(Some(0.5)))
            .with_rng(rng.clone())
            .fit(&ds)
            .expect("MOE fitted");
        let obs = Array1::linspace(0., 1., 100).insert_axis(Axis(1));
        let preds = moe.predict_values(&obs).expect("MOE prediction");
        // Work for a kriging only atm
        // let dpreds = moe.predict_jacobian(&obs).expect("MOE jac prediction");
        println!("Smooth moe {}", moe);
        assert_abs_diff_eq!(
            0.37579, // true value = 0.37*0.37 = 0.1369
            moe.predict_values(&array![[0.37]]).unwrap()[[0, 0]],
            epsilon = 1e-3
        );
        let moe = Moe::params()
            .n_clusters(3)
            .recombination(Recombination::Smooth(None))
            .with_rng(rng.clone())
            .fit(&ds)
            .expect("MOE fitted");
        println!("Smooth moe {}", moe);

        let test_dir = "target/tests";
        std::fs::create_dir_all(test_dir).ok();
        write_npy(format!("{}/obs_smooth.npy", test_dir), &obs).expect("obs saved");
        write_npy(format!("{}/preds_smooth.npy", test_dir), &preds).expect("preds saved");
        // write_npy(format!("{}/dpreds_smooth.npy", test_dir), &dpreds).expect("dpreds saved");
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
        let yt = f_test_1d(&xt);
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
        let yt = f_test_1d(&xt);
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
        let yt = f_test_1d(&xt);
        let _moe = Moe::params()
            .n_clusters(3)
            .with_rng(rng)
            .fit(&Dataset::new(xt, yt))
            .expect("MOE fitted");
    }

    #[cfg(feature = "persistent")]
    #[test]
    fn test_save_load_moe() {
        let test_dir = "target/tests";
        std::fs::create_dir_all(test_dir).ok();

        let mut rng = Isaac64Rng::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = f_test_1d(&xt);
        let ds = Dataset::new(xt, yt);
        let moe = Moe::params()
            .n_clusters(3)
            .with_rng(rng)
            .fit(&ds)
            .expect("MOE fitted");
        let xtest = array![[0.6]];
        let y_expected = moe.predict_values(&xtest).unwrap();
        let filename = format!("{}/saved_moe.json", test_dir);
        moe.save(&filename).expect("MoE saving");
        let new_moe = Moe::load(&filename).expect("MoE loading");
        assert_abs_diff_eq!(
            y_expected,
            new_moe.predict_values(&xtest).unwrap(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_moe_jac_hard() {
        let rng = Isaac64Rng::seed_from_u64(0);
        let xt = Lhs::new(&array![[0., 1.]]).sample(100);
        let yt = f_test_1d(&xt);

        let moe = Moe::params()
            .n_clusters(3)
            .regression_spec(RegressionSpec::CONSTANT)
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
            .recombination(Recombination::Hard)
            .with_rng(rng)
            .fit(&Dataset::new(xt, yt))
            .expect("MOE fitted");
        let obs = Array1::linspace(0., 1., 50).insert_axis(Axis(1));
        let preds = moe.predict_values(&obs).expect("MOE prediction");
        let dpreds = moe.predict_jacobian(&obs).expect("MOE jac prediction");
        println!("dpred = {}", dpreds);

        let test_dir = "target/tests";
        std::fs::create_dir_all(test_dir).ok();
        write_npy(format!("{}/obs_hard.npy", test_dir), &obs).expect("obs saved");
        write_npy(format!("{}/preds_hard.npy", test_dir), &preds).expect("preds saved");
        write_npy(format!("{}/dpreds_hard.npy", test_dir), &dpreds).expect("dpreds saved");

        for _ in 0..20 {
            let x1: f64 = rand::random::<f64>();

            if (0.39 < x1 && x1 < 0.41) || (0.79 < x1 && x1 < 0.81) {
                // avoid testing hard on discoontinuity
                continue;
            } else {
                let h = 1e-4;
                let xtest = array![[x1]];

                let x = array![[x1], [x1 + h], [x1 - h]];
                let preds = moe.predict_jacobian(&x).unwrap();
                let fdiff = preds[[1, 0]] - preds[[1, 0]] / 2. * h;

                let jac = moe.predict_jacobian(&xtest).unwrap();
                let df = df_test_1d(&xtest);

                let err = if jac[[0, 0]] < 0.2 {
                    (jac[[0, 0]] - fdiff).abs()
                } else {
                    (jac[[0, 0]] - fdiff).abs() / jac[[0, 0]]
                };
                println!(
                    "Test predicted derivatives at {}: jac {}, true df {}, fdiff {}",
                    xtest, jac, df, fdiff
                );
                assert_abs_diff_eq!(err, 0.0, epsilon = 1e-1);
            }
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_moe_jac_smooth() {
        let rng = Isaac64Rng::seed_from_u64(0);
        let branin_data = array![
            [0.75995265131225453, 0.26079519790587313, 27.413856556469955],
            [0.84894004568915193, 0.80302739205770612, 124.08729377805207],
            [0.68726748138605631, 0.65346310415713116, 89.533227931057937],
            [0.44974452501493944, 0.34751899935449693, 10.878287247643591],
            [0.55682300995092937, 0.5849265431496905, 44.932884959302825],
            [0.88627740147717915, 0.62416594934584435, 64.830308367191918],
            [
                0.81775204764384801,
                0.042196154022979895,
                15.712266949196216
            ],
            [0.4177766615942754, 0.014380375247307521, 28.681202422398538],
            [0.72827825291615522, 0.38750347099448401, 41.148874722093616],
            [0.9925696885362203, 0.91677913627656205, 119.30186879624367],
            [0.50066411471583749, 0.96942357925531941, 139.66117288085078],
            [
                0.010996806446222557,
                0.64007942287598996,
                61.784705762774074
            ],
            [0.92503880783912951, 0.43076540443369776, 21.264868723459951],
            [0.60689384591767881, 0.42000489901770877, 26.182329221032123],
            [0.29565445928275541, 0.3005839001387956, 24.023634159735444],
            [0.97167048017882951, 0.73721527367823791, 71.968581810738641],
            [0.7752377959676553, 0.50612962709783971, 60.838284481314375],
            [0.25832467045551272, 0.074226708771860814, 60.93611965583608],
            [0.23737147394919283, 0.46611483768001949, 13.712846818250346],
            [0.14754191103105269, 0.11034908131009495, 96.713306093753971],
            [0.62868483371396366, 0.7952593023639507, 116.41116142984536],
            [0.87089829204248115, 0.16733416291664127, 8.8940612557011605],
            [0.10656730695541761, 0.53049331264739186, 25.2319994760445],
            [0.48472632135451527, 0.47615844132400142, 20.544947002406388],
            [0.53744687012384518, 0.095909332659171537, 1.237590217166554],
            [0.19434903593158279, 0.83986960523098408, 12.669879380619737],
            [0.31415479092439919, 0.70503097578565577, 36.074304011442244],
            [0.080369359089066103, 0.3570089932983076, 75.371239989968771],
            [0.35870550688156772, 0.55241395061831078, 27.170423440769184],
            [0.3379834771033397, 0.86527973630952937, 69.840929358660901],
            [0.91939579348244926, 0.28325971389978516, 7.3555005965177296],
            [0.21775934895998508, 0.99094248637867188, 41.124891683745233],
            [0.57932964524447805, 0.23100636146802719, 4.2985919393100787],
            [0.39667793013374686, 0.17990857948428171, 19.21599164774678],
            [0.65154294285455405, 0.92958359792609102, 169.27373704769877],
            [
                0.056765241174697265,
                0.21176965373303497,
                140.57488603163392
            ],
            [0.45782002694659368, 0.7504156346810722, 67.686983708699358],
            [0.70097955899174469, 0.13594741077482542, 17.690973760768447],
            [0.047490637953598841, 0.8969607941824036, 9.0862937348881641],
            [0.1544447325667799, 0.690502880810686, 2.0997165869714083],
        ];
        let xt = branin_data.slice(s![.., 0..2]).to_owned();
        let yt = branin_data.slice(s![.., 2..3]).to_owned();

        // for debug purpose set gmx
        let weights = array![0.65709529, 0.34290471];
        let means = array![[0.38952724, 0.48319967], [0.64382447, 0.50513467]];
        let covariances = array![
            [[0.05592868, -0.04013687], [-0.04013687, 0.07469386]],
            [[0.09693959, 0.07294977], [0.07294977, 0.0925502]]
        ];

        let moe = Moe::params()
            .n_clusters(2)
            .regression_spec(RegressionSpec::CONSTANT)
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
            .recombination(Recombination::Smooth(Some(1.)))
            .gmx(weights, means, covariances)
            .with_rng(rng)
            .fit(&Dataset::new(xt, yt))
            .expect("MOE fitted");

        for _ in 0..20 {
            let xa: f64 = rand::random::<f64>();
            let xb: f64 = rand::random::<f64>();
            let e = 1e-5;

            let x = array![
                [xa, xb],
                [xa + e, xb],
                [xa - e, xb],
                [xa, xb + e],
                [xa, xb - e]
            ];
            let y_predicted = moe.predict_values(&x).unwrap();
            let y_jacob = moe.predict_jacobian(&x).unwrap();

            let diff_g = (y_predicted[[1, 0]] - y_predicted[[2, 0]]) / (2. * e);
            let diff_d = (y_predicted[[3, 0]] - y_predicted[[4, 0]]) / (2. * e);

            let jac_rel_error1 = (y_jacob[[0, 0]] - diff_g).abs() / y_jacob[[0, 0]];
            assert_abs_diff_eq!(jac_rel_error1, 0.0, epsilon = 1e-3);

            let jac_rel_error2 = (y_jacob[[0, 1]] - diff_d).abs() / y_jacob[[0, 1]];
            assert_abs_diff_eq!(jac_rel_error2, 0.0, epsilon = 1e-3);
        }
    }
}
