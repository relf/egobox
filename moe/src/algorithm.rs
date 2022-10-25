//use super::gaussian_mixture::GaussianMixture;
use super::gaussian_mixture::GaussianMixture;
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
            GaussianMixture::new(weights, means, covariances)?.heaviside_factor(factor)
        };

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
/// `gmx` is used to get the probability of x to belongs to one cluster
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
#[cfg_attr(feature = "persistent", derive(Serialize, Deserialize))]
pub struct Moe {
    /// The mode of recombination to get the output prediction from experts prediction
    recombination: Recombination<f64>,
    /// The list of the best experts trained on each cluster
    experts: Vec<Box<dyn Surrogate>>,
    /// The gaussian mixture allowing to predict cluster responsabilities for a given point
    gmx: GaussianMixture<f64>,
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

    fn predict_derivatives(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        match self.recombination {
            Recombination::Hard => self.predict_derivatives_hard(x),
            Recombination::Smooth(_) => self.predict_derivatives_smooth(x),
        }
    }

    fn predict_variance_derivatives(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        match self.recombination {
            Recombination::Hard => self.predict_variance_derivatives_hard(x),
            Recombination::Smooth(_) => self.predict_variance_derivatives_smooth(x),
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

/// A trait for surrogates using clustering
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

    /// Predict derivatives of the output at a set of points `x` specified as (n, nx) matrix.
    /// Return derivatives as a (n, nx) matrix where the ith row contain the partial derivatives of
    /// of the output wrt the nx components of `x` valued at the ith x point.
    /// The smooth recombination of each cluster expert responsability is used to get the result.
    pub fn predict_derivatives_smooth(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let probas = self.gmx.predict_probas(x);
        let probas_drv = self.gmx.predict_probas_derivatives(x);
        let mut drv = Array2::<f64>::zeros((x.nrows(), x.ncols()));

        Zip::from(drv.rows_mut())
            .and(x.rows())
            .and(probas.rows())
            .and(probas_drv.outer_iter())
            .for_each(|mut y, x, p, pprime| {
                let x = x.insert_axis(Axis(0));
                let preds: Array1<f64> = self
                    .experts
                    .iter()
                    .map(|gp| gp.predict_values(&x).unwrap()[[0, 0]])
                    .collect();
                let drvs: Vec<Array1<f64>> = self
                    .experts
                    .iter()
                    .map(|gp| gp.predict_derivatives(&x).unwrap().row(0).to_owned())
                    .collect();

                let preds = preds.insert_axis(Axis(1));
                let mut preds_drv = Array2::zeros((self.experts.len(), x.len()));
                Zip::indexed(preds_drv.rows_mut()).for_each(|i, mut jc| jc.assign(&drvs[i]));

                let mut term1 = Array2::zeros((self.experts.len(), x.len()));
                Zip::from(term1.rows_mut())
                    .and(&p)
                    .and(preds_drv.rows())
                    .for_each(|mut t, p, der| t.assign(&(der.to_owned().mapv(|v| v * p))));
                let term1 = term1.sum_axis(Axis(0));

                let term2 = pprime.to_owned() * preds;
                let term2 = term2.sum_axis(Axis(0));

                y.assign(&(term1 + term2));
            });
        Ok(drv)
    }

    /// Predict derivatives of the variance at a set of points `x` specified as (n, nx) matrix.
    /// Return derivatives as a (n, nx) matrix where the ith row contain the partial derivatives of
    /// of the vairance wrt the nx components of `x` valued at the ith x point.
    /// The smooth recombination of each cluster expert responsability is used to get the result.
    pub fn predict_variance_derivatives_smooth(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let probas = self.gmx.predict_probas(x);
        println!("probas={}", probas);
        let probas_drv = self.gmx.predict_probas_derivatives(x);
        println!("der_probas={}", probas_drv);

        let mut drv = Array2::<f64>::zeros((x.nrows(), x.ncols()));

        Zip::from(drv.rows_mut())
            .and(x.rows())
            .and(probas.rows())
            .and(probas_drv.outer_iter())
            .for_each(|mut y, xi, p, pprime| {
                let xii = xi.insert_axis(Axis(0));
                let preds: Array1<f64> = self
                    .experts
                    .iter()
                    .map(|gp| gp.predict_variances(&xii).unwrap()[[0, 0]])
                    .collect();
                let drvs: Vec<Array1<f64>> = self
                    .experts
                    .iter()
                    .map(|gp| {
                        gp.predict_variance_derivatives(&xii)
                            .unwrap()
                            .row(0)
                            .to_owned()
                    })
                    .collect();

                let preds = preds.insert_axis(Axis(1));
                let mut preds_drv = Array2::zeros((self.experts.len(), xi.len()));
                Zip::indexed(preds_drv.rows_mut()).for_each(|i, mut jc| jc.assign(&drvs[i]));

                println!("deriv = {}", preds_drv);
                println!("preds = {}", preds);

                let mut term1 = Array2::zeros((self.experts.len(), xi.len()));
                Zip::from(term1.rows_mut())
                    .and(&p)
                    .and(preds_drv.rows())
                    .for_each(|mut t, p, der| t.assign(&(der.to_owned().mapv(|v| v * p * p))));
                let term1 = term1.sum_axis(Axis(0));

                let term2 = (p.to_owned() * pprime * preds).mapv(|v| 2. * v);
                let term2 = term2.sum_axis(Axis(0));

                y.assign(&(term1 + term2));
            });

        Ok(drv)
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

    /// Predict variance at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// The expert of the cluster is used to predict variance value.
    /// Returns the variances as a (n, 1) column vector
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

    /// Predict derivatives of the output at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// The expert of the cluster is used to predict variance value.
    /// Returns derivatives as a (n, nx) matrix where the ith row contain the partial derivatives of
    /// of the output wrt the nx components of `x` valued at the ith x point.
    pub fn predict_derivatives_hard(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let mut drv = Array2::<f64>::zeros((x.nrows(), x.ncols()));
        let clustering = self.gmx.predict(x);
        Zip::from(drv.rows_mut())
            .and(x.rows())
            .and(&clustering)
            .for_each(|mut drv_i, xi, &c| {
                let x = xi.to_owned().insert_axis(Axis(0));
                let x_drv: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
                    self.experts[c].predict_derivatives(&x.view()).unwrap();
                drv_i.assign(&x_drv.column(0))
            });
        Ok(drv)
    }

    /// Predict derivatives of the variances at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// The expert of the cluster is used to predict variance value.
    /// Returns derivatives as a (n, nx) matrix where the ith row contain the partial derivatives of
    /// of the output wrt the nx components of `x` valued at the ith x point.
    pub fn predict_variance_derivatives_hard(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let mut vardrv = Array2::<f64>::zeros((x.nrows(), x.ncols()));
        let clustering = self.gmx.predict(x);
        Zip::from(vardrv.rows_mut())
            .and(x.rows())
            .and(&clustering)
            .for_each(|mut vardrv_i, xi, &c| {
                let x = xi.to_owned().insert_axis(Axis(0));
                let x_vardrv: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> = self
                    .experts[c]
                    .predict_variance_derivatives(&x.view())
                    .unwrap();
                vardrv_i.assign(&x_vardrv.row(0))
            });
        Ok(vardrv)
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

    pub fn predict_derivatives(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        <Moe as Surrogate>::predict_derivatives(self, &x.view())
    }

    pub fn predict_variance_derivatives(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        <Moe as Surrogate>::predict_variance_derivatives(self, &x.view())
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

/// Adaptator to implement `linfa::Predict` for variance prediction
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
    use argmin_testfunctions::rosenbrock;
    use egobox_doe::{Lhs, SamplingMethod};
    use ndarray::{array, Array, Array2, Zip};
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
        let x = Array1::linspace(0., 1., 30).insert_axis(Axis(1));
        let preds = moe.predict_values(&x).expect("MOE prediction");
        let dpreds = moe.predict_derivatives(&x).expect("MOE drv prediction");
        println!("dpred = {}", dpreds);
        let test_dir = "target/tests";
        std::fs::create_dir_all(test_dir).ok();
        write_npy(format!("{}/x_hard.npy", test_dir), &x).expect("x saved");
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
        let x = Array1::linspace(0., 1., 100).insert_axis(Axis(1));
        let preds = moe.predict_values(&x).expect("MOE prediction");
        // Work for a kriging only atm
        // let dpreds = moe.predict_derivatives(&x).expect("MOE drv prediction");
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
        write_npy(format!("{}/x_smooth.npy", test_dir), &x).expect("x saved");
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
        let x = Array1::linspace(0., 1., 100).insert_axis(Axis(1));
        let variances = moe.predict_variances(&x).expect("MOE variances prediction");
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
    fn test_moe_drv_hard() {
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
        let x = Array1::linspace(0., 1., 50).insert_axis(Axis(1));
        let preds = moe.predict_values(&x).expect("MOE prediction");
        let dpreds = moe.predict_derivatives(&x).expect("MOE drv prediction");
        println!("dpred = {}", dpreds);

        let test_dir = "target/tests";
        std::fs::create_dir_all(test_dir).ok();
        write_npy(format!("{}/x_hard.npy", test_dir), &x).expect("x saved");
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
                let preds = moe.predict_derivatives(&x).unwrap();
                let fdiff = preds[[1, 0]] - preds[[1, 0]] / 2. * h;

                let drv = moe.predict_derivatives(&xtest).unwrap();
                let df = df_test_1d(&xtest);

                let err = if drv[[0, 0]] < 0.2 {
                    (drv[[0, 0]] - fdiff).abs()
                } else {
                    (drv[[0, 0]] - fdiff).abs() / drv[[0, 0]]
                };
                println!(
                    "Test predicted derivatives at {}: drv {}, true df {}, fdiff {}",
                    xtest, drv, df, fdiff
                );
                assert_abs_diff_eq!(err, 0.0, epsilon = 1e-1);
            }
        }
    }

    fn norm1(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.abs())
            .sum_axis(Axis(1))
            .insert_axis(Axis(1))
            .to_owned()
    }

    fn rosenb(x: &Array2<f64>) -> Array2<f64> {
        let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
        Zip::from(y.rows_mut())
            .and(x.rows())
            .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec(), 1., 100.)]));
        y
    }

    #[allow(clippy::excessive_precision)]
    fn test_variance_derivatives(f: fn(&Array2<f64>) -> Array2<f64>) {
        let rng = Isaac64Rng::seed_from_u64(0);
        let xt = egobox_doe::FullFactorial::new(&array![[-1., 1.], [-1., 1.]]).sample(100);
        let yt = f(&xt);

        let moe = Moe::params()
            .n_clusters(2)
            .regression_spec(RegressionSpec::CONSTANT)
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
            .recombination(Recombination::Smooth(Some(1.)))
            .with_rng(rng)
            .fit(&Dataset::new(xt, yt))
            .expect("MOE fitted");

        for _ in 0..20 {
            let mut rng = Isaac64Rng::seed_from_u64(42);
            let x = Array::random_using((2,), Uniform::new(0., 1.), &mut rng);
            let xa: f64 = x[0];
            let xb: f64 = x[1];
            let e = 1e-4;

            println!("Test derivatives at [{}, {}]", xa, xb);

            let x = array![
                [xa, xb],
                [xa + e, xb],
                [xa - e, xb],
                [xa, xb + e],
                [xa, xb - e]
            ];
            let y_pred = moe.predict_values(&x).unwrap();
            let y_deriv = moe.predict_derivatives(&x).unwrap();

            let diff_g = (y_pred[[1, 0]] - y_pred[[2, 0]]) / (2. * e);
            let diff_d = (y_pred[[3, 0]] - y_pred[[4, 0]]) / (2. * e);

            assert_rel_or_abs_error(y_deriv[[0, 0]], diff_g);
            assert_rel_or_abs_error(y_deriv[[0, 1]], diff_d);

            let y_pred = moe.predict_variances(&x).unwrap();
            let y_deriv = moe.predict_variance_derivatives(&x).unwrap();

            let diff_g = (y_pred[[1, 0]] - y_pred[[2, 0]]) / (2. * e);
            let diff_d = (y_pred[[3, 0]] - y_pred[[4, 0]]) / (2. * e);

            assert_rel_or_abs_error(y_deriv[[0, 0]], diff_g);
            assert_rel_or_abs_error(y_deriv[[0, 1]], diff_d);
        }
    }

    fn assert_rel_or_abs_error(y_deriv: f64, fdiff: f64) {
        println!("analytic deriv = {}, fdiff = {}", y_deriv, fdiff);
        if fdiff.abs() < 1e-2 {
            assert_abs_diff_eq!(y_deriv, 0.0, epsilon = 1e-1); // check absolute when close to zero
        } else {
            let drv_rel_error1 = (y_deriv - fdiff).abs() / fdiff; // check relative
            assert_abs_diff_eq!(drv_rel_error1, 0.0, epsilon = 1e-1);
        }
    }

    #[test]
    fn test_moe_var_deriv_norm1() {
        test_variance_derivatives(norm1);
    }
    #[test]
    fn test_moe_var_deriv_rosenb() {
        test_variance_derivatives(rosenb);
    }
}
