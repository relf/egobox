#![allow(dead_code)]
use crate::Result;
use linfa::{dataset::WithLapack, dataset::WithoutLapack, traits::*, Float};
use ndarray::{s, Array, Array1, Array2, Array3, ArrayBase, Axis, Data, Ix2, Ix3, Zip};
use ndarray_linalg::{cholesky::*, triangular::*, Lapack, Scalar};
use ndarray_stats::QuantileExt;

pub struct GaussianMixture<F: Float> {
    weights: Array1<F>,
    means: Array2<F>,
    covariances: Array3<F>,
    precisions: Array3<F>,
    precisions_chol: Array3<F>,
    heaviside_factor: F,
}

impl<F: Float> Clone for GaussianMixture<F> {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.to_owned(),
            means: self.means.to_owned(),
            covariances: self.covariances.to_owned(),
            precisions: self.precisions.to_owned(),
            precisions_chol: self.precisions_chol.to_owned(),
            heaviside_factor: self.heaviside_factor,
        }
    }
}

impl<F: Float> GaussianMixture<F> {
    pub fn new(
        weights: Array1<F>,
        means: Array2<F>,
        covariances: Array3<F>,
    ) -> Result<GaussianMixture<F>> {
        let precisions_chol = Self::compute_precisions_cholesky_full(&covariances)?;
        let precisions = Self::compute_precisions_full(&precisions_chol);

        Ok(GaussianMixture {
            weights,
            means,
            covariances,
            precisions,
            precisions_chol,
            heaviside_factor: F::one(),
        })
    }

    pub fn weights(&self) -> &Array1<F> {
        &self.weights
    }

    pub fn means(&self) -> &Array2<F> {
        &self.means
    }

    // pub fn covariances(&self) -> &Array3<F> {
    //     &self.covariances
    // }

    // pub fn precisions(&self) -> &Array3<F> {
    //     &self.precisions
    // }

    pub fn heaviside_factor(&self) -> F {
        self.heaviside_factor
    }

    pub fn predict_probas<D: Data<Elem = F>>(&self, observations: &ArrayBase<D, Ix2>) -> Array2<F> {
        let (_, log_resp) = self.estimate_log_prob_resp(observations);
        log_resp.mapv(|v| v.exp())
    }

    pub fn with_heaviside_factor(mut self, heaviside_factor: F) -> Self {
        self.heaviside_factor = heaviside_factor;
        self
    }

    // NOT USED at the moment
    // fn estimate_gaussian_parameters<D: Data<Elem = F>>(
    //     observations: &ArrayBase<D, Ix2>,
    //     resp: &Array2<F>,
    //     reg_covar: F,
    // ) -> Result<(Array1<F>, Array2<F>, Array3<F>)> {
    //     let nk = resp.sum_axis(Axis(0));
    //     if nk.min().unwrap() < &(F::from(10.).unwrap() * F::epsilon()) {
    //         return Err(MoeError::EmptyCluster(format!(
    //           "Cluster #{} has no more point. Consider decreasing number of clusters or change initialization.",
    //           nk.argmin().unwrap() + 1
    //       )));
    //     }

    //     let nk2 = nk.to_owned().insert_axis(Axis(1));
    //     let means = resp.t().dot(observations) / nk2;
    //     let covariances =
    //         Self::estimate_gaussian_covariances_full(&observations, resp, &nk, &means, reg_covar);
    //     Ok((nk, means, covariances))
    // }

    // fn estimate_gaussian_covariances_full<D: Data<Elem = F>>(
    //     observations: &ArrayBase<D, Ix2>,
    //     resp: &Array2<F>,
    //     nk: &Array1<F>,
    //     means: &Array2<F>,
    //     reg_covar: F,
    // ) -> Array3<F> {
    //     let n_clusters = means.nrows();
    //     let n_features = means.ncols();
    //     let mut covariances = Array::zeros((n_clusters, n_features, n_features));
    //     for k in 0..n_clusters {
    //         let diff = observations - &means.row(k);
    //         let m = &diff.t() * &resp.index_axis(Axis(1), k);
    //         let mut cov_k = m.dot(&diff) / nk[k];
    //         cov_k.diag_mut().mapv_inplace(|x| x + reg_covar);
    //         covariances.slice_mut(s![k, .., ..]).assign(&cov_k);
    //     }
    //     covariances
    // }

    pub fn compute_precisions_cholesky_full<D: Data<Elem = F>>(
        covariances: &ArrayBase<D, Ix3>,
    ) -> Result<Array3<F>> {
        let n_clusters = covariances.shape()[0];
        let n_features = covariances.shape()[1];
        let mut precisions_chol = Array::zeros((n_clusters, n_features, n_features));
        for (k, covariance) in covariances.outer_iter().enumerate() {
            let cov_chol = covariance.with_lapack().cholesky(UPLO::Lower)?;
            let sol = cov_chol
                .solve_triangular(UPLO::Lower, Diag::NonUnit, &Array::eye(n_features))?
                .without_lapack();
            precisions_chol.slice_mut(s![k, .., ..]).assign(&sol.t());
        }
        Ok(precisions_chol)
    }

    fn compute_precisions_full<D: Data<Elem = F>>(
        precisions_chol: &ArrayBase<D, Ix3>,
    ) -> Array3<F> {
        let mut precisions = Array3::zeros(precisions_chol.dim());
        for (k, prec_chol) in precisions_chol.outer_iter().enumerate() {
            precisions
                .slice_mut(s![k, .., ..])
                .assign(&prec_chol.dot(&prec_chol.t()));
        }
        precisions
    }

    // Estimate log probabilities (log P(X)) and responsibilities for each sample.
    // Compute weighted log probabilities per component (log P(X)) and responsibilities
    // for each sample in X with respect to the current state of the model.
    fn estimate_log_prob_resp<D: Data<Elem = F>>(
        &self,
        observations: &ArrayBase<D, Ix2>,
    ) -> (Array1<F>, Array2<F>) {
        let weighted_log_prob = self.estimate_weighted_log_prob(&observations);
        let log_prob_norm = weighted_log_prob
            .mapv(|v| v.exp())
            .sum_axis(Axis(1))
            .mapv(|v| v.ln());
        let log_resp = weighted_log_prob - log_prob_norm.to_owned().insert_axis(Axis(1));
        (log_prob_norm, log_resp)
    }

    // Estimate weighted log probabilities for each samples wrt to the model
    fn estimate_weighted_log_prob<D: Data<Elem = F>>(
        &self,
        observations: &ArrayBase<D, Ix2>,
    ) -> Array2<F> {
        self.estimate_log_prob(&observations) + self.estimate_log_weights()
    }

    // Compute log probabilities for each samples wrt to the model which is gaussian
    fn estimate_log_prob<D: Data<Elem = F>>(&self, observations: &ArrayBase<D, Ix2>) -> Array2<F> {
        self.estimate_log_gaussian_prob(&observations)
    }

    // Compute the log LikelihoodComputation in case of the gaussian probabilities
    // log(P(X|Mean, Precision)) = -0.5*(d*ln(2*PI)-ln(det(Precision))-(X-Mean)^t.Precision.(X-Mean)
    fn estimate_log_gaussian_prob<D: Data<Elem = F>>(
        &self,
        observations: &ArrayBase<D, Ix2>,
    ) -> Array2<F> {
        let n_samples = observations.nrows();
        let n_features = observations.ncols();
        let means = self.means();
        let n_clusters = means.nrows();
        let factor = ndarray_rand::rand_distr::num_traits::Float::powf(
            self.heaviside_factor(),
            F::from(-0.5).unwrap(),
        );
        let precs = &self.precisions_chol * factor;
        // GmmCovarType = full
        // det(precision_chol) is half of det(precision)
        let log_det = Self::compute_log_det_cholesky_full(&precs, n_features);
        let mut log_prob: Array2<F> = Array::zeros((n_samples, n_clusters));
        Zip::indexed(means.rows())
            .and(precs.outer_iter())
            .for_each(|k, mu, prec_chol| {
                let diff = (&observations.to_owned() - &mu).dot(&prec_chol);
                log_prob
                    .slice_mut(s![.., k])
                    .assign(&diff.mapv(|v| v * v).sum_axis(Axis(1)))
            });
        log_prob.mapv(|v| {
            F::from(-0.5).unwrap()
                * (v + F::from(n_features as f64 * f64::ln(2. * std::f64::consts::PI)).unwrap())
        }) + log_det
    }

    fn compute_log_det_cholesky_full<D: Data<Elem = F>>(
        matrix_chol: &ArrayBase<D, Ix3>,
        n_features: usize,
    ) -> Array1<F> {
        let n_clusters = matrix_chol.shape()[0];
        let log_diags = &matrix_chol
            .to_owned()
            .into_shape((n_clusters, n_features * n_features))
            .unwrap()
            .slice(s![.., ..; n_features+1])
            .to_owned()
            .mapv(|v| v.ln());
        log_diags.sum_axis(Axis(1))
    }

    fn estimate_log_weights(&self) -> Array1<F> {
        self.weights().mapv(|v| v.ln())
    }

    /// Compute the weighted log probabilities for each sample.
    pub fn score_samples<D: Data<Elem = F>>(&self, x: &ArrayBase<D, Ix2>) -> Array1<F> {
        self.estimate_weighted_log_prob(x)
            .mapv(|v| v.exp())
            .sum_axis(Axis(1))
            .mapv(|v| v.ln())
    }

    // Compute the per-sample average log-likelihood of the given data X.
    pub fn score<D: Data<Elem = F>>(&self, x: &ArrayBase<D, Ix2>) -> F {
        self.score_samples(x).mean().unwrap()
    }

    /// Return the number of free parameters in the model.
    pub fn n_parameters(&self) -> usize {
        let (n_clusters, n_features) = (self.means.nrows(), self.means.ncols());
        let cov_params = n_clusters * n_features * (n_features + 1) / 2;
        let mean_params = n_features * n_clusters;
        (cov_params + mean_params + n_clusters - 1) as usize
    }

    /// Bayesian information criterion for the current model on the input X.
    /// The lower the better.
    pub fn bic<D: Data<Elem = F>>(&self, x: &ArrayBase<D, Ix2>) -> F {
        let n_samples = F::from(x.shape()[0]).unwrap();
        F::from(-2.).unwrap() * self.score(x) * n_samples
            + F::from(self.n_parameters()).unwrap() * n_samples.ln()
    }

    /// Akaike information criterion for the current model on the input X.
    /// The lower the better
    pub fn aic<D: Data<Elem = F>>(&self, x: &ArrayBase<D, Ix2>) -> F {
        let two = F::from(2.).unwrap();
        -two * (F::from(self.score(x)).unwrap()) * (F::from(x.shape()[0]).unwrap())
            + two * (F::from(self.n_parameters()).unwrap())
    }
}

impl<F: Float + Lapack + Scalar, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<usize>>
    for GaussianMixture<F>
{
    fn predict_inplace(&self, observations: &ArrayBase<D, Ix2>, targets: &mut Array1<usize>) {
        assert_eq!(
            observations.nrows(),
            targets.len(),
            "The number of data points must match the number of output targets."
        );

        let (_, log_resp) = self.estimate_log_prob_resp(observations);
        *targets = log_resp
            .mapv(Scalar::exp)
            .map_axis(Axis(1), |row| row.argmax().unwrap());
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<usize> {
        Array1::zeros(x.nrows())
    }
}

#[cfg(test)]
mod tests {
    // extern crate openblas_src;
    // extern crate intel_mkl_src;
    use super::*;
    use approx::assert_abs_diff_eq;
    use linfa::DatasetBase;
    use linfa_clustering::GaussianMixtureModel;
    use ndarray::{array, Array, Array2};
    use ndarray_linalg::solve::*;
    use ndarray_npy::write_npy;
    use ndarray_rand::rand::{rngs::SmallRng, SeedableRng};
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::RandomExt;
    use ndarray_stats::CorrelationExt;

    #[test]
    fn test_gaussian_mixture() {
        let weights = array![0.5, 0.5];
        let means = array![[0., 0.], [4., 4.]];
        let covs = array![[[3., 0.], [0., 3.]], [[3., 0.], [0., 3.]]];
        let gmix = GaussianMixture::new(weights, means, covs)
            .expect("Gaussian mixture creation failed")
            .with_heaviside_factor(0.99);
        let mut obs = Array2::from_elem((101, 2), 0.);
        Zip::from(obs.rows_mut())
            .and(&Array::linspace(0., 4., 101))
            .for_each(|mut o, &v| o.assign(&array![v, v]));
        let _preds = gmix.predict(&obs);
        let probas = gmix.predict_probas(&obs);
        write_npy("probes.npy", &obs).expect("failed to save");
        write_npy("probas.npy", &probas).expect("failed to save");
    }

    #[test]
    fn test_gaussian_mixture_aic_bic() {
        // Test the aic and bic criteria
        let mut rng = SmallRng::seed_from_u64(42);
        let (n_samples, n_features, n_components) = (50, 3, 2);
        let x = Array::random_using((n_samples, n_features), StandardNormal, &mut rng);
        write_npy("test_bic_aic.npy", &x).expect("failed to save");

        let dataset = DatasetBase::from(x.to_owned());
        let g = GaussianMixtureModel::params(n_components)
            .max_n_iterations(200)
            .with_rng(rng)
            .fit(&dataset)
            .expect("GMM fails");
        let gmx = GaussianMixture::new(
            g.weights().to_owned(),
            g.means().to_owned(),
            g.covariances().to_owned(),
        )
        .unwrap();
        // write_npy("test_bic_aic_weights.npy", g.weights()).expect("failed to save");
        // write_npy("test_bic_aic_means.npy", g.means()).expect("failed to save");
        // write_npy("test_bic_aic_precisions.npy", g.precisions()).expect("failed to save");
        // write_npy(
        //     "test_bic_aic_precisions_chol.npy",
        //     &GaussianMixture::compute_precisions_cholesky_full(g.covariances()).unwrap(),
        // )
        // .expect("failed to save");

        // True values checked against sklearn 0.24.1
        assert_abs_diff_eq!(489.5790028439929, gmx.bic(&x), epsilon = 1e-7);
        assert_abs_diff_eq!(453.2505657408581, gmx.aic(&x), epsilon = 1e-7);
        // standard gaussian entropy
        let sgh = 0.5
            * (f64::ln(x.t().cov(0.).unwrap().det().unwrap())
                + (n_features as f64) * (1. + f64::ln(2. * std::f64::consts::PI)));

        assert_abs_diff_eq!(4.193902320935888, sgh, epsilon = 1e-7);

        let aic = 2. * (n_samples as f64) * sgh + 2. * (gmx.n_parameters() as f64);
        let bic =
            2. * (n_samples as f64) * sgh + (n_samples as f64).ln() * (gmx.n_parameters() as f64);
        let bound = (n_features as f64) / (n_samples as f64).sqrt();

        assert_eq!(19, gmx.n_parameters());
        assert_abs_diff_eq!(493.71866919672357, bic, epsilon = 1e-7);
        assert_abs_diff_eq!(457.3902320935888, aic, epsilon = 1e-7);
        assert_abs_diff_eq!(0.4242640687119285, bound, epsilon = 1e-7);

        let v_aic = (gmx.aic(&x) - aic) / (n_samples as f64);
        let v_bic = (gmx.bic(&x) - bic) / (n_samples as f64);

        assert!(v_aic < bound);
        assert!(v_bic < bound);
    }
}
