#![allow(dead_code)]
use crate::algorithm::sort_by_cluster;
use crate::gaussian_mixture::GaussianMixture;
use crate::parameters::{CorrelationSpec, MoeParams, Recombination, RegressionSpec};
use log::debug;

use linfa::dataset::{Dataset, DatasetView};
use linfa::traits::{Fit, Predict};
use linfa_clustering::GaussianMixtureModel;
use ndarray::{concatenate, ArrayBase, Axis, Data, Ix2};
use ndarray_rand::rand::Rng;
use std::ops::Sub;

pub trait Clustered {
    fn n_clusters(&self) -> usize;
    fn recombination(&self) -> Recombination<f64>;

    fn to_clustering(&self) -> Clustering;
}

#[derive(Clone)]
pub struct Clustering {
    pub(crate) recombination: Recombination<f64>,
    pub(crate) gmx: GaussianMixture<f64>,
}

impl Clustering {
    pub fn new(gmx: GaussianMixture<f64>, recombination: Recombination<f64>) -> Self {
        Clustering { gmx, recombination }
    }

    pub fn recombination(&self) -> Recombination<f64> {
        self.recombination
    }
    pub fn gmx(&self) -> &GaussianMixture<f64> {
        &self.gmx
    }
}

fn mean(list: &[f64]) -> f64 {
    let sum: f64 = Iterator::sum(list.iter());
    sum / (list.len() as f64)
}

fn median(v: &[f64]) -> f64 {
    let mut list = v.to_vec();
    list.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = list.len();
    let mid = len / 2;
    if len % 2 == 0 {
        mean(&list[(mid - 1)..(mid + 1)])
    } else {
        list[mid]
    }
}

/// Find the best number of cluster thanks to cross validation
pub fn find_best_number_of_clusters<R: Rng + Clone>(
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    y: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    max_nb_clusters: usize,
    kpls_dim: Option<usize>,
    regression_spec: RegressionSpec,
    correlation_spec: CorrelationSpec,
    rng: R,
) -> (usize, Recombination<f64>) {
    let max_nb_clusters = if max_nb_clusters == 0 {
        (x.len() / 10) + 1
    } else {
        max_nb_clusters
    };
    let nx = x.ncols();
    let dataset: DatasetView<f64, f64> = DatasetView::new(x.view(), y.view());

    // Stock
    let mut mean_err_h: Vec<f64> = Vec::new();
    let mut mean_err_s: Vec<f64> = Vec::new();
    let mut median_err_h: Vec<f64> = Vec::new();
    let mut median_err_s: Vec<f64> = Vec::new();
    let mut nb_clusters_ok: Vec<usize> = Vec::new();

    // Init Output Loop
    let mut auxkh;
    let mut auxkph;
    let mut auxkpph;
    let mut auxks;
    let mut auxkps;
    let mut auxkpps;
    let mut ok1 = true;
    let mut ok2;
    let mut i = 0;
    let mut exit_ = false;

    let use_median = true;

    // Find error for each cluster
    while i < max_nb_clusters && !exit_ {
        let _kpls = nx > 9;

        let mut h_errors: Vec<f64> = Vec::new();
        let mut s_errors: Vec<f64> = Vec::new();
        let mut ok = true; // Say if this number of cluster is possible

        let n_clusters = i + 1;

        let xydata = Dataset::from(concatenate(Axis(1), &[x.view(), y.view()]).unwrap());
        let gmm = Box::new(
            GaussianMixtureModel::params(n_clusters)
                .n_runs(20)
                .with_rng(rng.clone())
                .fit(&xydata)
                .expect("Training data clustering"),
        );

        // Cross Validation
        if ok {
            for (train, valid) in dataset.fold(5).into_iter() {
                if let Ok(mixture) = MoeParams::default()
                    .n_clusters(n_clusters)
                    .regression_spec(regression_spec)
                    .correlation_spec(correlation_spec)
                    .kpls_dim(kpls_dim)
                    .gmm(Some(gmm.clone()))
                    .fit(&train)
                {
                    let xytrain =
                        concatenate(Axis(1), &[train.records().view(), train.targets.view()])
                            .unwrap();
                    let data_clustering = gmm.predict(&xytrain);
                    let clusters = sort_by_cluster(n_clusters, &xytrain, &data_clustering);
                    for cluster in clusters.iter().take(i + 1) {
                        // If there is at least 3 points
                        ok = ok && cluster.len() > 3
                    }
                    let actual = valid.targets();
                    let mixture = mixture.set_recombination(Recombination::Hard);
                    let h_error = if let Ok(pred) = mixture.predict_values(valid.records()) {
                        pred.sub(actual).mapv(|x| x * x).sum().sqrt()
                            / actual.mapv(|x| x * x).sum().sqrt()
                    } else {
                        ok = false;
                        1.0
                    };
                    h_errors.push(h_error);
                    let mixture = mixture.set_recombination(Recombination::Smooth(None));
                    let s_error = if let Ok(pred) = mixture.predict_values(valid.records()) {
                        pred.sub(actual).mapv(|x| x * x).sum().sqrt()
                            / actual.mapv(|x| x * x).sum().sqrt()
                    } else {
                        ok = false;
                        1.0
                    };
                    s_errors.push(s_error);
                } else {
                    ok = false;
                    s_errors.push(1.0);
                    h_errors.push(1.0);
                }
            }
        }

        // Stock median errors
        median_err_s.push(median(&s_errors));
        median_err_h.push(median(&h_errors));

        // Stock mean errors
        mean_err_s.push(mean(&s_errors));
        mean_err_h.push(mean(&h_errors));

        debug!(
            "Number of cluster: {} \
            | Possible: {} \
            | Error(hard): {} \
            | Error(smooth): {} \
            | Median (hard): {} \
            | Median (smooth): {}",
            i + 1,
            ok,
            mean_err_h[mean_err_h.len() - 1],
            mean_err_s[mean_err_s.len() - 1],
            median_err_h[median_err_h.len() - 1],
            median_err_s[median_err_s.len() - 1],
        );
        debug!("#######");

        // Stock possible numbers of cluster
        if ok {
            nb_clusters_ok.push(i);
        }

        if i > 3 {
            // Stop the search if the clustering can not be performed three times
            ok2 = ok1;
            ok1 = ok;
            exit_ = !ok && !ok1 && !ok2;
        }
        if use_median {
            // Stop the search if the median increases three times
            if i > 3 {
                auxkh = median_err_h[i - 2];
                auxkph = median_err_h[i - 1];
                auxkpph = median_err_h[i];
                auxks = median_err_s[i - 2];
                auxkps = median_err_s[i - 1];
                auxkpps = median_err_s[i];

                exit_ =
                    auxkph >= auxkh && auxkps >= auxks && auxkpph >= auxkph && auxkpps >= auxkps;
            }
        } else if i > 3 {
            // Stop the search if the means of errors increase three times
            auxkh = mean_err_h[i - 2];
            auxkph = mean_err_h[i - 1];
            auxkpph = mean_err_h[i];
            auxks = mean_err_s[i - 2];
            auxkps = mean_err_s[i - 1];
            auxkpps = mean_err_s[i];

            exit_ = auxkph >= auxkh && auxkps >= auxks && auxkpph >= auxkph && auxkpps >= auxkps;
        }

        i += 1;
    }
    // Find The best number of cluster
    let mut cluster_mse = 1;
    let mut cluster_mses = 1;
    let (mut min_err, mut min_errs) = if use_median {
        (
            median_err_h[nb_clusters_ok[0]],
            median_err_s[nb_clusters_ok[0]],
        )
    } else {
        (mean_err_h[nb_clusters_ok[0]], mean_err_s[nb_clusters_ok[0]])
    };

    debug!("Median errors hard: {:?}", median_err_h);
    debug!("Median errors soft: {:?}", median_err_s);
    debug!("Mean errors hard: {:?}", mean_err_h);
    debug!("Mean errors soft: {:?}", mean_err_s);

    for k in nb_clusters_ok {
        if use_median {
            if min_err > median_err_h[k] {
                min_err = median_err_h[k];
                cluster_mse = k + 1;
            }
            if min_errs > median_err_s[k] {
                min_errs = median_err_s[k];
                cluster_mses = k + 1;
            }
        } else {
            if min_err > mean_err_h[k] {
                min_err = mean_err_h[k];
                cluster_mse = k + 1;
            }
            if min_errs > mean_err_s[k] {
                min_errs = mean_err_s[k];
                cluster_mses = k + 1;
            }
        }
    }

    // Choose between hard or smooth recombination
    let cluster;
    let hardi;
    if use_median {
        if median_err_h[cluster_mse - 1] < median_err_s[cluster_mses - 1] {
            cluster = cluster_mse;
            hardi = true;
        } else {
            cluster = cluster_mses;
            hardi = false;
        }
    } else if mean_err_h[cluster_mse - 1] < mean_err_s[cluster_mses - 1] {
        cluster = cluster_mse;
        hardi = true;
    } else {
        cluster = cluster_mses;
        hardi = false;
    }

    let number_cluster = cluster;

    let method = if use_median {
        "| Method: Minimum of Median errors"
    } else {
        "| Method: Minimum of relative L2"
    };

    debug!("Optimal Number of cluster: {} {}", cluster, method);
    debug!("Recombination Hard: {}", hardi);
    let recomb = if hardi {
        Recombination::Hard
    } else {
        Recombination::Smooth(None)
    };
    (number_cluster, recomb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::Moe;
    use approx::assert_abs_diff_eq;
    use egobox_doe::{FullFactorial, Lhs, SamplingMethod};
    #[cfg(not(feature = "blas"))]
    use linfa_linalg::norm::*;
    use ndarray::{array, Array1, Array2, Axis, Zip};
    #[cfg(feature = "blas")]
    use ndarray_linalg::Norm;
    use ndarray_npy::write_npy;
    use ndarray_rand::rand::SeedableRng;
    use rand_isaac::Isaac64Rng;

    fn l1norm(x: &Array2<f64>) -> Array2<f64> {
        x.map_axis(Axis(1), |x| x.norm_l1()).insert_axis(Axis(1))
    }

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
    fn test_find_best_cluster_nb_1d() {
        let rng = Isaac64Rng::seed_from_u64(42);
        let doe = Lhs::new(&array![[0., 1.]]).with_rng(rng);
        //write_npy("doe.npy", &doe);
        let xtrain = doe.sample(50);
        //write_npy("xtrain.npy", &xtrain);
        let ytrain = function_test_1d(&xtrain);
        //write_npy("ytrain.npy", &ytrain);
        let rng = Isaac64Rng::seed_from_u64(42);
        let (nb_clusters, recombination) = find_best_number_of_clusters(
            &xtrain,
            &ytrain,
            5,
            None,
            RegressionSpec::ALL,
            CorrelationSpec::ALL,
            rng,
        );
        let moe = Moe::params()
            .n_clusters(nb_clusters)
            .recombination(recombination)
            .fit(&Dataset::new(xtrain, ytrain))
            .unwrap();
        let obs = Array1::linspace(0., 1., 100).insert_axis(Axis(1));
        let preds = moe.predict_values(&obs).unwrap();
        write_npy("best_obs.npy", &obs).expect("saved");
        write_npy("best_preds.npy", &preds).expect("saved");
        assert_eq!(3, nb_clusters);
    }

    #[test]
    fn test_find_best_cluster_nb_2d() {
        let doe = Lhs::new(&array![[-1., 1.], [-1., 1.]]);
        let xtrain = doe.sample(200);
        let ytrain = l1norm(&xtrain);
        let rng = Isaac64Rng::seed_from_u64(42);
        let (n_clusters, recomb) = find_best_number_of_clusters(
            &xtrain,
            &ytrain,
            5,
            None,
            RegressionSpec::ALL,
            CorrelationSpec::ALL,
            rng,
        );
        let valid = FullFactorial::new(&array![[-1., 1.], [-1., 1.]]);
        let xvalid = valid.sample(100);
        let yvalid = l1norm(&xvalid);
        let moe = Moe::params()
            .n_clusters(n_clusters)
            .recombination(recomb)
            .fit(&Dataset::new(xtrain, ytrain))
            .unwrap();
        let ypreds = moe.predict_values(&xvalid).expect("moe not fitted");
        debug!("{:?}", concatenate![Axis(1), ypreds, yvalid]);
        assert_abs_diff_eq!(&ypreds, &yvalid, epsilon = 1e-2);
    }
}
