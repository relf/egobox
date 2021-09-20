use crate::algorithm::{extract_part, sort_by_cluster, Moe};
use crate::gaussian_mixture::GaussianMixture;
use crate::parameters::{MoeParams, Recombination};
use linfa::dataset::{Dataset, DatasetView};
use linfa::traits::{Fit, Predict};
use linfa_clustering::GaussianMixtureModel;
use ndarray::{concatenate, s, Array, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_npy::write_npy;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_stats::Quantile1dExt;
use std::ops::Sub;

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
pub fn find_best_number_of_clusters<R: Rng + SeedableRng + Clone>(
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    y: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    max_nb_clusters: usize,
    rng: R,
) -> usize {
    let max_nb_clusters = if max_nb_clusters == 0 {
        (x.len() / 10) + 1
    } else {
        max_nb_clusters
    };
    //let max_nb_clusters = 3;
    let val = concatenate(Axis(1), &[x.view(), y.view()]).unwrap();
    let nx = x.ncols();

    let dataset: DatasetView<f64, f64> = DatasetView::new(x.view(), y.view());

    // Stock
    let mut errorih: Vec<f64> = Vec::new();
    let mut erroris: Vec<f64> = Vec::new();
    let mut posi: Vec<usize> = Vec::new();
    let mut b_ic: Vec<Vec<f64>> = Vec::new();
    let mut a_ic: Vec<Vec<f64>> = Vec::new();
    let mut error_h: Vec<Vec<f64>> = Vec::new();
    let mut error_s: Vec<Vec<f64>> = Vec::new();
    let mut median_eh: Vec<f64> = Vec::new();
    let mut median_es: Vec<f64> = Vec::new();

    // Init Output Loop
    let mut auxkh = 0.;
    let mut auxkph = 0.;
    let mut auxkpph = 0.;
    let mut auxks = 0.;
    let mut auxkps = 0.;
    let mut auxkpps = 0.;
    let mut ok1 = true;
    let mut ok2 = true;
    let mut i = 0;
    let mut exit_ = false;

    let use_median = true;

    // Find error for each cluster
    while i < max_nb_clusters && !exit_ {
        let kpls = nx > 9;

        let mut h_errors: Vec<f64> = Vec::new();
        let mut s_errors: Vec<f64> = Vec::new();
        let mut bic_c: Vec<f64> = Vec::new();
        let mut aic_c: Vec<f64> = Vec::new();
        let mut ok = true; // Say if this number of cluster is possible

        let n_clusters = i + 1;

        let xydata = Dataset::from(concatenate(Axis(1), &[x.view(), y.view()]).unwrap());
        let gmm = Box::new(
            GaussianMixtureModel::params(n_clusters)
                .with_n_runs(20)
                //.with_reg_covariance(1e-6)
                .with_rng(rng.clone())
                .fit(&xydata)
                .expect("Training data clustering"),
        );

        // Cross Validation
        if ok {
            let mut k = 0;
            for (train, valid) in dataset.fold(5).into_iter() {
                k = k + 1;
                if let Ok(mixture) = Moe::params(n_clusters)
                    //.set_kpls_dim(Some(1))
                    .set_gmm(Some(gmm.clone()))
                    .fit(&train.records(), &train.targets())
                {
                    let xytrain =
                        concatenate(Axis(1), &[train.records().view(), train.targets.view()])
                            .unwrap();
                    let data_clustering = gmm.predict(&xytrain);
                    let clusters =
                        sort_by_cluster(n_clusters, &xytrain, &data_clustering, rng.clone());
                    let gmx = GaussianMixture::new(
                        gmm.weights().to_owned(),
                        gmm.means().to_owned(),
                        gmm.covariances().to_owned(),
                    )
                    .unwrap();

                    let records = valid.records();
                    let targets = valid.targets();
                    let valid_set =
                        concatenate(Axis(1), &[records.view(), targets.view()]).unwrap();
                    bic_c.push(gmx.bic(&valid_set));
                    aic_c.push(gmx.aic(&valid_set));
                    for j in 0..i + 1 {
                        // If there is at least one point
                        ok = clusters[j].len() > 3
                    }
                    let actual = valid.targets();
                    let h_error =
                        if let Ok(pred) = mixture.predict_hard(&valid.records().to_owned()) {
                            // write_npy(format!("valid_x_{}_{}.npy", n_clusters, k), valid.records())
                            //     .expect("valid x saved");
                            // write_npy(format!("valid_y_{}_{}.npy", n_clusters, k), actual)
                            //     .expect("valid y saved");
                            // write_npy(format!("pred_{}_{}.npy", n_clusters, k), &pred)
                            //     .expect("pred saved");
                            pred.sub(actual).mapv(|x| x * x).sum().sqrt()
                                / actual.mapv(|x| x * x).sum().sqrt()
                        } else {
                            1.0
                        };
                    h_errors.push(h_error);
                    let s_error =
                        if let Ok(pred) = mixture.predict_smooth(&valid.records().to_owned()) {
                            pred.sub(actual).mapv(|x| x * x).sum().sqrt()
                                / actual.mapv(|x| x * x).sum().sqrt()
                        } else {
                            1.0
                        };
                    s_errors.push(s_error);
                } else {
                    s_errors.push(1.0);
                    h_errors.push(1.0);
                }
            }
        }

        // Stock for box plot
        b_ic.push(bic_c);
        a_ic.push(aic_c);
        error_s.push(s_errors.to_owned());
        error_h.push(h_errors.to_owned());

        // Stock median
        median_es.push(median(&s_errors));
        median_eh.push(median(&h_errors));

        // Stock possible numbers of cluster
        if ok {
            posi.push(i);
        }

        // Stock mean errors
        erroris.push(mean(&s_errors));
        errorih.push(mean(&h_errors));

        println!(
            "Number of cluster: {} \
            | Possible: {} \
            | Error(hard): {} \
            | Error(smooth): {} \
            | Median (hard): {} \
            | Median (smooth): {}",
            i + 1,
            ok,
            mean(&h_errors),
            mean(&s_errors),
            median(&h_errors),
            median(&s_errors),
        );
        println!("#######");

        if i > 3 {
            // Stop the search if the clustering can not be performed three times
            ok2 = ok1;
            ok1 = ok;
            exit_ = !ok && !ok1 && !ok2;
        }
        if use_median {
            // Stop the search if the median increases three times
            if i > 3 {
                auxkh = median_eh[i - 2];
                auxkph = median_eh[i - 1];
                auxkpph = median_eh[i];
                auxks = median_es[i - 2];
                auxkps = median_es[i - 1];
                auxkpps = median_es[i];

                exit_ =
                    auxkph >= auxkh && auxkps >= auxks && auxkpph >= auxkph && auxkpps >= auxkps;
            }
        } else {
            if i > 3 {
                // Stop the search if the means of errors increase three times
                auxkh = errorih[i - 2];
                auxkph = errorih[i - 1];
                auxkpph = errorih[i];
                auxks = erroris[i - 2];
                auxkps = erroris[i - 1];
                auxkpps = erroris[i];

                exit_ =
                    auxkph >= auxkh && auxkps >= auxks && auxkpph >= auxkph && auxkpps >= auxkps;
            }
        }
        i += 1;
    }
    // Find The best number of cluster
    let mut cluster_mse = 1;
    let mut cluster_mses = 1;
    let (mut min_err, mut min_errs) = if use_median {
        (median_eh[posi[0]], median_es[posi[0]])
    } else {
        (errorih[posi[0]], erroris[posi[0]])
    };

    for k in posi {
        if use_median {
            if min_err > median_eh[k] {
                min_err = median_eh[k];
                cluster_mse = k + 1;
            }
            if min_errs > median_es[k] {
                min_errs = median_es[k];
                cluster_mses = k + 1;
            }
        } else {
            if min_err > errorih[k] {
                min_err = errorih[k];
                cluster_mse = k + 1;
            }
            if min_errs > erroris[k] {
                min_errs = erroris[k];
                cluster_mses = k + 1;
            }
        }
    }

    // Choose between hard or smooth recombination
    let cluster;
    let hardi;
    if use_median {
        if median_eh[cluster_mse - 1] < median_es[cluster_mses - 1] {
            cluster = cluster_mse;
            hardi = true;
        } else {
            cluster = cluster_mses;
            hardi = false;
        }
    } else {
        if errorih[cluster_mse - 1] < erroris[cluster_mses - 1] {
            cluster = cluster_mse;
            hardi = true;
        } else {
            cluster = cluster_mses;
            hardi = false;
        }
    }

    let number_cluster = cluster;

    let method = if use_median {
        "| Method: Minimum of Median errors"
    } else {
        "| Method: Minimum of relative L2"
    };

    println!("Optimal Number of cluster: {} {}", cluster, method);
    println!("Recombination Hard: {}", hardi);

    number_cluster
}

#[cfg(test)]
mod test {
    use super::*;
    use doe::{SamplingMethod, LHS};
    use ndarray::{array, Array2, Axis, Zip};
    use ndarray_linalg::norm::*;
    use ndarray_rand::rand::SeedableRng;
    use rand_isaac::Isaac64Rng;

    fn l1norm(x: &Array2<f64>) -> Array2<f64> {
        x.map_axis(Axis(1), |x| x.norm_l1()).insert_axis(Axis(1))
    }

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
    fn test_find_best_cluster_nb_1d() {
        let doe = LHS::new(&array![[0., 1.]]);
        //write_npy("doe.npy", &doe);
        let xtrain = doe.sample(100);
        //write_npy("xtrain.npy", &xtrain);
        let ytrain = function_test_1d(&xtrain);
        //write_npy("ytrain.npy", &ytrain);
        let rng = Isaac64Rng::seed_from_u64(42);
        let nb_clusters = find_best_number_of_clusters(&xtrain, &ytrain, 5, rng);
        assert_eq!(3, nb_clusters);
    }

    #[test]
    fn test_find_best_cluster_nb_2d() {
        let doe = LHS::new(&array![[-1., 1.], [-1., 1.]]);
        let xtrain = doe.sample(200);
        let ytrain = l1norm(&xtrain);
        let rng = Isaac64Rng::seed_from_u64(42);
        let nb_clusters = find_best_number_of_clusters(&xtrain, &ytrain, 5, rng);
        assert_eq!(4, nb_clusters);
    }
}