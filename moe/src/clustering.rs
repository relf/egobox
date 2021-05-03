use crate::algorithm::{extract_part, sort_by_cluster, Moe};
use crate::gaussian_mixture::GaussianMixture;
use crate::parameters::MoeParams;
use linfa_clustering::GaussianMixtureModel;
use ndarray::{concatenate, s, Array, Array2, ArrayBase, Axis, Data, Ix2};
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
    rng: R,
) -> usize {
    let max_nb_clusters = x.len() / 10 + 1;
    let val = concatenate(Axis(1), &[x.view(), y.view()]).unwrap();
    let nx = x.ncols();

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
    let mut auxkh = 0;
    let mut auxkph = 0;
    let mut auxkpph = 0;
    let mut auxks = 0;
    let mut auxkps = 0;
    let mut auxkpps = 0;
    let mut ok1 = true;
    let mut ok2 = true;
    let mut i = 0;
    let mut exit_ = true;

    // Find error for each cluster
    while i < max_nb_clusters && exit_ {
        let kpls = nx > 9;

        let mut h_errors: Vec<f64> = Vec::new();
        let mut s_errors: Vec<f64> = Vec::new();
        let mut bic_c: Vec<f64> = Vec::new();
        let mut aic_c: Vec<f64> = Vec::new();
        let mut ok = true; // Say if this number of cluster is possible

        // Cross Validation
        for c in 0..5 {
            if ok {
                // Create training and test samples
                let (val_train, val_test) = extract_part(&val, 5);
                let xtrain = val_train.slice(s![.., ..nx]).to_owned();
                let ytrain = val_train.slice(s![.., nx..]).to_owned();

                // Create the MoE for the cross validation
                let n_clusters = i + 1;

                let mixture = Moe::params(n_clusters)
                    .set_kpls_dim(Some(1))
                    .fit(&xtrain, &ytrain)
                    .unwrap();

                let (clusters, gmm) = sort_by_cluster(n_clusters, &val_train, rng.clone());
                let gmx = GaussianMixture::new(
                    gmm.weights().to_owned(),
                    gmm.means().to_owned(),
                    gmm.covariances().to_owned(),
                )
                .unwrap();

                bic_c.push(gmx.bic(&val_test));
                aic_c.push(gmx.aic(&val_test));

                for c in clusters {
                    ok = ok && (c.len() >= 4);
                }

                if ok {
                    let actual = val_test.slice(s![.., nx..]);
                    let h_error = if let Ok(pred) =
                        mixture.predict_hard(&val_test.slice(s![.., 0..nx]).to_owned())
                    {
                        pred.sub(&actual).mapv(|x| x * x).sum().sqrt()
                            / actual.mapv(|x| x * x).sum().sqrt()
                    } else {
                        1.0
                    };
                    h_errors.push(h_error);

                    let s_error = if let Ok(pred) =
                        mixture.predict_smooth(&val_test.slice(s![.., 0..nx]).to_owned())
                    {
                        pred.sub(&actual).mapv(|x| x * x).sum().sqrt()
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
        errorih.push(mean(&h_errors));
        erroris.push(mean(&s_errors));

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

        //       if i > 3:
        //           # Stop the search if the clustering can not be performed three
        //           # times
        //           ok2 = ok1
        //           ok1 = ok
        //           if ok1 is False and ok is False and ok2 is False:
        //               exit_ = False
        //       if median:
        //           # Stop the search if the median increases three times
        //           if i > 3:
        //               auxkh = median_eh[i - 2]
        //               auxkph = median_eh[i - 1]
        //               auxkpph = median_eh[i]
        //               auxks = median_es[i - 2]
        //               auxkps = median_es[i - 1]
        //               auxkpps = median_es[i]

        //               if (
        //                   auxkph >= auxkh
        //                   and auxkps >= auxks
        //                   and auxkpph >= auxkph
        //                   and auxkpps >= auxkps
        //               ):
        //                   exit_ = False
        //       else:
        //           if i > 3:
        //               # Stop the search if the means of errors increase three
        //               # times
        //               auxkh = errori[i - 2]
        //               auxkph = errori[i - 1]
        //               auxkpph = errori[i]
        //               auxks = erroris[i - 2]
        //               auxkps = erroris[i - 1]
        //               auxkpps = erroris[i]

        //               if (
        //                   auxkph >= auxkh
        //                   and auxkps >= auxks
        //                   and auxkpph >= auxkph
        //                   and auxkpps >= auxkps
        //               ):
        //                   exit_ = False
        i += 1;
    }
    //   # Find The best number of cluster
    //   cluster_mse = 1
    //   cluster_mses = 1
    //   if median:
    //       min_err = median_eh[posi[0]]
    //       min_errs = median_es[posi[0]]
    //   else:
    //       min_err = errori[posi[0]]
    //       min_errs = erroris[posi[0]]
    //   for k in posi:
    //       if median:
    //           if min_err > median_eh[k]:
    //               min_err = median_eh[k]
    //               cluster_mse = k + 1
    //           if min_errs > median_es[k]:
    //               min_errs = median_es[k]
    //               cluster_mses = k + 1
    //       else:
    //           if min_err > errori[k]:
    //               min_err = errori[k]
    //               cluster_mse = k + 1
    //           if min_errs > erroris[k]:
    //               min_errs = erroris[k]
    //               cluster_mses = k + 1

    //   # Choose between hard or smooth recombination
    //   if median:
    //       if median_eh[cluster_mse - 1] < median_es[cluster_mses - 1]:
    //           cluster = cluster_mse
    //           hardi = True

    //       else:
    //           cluster = cluster_mses
    //           hardi = False
    //   else:
    //       if errori[cluster_mse - 1] < erroris[cluster_mses - 1]:
    //           cluster = cluster_mse
    //           hardi = True
    //       else:
    //           cluster = cluster_mses
    //           hardi = False

    //   self.number_cluster = cluster

    //   if plot:
    //       self._plot_number_cluster(
    //           posi,
    //           errori,
    //           erroris,
    //           median_eh,
    //           median_es,
    //           b_ic,
    //           a_ic,
    //           error_h,
    //           error_s,
    //           cluster,
    //       )

    //   if median:
    //       method = "| Method: Minimum of Median errors"
    //   else:
    //       method = "| Method: Minimum of relative L2"

    //   tps_fin = time.process_time()

    //   if detail:  # pragma: no cover
    //       print("Optimal Number of cluster: ", cluster, method)
    //       print("Recombination Hard: ", hardi)
    //       print("Computation time (cluster): ", self.format_time(tps_fin - tps_dep))

    //self.nb_clusters()
    0
}

#[cfg(test)]
mod test {
    use super::*;
    use doe::{FullFactorial, SamplingMethod};
    use ndarray::{array, Array2, Axis};
    use ndarray_linalg::norm::*;
    use ndarray_rand::rand::SeedableRng;
    use rand_isaac::Isaac64Rng;

    fn l2norm(x: &Array2<f64>) -> Array2<f64> {
        x.map_axis(Axis(1), |x| x.norm_l2()).insert_axis(Axis(1))
    }

    // #[test]
    // fn test_find_best_cluster_nb() {
    //     let doe = FullFactorial::new(&array![[-1., 1.], [-1., 1.]]);
    //     let xtrain = doe.sample(200);
    //     println!("{:?}", xtrain);
    //     let ytrain = l2norm(&xtrain);
    //     println!("{:?}", ytrain);

    //     let rng = Isaac64Rng::seed_from_u64(42);
    //     let nb_clusters = find_best_number_of_clusters(&xtrain, &ytrain, rng);
    // }
}
