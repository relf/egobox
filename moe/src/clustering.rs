use ndarray::{s, stack, Array2, ArrayBase, Axis, Data, Ix2};

pub fn chunk<S>(x: &Array2<f64>, n: usize) -> Vec<Array2<f64>>
where
    S: Data<Elem = f64>,
{
    let mut res = vec![];
    for i in 0..n {
        res.push(x.slice(s![i..;n, ..]).to_owned());
    }
    res
}

/// Find the best number of cluster thanks to cross
pub fn find_best_number_of_clusters(
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    y: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> usize {
    let max_nb_clusters = x.len() / 10 + 1;
    let val = stack(Axis(1), &[x.view(), y.view()]).unwrap();
    let total_length = val.nrows();
    let val_cut = chunk(x, 5);

    let errori = vec![];
    let erroris = vec![];
    let posi = vec![];
    let b_ic = vec![];
    let a_ic = vec![];
    let error_h = vec![];
    let error_s = vec![];
    let median_eh = vec![];
    let median_es = vec![];

    // // Init Output Loop
    // let auxkh = 0;
    // let auxkph = 0;
    // let auxkpph = 0;
    // let auxks = 0;
    // let auxkps = 0;
    // let auxkpps = 0;
    // let ok1 = true;
    // let ok2 = true;
    // let i = 0;
    // let exit_ = true;

    // // Find error for each cluster
    // while i < max_number_cluster and exit_ {
    //     let kpls = dim > 9;

    //     let errorc = vec![];
    //     let errors = vec![];
    //     let bic_c = vec![];
    //     let aic_c = vec![];
    //     let ok = true;  // Say if this number of cluster is possible

    //     // Cross Validation
    //     for c in 0..5 {
    //         if ok {
    //             // Create training and test samples
    //             let val_train, val_test = extract_part(&val, c);
    //         }
    //     }
    //               # Create the MoE for the cross validation
    //               mixture = MoE(i + 1)

    //               mixture.set_possible_models(available_models)
    //               mixture.cluster = cls.create_clustering(
    //                   val_train[:, 0:dim],
    //                   val_train[:, dim + 1 : total_length],
    //                   i + 1,
    //                   "GMM",
    //               )  # Clustering
    //               valc = np.c_[
    //                   val_train[:, 0:dim], val_train[:, dim + 1 : total_length]
    //               ]
    //               sort = mixture.cluster.predict(valc)
    //               clus_train = cls.sort_values_by_cluster(
    //                   val_train[:, 0 : dim + 1], i + 1, sort
    //               )
    //               bic_c.append(
    //                   mixture.cluster.bic(
    //                       np.c_[
    //                           val_test[:, 0:dim], val_test[:, dim + 1 : total_length]
    //                       ]
    //                   )
    //               )
    //               aic_c.append(
    //                   mixture.cluster.aic(
    //                       np.c_[
    //                           val_test[:, 0:dim], val_test[:, dim + 1 : total_length]
    //                       ]
    //                   )
    //               )
    //               for j in range(i + 1):
    //                   # If there is at least one point
    //                   if len(clus_train[j]) < 4:
    //                       ok = False

    //               if ok:
    //                   # calculate error
    //                   try:
    //                       # Train the MoE for the cross validation
    //                       mixture.fit_without_clustering(
    //                           dim,
    //                           val_train[:, 0:dim],
    //                           val_train[:, dim],
    //                           val_train[:, dim + 1 : total_length],
    //                       )
    //                       # errors Of the MoE
    //                       errorcc = error.Error(
    //                           val_test[:, dim],
    //                           mixture._predict_hard_output(val_test[:, 0:dim]),
    //                       )
    //                       errorsc = error.Error(
    //                           val_test[:, dim],
    //                           mixture._predict_smooth_output(val_test[:, 0:dim]),
    //                       )
    //                       errors.append(errorsc.l_two_rel)
    //                       errorc.append(errorcc.l_two_rel)
    //                   except:
    //                       errorc.append(1.0)  # extrem value
    //                       errors.append(1.0)
    //               else:
    //                   errorc.append(1.0)  # extrem value
    //                   errors.append(1.0)

    //       # Stock for box plot
    //       b_ic.append(bic_c)
    //       a_ic.append(aic_c)
    //       error_s.append(errors)
    //       error_h.append(errorc)

    //       # Stock median
    //       median_eh.append(np.median(errorc))
    //       median_es.append(np.median(errors))

    //       # Stock possible numbers of cluster
    //       if ok:
    //           posi.append(i)

    //       # Stock mean errors
    //       errori.append(np.mean(errorc))
    //       erroris.append(np.mean(errors))

    //       if detail:  # pragma: no cover
    //           # Print details about the clustering
    //           print(
    //               "Number of cluster:",
    //               i + 1,
    //               "| Possible:",
    //               ok,
    //               "| Error(hard):",
    //               np.mean(errorc),
    //               "| Error(smooth):",
    //               np.mean(errors),
    //               "| Median (hard):",
    //               np.median(errorc),
    //               "| Median (smooth):",
    //               np.median(errors),
    //           )
    //           print("#######")

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
    //       i = i + 1

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

    self.nb_clusters()
}
