use crate::utils::{constant, l1_cross_distances, normalize, squared_exponential};
use ndarray::{arr1, arr2, s, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2, Zip};
use ndarray_einsum_beta::*;
use ndarray_linalg::cholesky::*;
use ndarray_linalg::qr::*;
use ndarray_linalg::svd::*;
use ndarray_linalg::triangular::*;
use ndarray_stats::*;
use nlopt::*;
use std::collections::HashMap;

pub struct NormalizedMatrix {
    pub data: Array2<f64>,
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
}

impl NormalizedMatrix {
    pub fn new(x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> NormalizedMatrix {
        let (data, mean, std) = normalize(x);
        NormalizedMatrix {
            data: data.to_owned(),
            mean: mean.to_owned(),
            std: std.to_owned(),
        }
    }
}

pub struct DistanceMatrix {
    d: Array2<f64>,
    d_indices: Array2<usize>,
    f: Array2<f64>,
    p: usize,
    n_obs: usize,
    n_features: usize,
}

impl DistanceMatrix {
    pub fn new(x: &NormalizedMatrix) -> DistanceMatrix {
        let (d, d_indices) = l1_cross_distances(&x.data);
        let f = constant(&x.data);
        let p = f.shape()[1];
        let n_obs = x.data.nrows();
        let n_features = x.data.ncols();

        DistanceMatrix {
            d: d.to_owned(),
            d_indices: d_indices.to_owned(),
            f: f.to_owned(),
            p,
            n_obs,
            n_features,
        }
    }
}

pub struct Kriging {
    xnorm: NormalizedMatrix,
    ynorm: NormalizedMatrix,
    distances: DistanceMatrix,
    hyper_parameters: Hyperparameters,
}

impl Kriging {
    pub fn fit(
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Kriging {
        let xnorm = NormalizedMatrix::new(x);
        let ynorm = NormalizedMatrix::new(y);
        let distances = DistanceMatrix::new(&xnorm);

        let hyper_parameters =
            optimize_hyperparameters(&arr1(&[0.01]), &distances, &ynorm).unwrap();

        Kriging {
            xnorm,
            ynorm,
            distances,
            hyper_parameters,
        }
    }

    pub fn predict_values(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> ArrayBase<impl Data<Elem = f64>, Ix2> {
        let r = self._compute_correlation(&x);
        // Compute the regression function
        let f = constant(x);
        // Scaled predictor
        let y_ = &f.dot(&self.hyper_parameters.likelihood.beta)
            + &r.dot(&self.hyper_parameters.likelihood.gamma);
        // Predictor
        &y_ * &self.ynorm.std + &self.ynorm.mean
    }

    pub fn predict_variances(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> ArrayBase<impl Data<Elem = f64>, Ix2> {
        let r = self._compute_correlation(&x);
        let lh = &self.hyper_parameters.likelihood;

        let tr = r.t().to_owned();
        let rt =
            lh.C.solve_triangular(UPLO::Lower, Diag::NonUnit, &tr)
                .unwrap();
        let lhs = lh.Ft.t().dot(&rt) - constant(x).t();
        let u =
            lh.G.t()
                .solve_triangular(UPLO::Upper, Diag::NonUnit, &lhs)
                .unwrap();

        let A = &lh.sigma2;
        let B = 1.0 - rt.mapv(|v| v * v).sum_axis(Axis(0)) + u.mapv(|v| v * v).sum_axis(Axis(0));
        let mut MSE = einsum("i,j->ji", &[A, &B])
            .unwrap()
            .into_shape((x.shape()[0], 1))
            .unwrap();

        // Mean Squared Error might be slightly negative depending on
        // machine precision: force to zero!
        MSE.mapv(|v| if v < 0. { 0. } else { v })
    }

    fn _compute_correlation(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> ArrayBase<impl Data<Elem = f64>, Ix2> {
        let n_obs = x.shape()[0];
        let n_features = x.shape()[1];

        let xnorm = (x - &self.xnorm.mean) / &self.xnorm.std;
        let nt = self.xnorm.data.shape()[0];
        // Get pairwise componentwise L1-distances to the input training set
        let mut dx: Array2<f64> = Array2::zeros((nt * n_obs, n_features));
        for (i, xrow) in xnorm.genrows().into_iter().enumerate() {
            let dxrows = &self.xnorm.data - &xrow.into_shape((1, n_features)).unwrap();
            let a = i * nt;
            let b = (i + 1) * nt;
            dx.slice_mut(s![a..b, ..]).assign(&dxrows);
        }
        // Compute the correlation function
        let r = squared_exponential(&self.hyper_parameters.thetas, &dx);
        r.into_shape((n_obs, nt)).unwrap()
    }
}

pub struct Hyperparameters {
    thetas: Array1<f64>,
    likelihood: ReducedLikelihood,
}

pub fn optimize_hyperparameters(
    theta0s: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    distances: &DistanceMatrix,
    ynorm: &NormalizedMatrix,
) -> Option<Hyperparameters> {
    let base: f64 = 10.;
    let objfn = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
        let thetas =
            Array1::from_shape_vec((x.len(),), x.iter().map(|v| base.powf(*v)).collect()).unwrap();
        let r = reduced_likelihood(&thetas, &distances, &ynorm).unwrap();
        -r.value
    };
    let mut optimizer = Nlopt::new(
        Algorithm::Cobyla,
        distances.n_features,
        objfn,
        Target::Minimize,
        (),
    );

    for i in 0..theta0s.len() {
        let cstrfn1 = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            // -(f64::log10(100.) - x[i])
            x[i] - 2.
        };
        let cstrfn2 = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
            // -(x[i] - f64::log10(1e-6))
            -x[i] - 6.
        };
        optimizer.add_inequality_constraint(cstrfn1, (), 1e-2);
        optimizer.add_inequality_constraint(cstrfn2, (), 1e-2);
    }

    let mut k = 1;
    let mut stop = 1;
    let mut best_rlf = f64::MIN;

    let mut thetas_vec = theta0s.mapv(|t| f64::log10(t)).into_raw_vec();
    optimizer.set_initial_step1(0.5);
    optimizer.set_maxeval(10 * distances.n_features as u32);
    let resOpt = optimizer.optimize(&mut thetas_vec);

    if let Err(e) = resOpt {
        println!("{:?}", e);
    }
    let thetas = arr1(&thetas_vec).mapv(|v| base.powf(v));
    let likelihood = reduced_likelihood(&thetas, &distances, &ynorm).unwrap();
    Some(Hyperparameters { thetas, likelihood })
}

#[derive(Debug)]
pub struct ReducedLikelihood {
    value: f64,
    sigma2: Array1<f64>,
    beta: Array2<f64>,
    gamma: Array2<f64>,
    C: Array2<f64>,
    Ft: Array2<f64>,
    G: Array2<f64>,
}

pub fn reduced_likelihood(
    thetas: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    distances: &DistanceMatrix,
    ynorm: &NormalizedMatrix,
) -> Option<ReducedLikelihood> {
    let mut reduced_likelihood = None;
    let nugget = 10. * f64::EPSILON;

    let r = squared_exponential(thetas, &distances.d);
    let mut R: Array2<f64> = Array2::eye(distances.n_obs);
    for (i, ij) in distances.d_indices.outer_iter().enumerate() {
        R[[ij[0], ij[1]]] = r[[i, 0]];
        R[[ij[1], ij[0]]] = r[[i, 0]];
    }
    let C = R.cholesky(UPLO::Lower).unwrap();
    let Ft = C
        .solve_triangular(UPLO::Lower, Diag::NonUnit, &distances.f)
        .unwrap();
    let (Q, G) = Ft.qr().unwrap();
    let (_, sv_g, _) = G.svd(false, false).unwrap();

    let cond_G = sv_g[sv_g.len() - 1] / sv_g[0];
    if cond_G < 1e-10 {
        let (_, sv_f, _) = distances.f.svd(false, false).unwrap();
        let cond_F = sv_f[0] / sv_f[sv_f.len() - 1];
        if cond_F > 1e15 {
            panic!(
                "F is too ill conditioned. Poor combination \
                   of regression model and observations."
            );
        } else {
            // Ft is too ill conditioned, get out (try different theta)
            return reduced_likelihood;
        }
    }

    let Yt = C
        .solve_triangular(UPLO::Lower, Diag::NonUnit, &ynorm.data)
        .unwrap();
    let beta = G
        .solve_triangular(UPLO::Upper, Diag::NonUnit, &Q.t().dot(&Yt))
        .unwrap();
    let rho = Yt - Ft.dot(&beta);

    // The determinant of R is equal to the squared product of the diagonal
    // elements of its Cholesky decomposition C
    let exp = 2.0 / distances.n_obs as f64;
    let mut detR = 1.0;
    for v in C.diag().mapv(|v| v.powf(exp)).iter() {
        detR *= v;
    }
    let rho_sqr = rho.map(|v| v.powf(2.));
    let sigma2 = rho_sqr.sum_axis(Axis(0)) / distances.n_obs as f64;
    let reduced_likelihood = ReducedLikelihood {
        value: -sigma2.sum() * detR,
        sigma2: sigma2 * ynorm.std.mapv(|v| v.powf(2.0)),
        beta: beta,
        gamma: C
            .t()
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &rho)
            .unwrap(),
        C: C,
        Ft: Ft,
        G: G,
    };
    Some(reduced_likelihood)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_train_and_predict_values() {
        let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
        let yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];
        let kriging = Kriging::fit(&xt, &yt);
        let expected = 5.62341325;
        assert_abs_diff_eq!(expected, kriging.hyper_parameters.thetas[0], epsilon = 1e-6);
        let yvals = kriging.predict_values(&arr2(&[[1.0], [2.1]]));
        let expected_y = arr2(&[[0.6856779931432053], [1.4484644169993859]]);
        assert_abs_diff_eq!(expected_y, yvals, epsilon = 1e-6);
    }
    #[test]
    fn test_train_and_predict_variances() {
        let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
        let yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];
        let kriging = Kriging::fit(&xt, &yt);
        let expected = 5.62341325;
        assert_abs_diff_eq!(expected, kriging.hyper_parameters.thetas[0], epsilon = 1e-6);
        let yvars = kriging.predict_variances(&arr2(&[[1.0], [2.1]]));
        let expected_vars = arr2(&[[0.03422835527498675], [0.014105203477142668]]);
        assert_abs_diff_eq!(expected_vars, yvars, epsilon = 1e-6);
    }

    #[test]
    fn test_reduced_likelihood() {
        let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
        let yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];
        let xnorm = NormalizedMatrix::new(&xt);
        let ynorm = NormalizedMatrix::new(&yt);
        let distances = DistanceMatrix::new(&xnorm);
        let likelihood = reduced_likelihood(&arr1(&[0.01]), &distances, &ynorm).unwrap();
        let expectedC = array![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.9974877605580126, 0.07083902552238376, 0.0, 0.0, 0.0],
            [
                0.9885161407188499,
                0.1508662574804237,
                0.008672479008999145,
                0.0,
                0.0
            ],
            [
                0.9684250479962548,
                0.24722219638844298,
                0.0321021196409536,
                0.0018883699011518966,
                0.0
            ],
            [
                0.9390514487564239,
                0.33682559354693586,
                0.06830490304103372,
                0.008072558118420513,
                0.0004124878125062375
            ]
        ];
        let expectedFt = array![
            [1.0],
            [0.035464059866176435],
            [0.7072406041817856],
            [0.05482333748840767],
            [0.6128247876983186]
        ];
        let expected = -8822.907752408328;
        assert_abs_diff_eq!(expected, likelihood.value, epsilon = 1e-6);
    }

    #[test]
    fn test_optimize_hyperparameters() {
        let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
        let yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];
        let xnorm = NormalizedMatrix::new(&xt);
        let ynorm = NormalizedMatrix::new(&yt);
        let distances = DistanceMatrix::new(&xnorm);
        optimize_hyperparameters(&arr1(&[0.01]), &distances, &ynorm);
    }
}
