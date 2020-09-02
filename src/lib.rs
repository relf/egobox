//extern crate ndarray;
//extern crate ndarray_linalg;
//extern crate openblas_src; 

pub mod utils;

use std::collections::HashMap;
use ndarray::{arr1, arr2, s, Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use ndarray_linalg::cholesky::*;
use ndarray_linalg::triangular::*;
use ndarray_linalg::qr::*;
use ndarray_linalg::svd::*;
use utils::{constant, l1_cross_distances, normalize, squared_exponential};

pub struct NormalizedMatrix {
    data: Array2<f64>,
    mean: Array1<f64>,
    std: Array1<f64>,
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
    regression: DistanceMatrix,
}

impl Kriging {
    pub fn fit(
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Kriging {
        // # Optimization
        // (
        //     self.optimal_rlf_value,
        //     self.optimal_par,
        //     self.optimal_theta,
        // ) = self._optimize_hyperparam(D)

        let (
            xnorm,
            ynorm,
            regression, // rlf_value, params, thetas
        ) = train(x, y);

        Kriging {
            xnorm,
            ynorm,
            regression,
        }
    }
}

pub fn train(
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    y: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> (NormalizedMatrix, NormalizedMatrix, DistanceMatrix) {
    let xnorm = NormalizedMatrix::new(x);
    let ynorm = NormalizedMatrix::new(y);

    let distances = DistanceMatrix::new(&xnorm);

    reduced_likelihood(&arr1(&[0.01]), &distances, &ynorm);

    (xnorm, ynorm, distances)
}

pub fn reduced_likelihood(
    thetas: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    distances: &DistanceMatrix,
    ynorm: &NormalizedMatrix
) -> (f64, HashMap<String, f64>) {
    let res = f64::MIN;
    let params = HashMap::new();
    let nugget = 10. * f64::EPSILON;

    let r = squared_exponential(thetas, &distances.d);
    println!("DDDDD {:?} THETAS {:?}", distances.d, thetas);
    println!("rrrrrr {:?}", r);

    let mut R: Array2<f64> = Array2::eye(distances.n_obs);
    for (i, ij) in distances.d_indices.outer_iter().enumerate() {
        R[[ij[0], ij[1]]] = r[[i, 0]];
        R[[ij[1], ij[0]]] = r[[i, 0]];
    }
    println!("RRRRR {:?}", &R);
    let C = R.cholesky(UPLO::Lower).unwrap();
    let Ft = C.solve_triangular(UPLO::Lower, Diag::NonUnit, &distances.f).unwrap();

    println!("{:?}", &C);
    println!("{:?}", &distances.f);
    println!("FFFFTTTTT {:?}", &Ft);

    let (Q, G) = Ft.qr().unwrap();
    let (_, sv_g, _) = G.svd(false, false).unwrap();
    let cond_G = sv_g[sv_g.len()-1] / sv_g[0];
    if cond_G < 1e-10 {
        let (_, sv_f, _) = distances.f.svd(false, false).unwrap();
        let cond_F = sv_f[0] / sv_f[sv_f.len()-1];
        if cond_F > 1e15 {
            panic!("F is too ill conditioned. Poor combination \
                   of regression model and observations.");
        } else {
            // Ft is too ill conditioned, get out (try different theta)
            return (res, params)
        }
    }

    let Yt = C.solve_triangular(UPLO::Lower, Diag::NonUnit, &ynorm.data).unwrap();
    let beta = G.solve_triangular(UPLO::Upper, Diag::NonUnit, &Q.t().dot(&Yt)).unwrap();
    let rho = Yt - Ft.dot(&beta);

    // The determinant of R is equal to the squared product of the diagonal
    // elements of its Cholesky decomposition C
    let exp = 2.0 / distances.n_obs as f64;
    let mut detR = 1.0;
    for v in C.diag().mapv(|v| v.powf(exp)).iter() {
        detR *= v;
    }

    let sigma2 = (rho.map(|v| v.powf(2.))).sum(Axis(0)) / distances.n_obs;
    // reduced_likelihood_function_value = -sigma2.sum() * detR
    // par["sigma2"] = sigma2 * self.y_std ** 2.0
    // par["beta"] = beta
    // par["gamma"] = linalg.solve_triangular(C.T, rho)
    // par["C"] = C
    // par["Ft"] = Ft
    // par["G"] = G

    // # A particular case when f_min_cobyla fail
    // if (self.best_iteration_fail is not None) and (
    //     not np.isinf(reduced_likelihood_function_value)
    // ):

    //     if reduced_likelihood_function_value > self.best_iteration_fail:
    //         self.best_iteration_fail = reduced_likelihood_function_value
    //         self._thetaMemory = np.array(tmp_var)

    // elif (self.best_iteration_fail is None) and (
    //     not np.isinf(reduced_likelihood_function_value)
    // ):
    //     self.best_iteration_fail = reduced_likelihood_function_value
    //     self._thetaMemory = np.array(tmp_var)

    // return reduced_likelihood_function_value, par

    (res, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // #[test]
    // fn test_kriging_fit() {
    //     let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
    //     let yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];
    //     let kriging = Kriging::fit(&xt, &yt);
    // }

    #[test]
    fn test_reduced_likelihood() {
        let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
        let yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];
        let xnorm = NormalizedMatrix::new(&xt);
        let ynorm = NormalizedMatrix::new(&yt);
        let distances = DistanceMatrix::new(&xnorm);
        let (C) = reduced_likelihood(&arr1(&[0.01]), &distances, &ynorm);
        let expectedC = 
            array![[1.0, 0.0, 0.0, 0.0, 0.0],
            [0.9974877605580126, 0.07083902552238376, 0.0, 0.0, 0.0],
            [0.9885161407188499, 0.1508662574804237, 0.008672479008999145, 0.0, 0.0],
            [0.9684250479962548, 0.24722219638844298, 0.0321021196409536, 0.0018883699011518966, 0.0],
            [0.9390514487564239, 0.33682559354693586, 0.06830490304103372, 0.008072558118420513, 0.0004124878125062375]];
        let expectedFt =
            array![[1.0],
            [0.035464059866176435],
            [0.7072406041817856],
            [0.05482333748840767],
            [0.6128247876983186]];
    }
}
