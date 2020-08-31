//extern crate ndarray;
//extern crate ndarray_linalg;
//extern crate openblas_src; 

pub mod utils;

use ndarray::{arr1, s, Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use ndarray_linalg::cholesky::*;
use ndarray_linalg::triangular::*;
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

    reduced_likelihood(&arr1(&[0.1]), &distances);

    (xnorm, ynorm, distances)
}

pub fn reduced_likelihood(
    thetas: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    distances: &DistanceMatrix,
) {
    let res = f64::MIN;
    let nugget = 10. * f64::EPSILON;

    let r = squared_exponential(thetas, &distances.d);
    let mut R: Array2<f64> = Array2::eye(distances.n_obs);
    for (i, ij) in distances.d_indices.outer_iter().enumerate() {
        R[[ij[0], ij[1]]] = r[[i, 0]];
        R[[ij[1], ij[0]]] = r[[i, 0]];
    }
    let C = R.cholesky(UPLO::Lower).unwrap();
    let Ft = C.solve_triangular(UPLO::Lower, Diag::Unit, &distances.f);

    // # Get generalized least squares solution
    // Ft = linalg.solve_triangular(C, self.F, lower=True)
    // Q, G = linalg.qr(Ft, mode="economic")
    // sv = linalg.svd(G, compute_uv=False)
    // rcondG = sv[-1] / sv[0]
    // if rcondG < 1e-10:
    //     # Check F
    //     sv = linalg.svd(self.F, compute_uv=False)
    //     condF = sv[0] / sv[-1]
    //     if condF > 1e15:
    //         raise Exception(
    //             "F is too ill conditioned. Poor combination "
    //             "of regression model and observations."
    //         )

    //     else:
    //         # Ft is too ill conditioned, get out (try different theta)
    //         return reduced_likelihood_function_value, par

    // Yt = linalg.solve_triangular(C, self.y_norma, lower=True)
    // beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
    // rho = Yt - np.dot(Ft, beta)

    // # The determinant of R is equal to the squared product of the diagonal
    // # elements of its Cholesky decomposition C
    // detR = (np.diag(C) ** (2.0 / self.nt)).prod()

    ()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kriging_fit() {
        let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
        let yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];
        let kriging = Kriging::fit(&xt, &yt);
    }

    #[test]
    fn test_train() {
        let xt = array![[0.5], [1.2], [2.0], [3.0], [4.0]];
        let yt = array![[0.0], [1.0], [1.5], [0.5], [1.0]];
        train(&xt, &yt);
    }
}
