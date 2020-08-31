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

    reduced_likelihood(&arr1(&[0.01]), &distances);

    (xnorm, ynorm, distances)
}

pub fn reduced_likelihood(
    thetas: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    distances: &DistanceMatrix,
) -> (ArrayBase<impl Data<Elem = f64>, Ix2>) {
    let res = f64::MIN;
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
    let Ft = C.solve_triangular(UPLO::Lower, Diag::NonUnit, &distances.f);
    println!("{:?}", &C);
    println!("{:?}", &distances.f);
    println!("FFFFTTTTT {:?}", &Ft);
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

    (C)
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
        let distances = DistanceMatrix::new(&xnorm);
        let (C) = reduced_likelihood(&arr1(&[0.01]), &distances);
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
