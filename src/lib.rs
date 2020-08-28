pub mod utils;

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use utils::{constant, l1_cross_distances, normalize};

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

pub struct RegressionMatrix {
    d: Array2<f64>,
    indices: Array2<usize>,
    f: Array2<f64>,
    p: usize,
}

impl RegressionMatrix {
    pub fn new(x: &NormalizedMatrix) -> RegressionMatrix {
        let (d, indices) = l1_cross_distances(&x.data);
        let f = constant(&x.data);
        let p = f.shape()[1];

        RegressionMatrix {
            d: d.to_owned(),
            indices: indices.to_owned(),
            f: f.to_owned(),
            p,
        }
    }
}

pub struct Kriging {
    xnorm: NormalizedMatrix,
    ynorm: NormalizedMatrix,
    regression: RegressionMatrix,
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
) -> (NormalizedMatrix, NormalizedMatrix, RegressionMatrix) {
    let xnorm = NormalizedMatrix::new(x);
    let ynorm = NormalizedMatrix::new(y);

    let regression = RegressionMatrix::new(&xnorm);

    (xnorm, ynorm, regression)
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
}
