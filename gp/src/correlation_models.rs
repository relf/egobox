use ndarray::{Array2, ArrayBase, Axis, Data, Ix1, Ix2};

pub trait CorrelationModel: Clone + Copy + Default {
    fn apply(
        &self,
        theta: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        d: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        weights: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Array2<f64>;
}

#[derive(Clone, Copy, Default)]
pub struct SquaredExponentialKernel();

impl CorrelationModel for SquaredExponentialKernel {
    fn apply(
        &self,
        theta: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        d: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        weights: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Array2<f64> {
        let wd = d.mapv(|v| v * v).dot(&weights.mapv(|v| v * v));
        let r = (wd * theta).sum_axis(Axis(1)).mapv(|v| f64::exp(-v));
        r.into_shape((d.nrows(), 1)).unwrap()
    }
}

#[derive(Clone, Copy, Default)]
pub struct AbsoluteExponentialKernel();

impl CorrelationModel for AbsoluteExponentialKernel {
    fn apply(
        &self,
        theta: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        d: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        weights: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Array2<f64> {
        let wd = d.mapv(|v| v.abs()).dot(&weights.mapv(|v| v.abs()));
        let r = (wd * theta).sum_axis(Axis(1)).mapv(|v| f64::exp(-v));
        r.into_shape((d.nrows(), 1)).unwrap()
    }
}

#[derive(Clone, Copy, Default)]
pub struct Matern32Kernel();

impl CorrelationModel for Matern32Kernel {
    fn apply(
        &self,
        theta: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        d: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        weights: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Array2<f64> {
        let wd = d
            .mapv(|v| v.abs() * f64::sqrt(3.))
            .dot(&weights.mapv(|v| v.abs()));
        let r = (wd * theta).mapv(|v| (1. + v) * f64::exp(-v));
        r.into_shape((d.nrows(), 1)).unwrap()
    }
}

#[derive(Clone, Copy, Default)]
pub struct Matern52Kernel();

impl CorrelationModel for Matern52Kernel {
    fn apply(
        &self,
        theta: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        d: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        weights: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Array2<f64> {
        let wd = d
            .mapv(|v| v.abs() * f64::sqrt(5.))
            .dot(&weights.mapv(|v| v.abs()));
        let r = (wd * theta).mapv(|v| (1. + v + v * v / 3.) * f64::exp(-v));
        r.into_shape((d.nrows(), 1)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::DistanceMatrix;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, array};

    #[test]
    fn test_squared_exponential() {
        let xt = array![[4.5], [1.2], [2.0], [3.0], [4.0]];
        let dm = DistanceMatrix::new(&xt);
        let res = SquaredExponentialKernel::default().apply(&arr1(&[0.1]), &dm.d, &array![[1.]]);
        let expected = array![
            [0.336552878364737],
            [0.5352614285189903],
            [0.7985162187593771],
            [0.9753099120283326],
            [0.9380049995307295],
            [0.7232502423798424],
            [0.4565760496233148],
            [0.9048374180359595],
            [0.6703200460356393],
            [0.9048374180359595]
        ];
        assert_abs_diff_eq!(res, expected, epsilon = 1e-6);
    }
}
