use linfa_clustering::{dataset::Dataset, dataset::Float, traits::Fit, GaussianMixtureModel};
use ndarray::{arr1, s, stack, Array, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use ndarray_linalg::{Lapack, Scalar};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;

enum Recombination {
	Hard,
	Smooth,
}

struct MixtureOfExpertsParams<F: Float + Scalar + Lapack, R: Rng + Clone> {
	n_clusters: usize,
	recombination: Recombination,
	heaviside_factor: Option<F>,
	rng: R,
}

impl<F: Float + Scalar + Lapack> MixtureOfExpertsParams<F, Isaac64Rng> {
	pub fn new(n_clusters: usize) -> MixtureOfExpertsParams<F, Isaac64Rng> {
		Self::new_with_rng(n_clusters, Isaac64Rng::seed_from_u64(42))
	}

	pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> MixtureOfExpertsParams<F, R2> {
		MixtureOfExpertsParams {
			n_clusters: self.n_clusters,
			recombination: self.recombination,
			heaviside_factor: self.heaviside_factor,
			rng,
		}
	}
}

impl<F: Float + Scalar + Lapack, R: Rng + Clone> MixtureOfExpertsParams<F, R> {
	pub fn new_with_rng(n_clusters: usize, rng: R) -> MixtureOfExpertsParams<F, R> {
		MixtureOfExpertsParams {
			n_clusters,
			recombination: Recombination::Hard,
			heaviside_factor: None,
			rng,
		}
	}

	fn fit(self, xt: &ArrayBase<impl Data<Elem = F>, Ix2>, yt: &ArrayBase<impl Data<Elem = F>, Ix2>) {
		//-> MixtureOfExpert {

		// Separate training data and test data
		let data = stack(Axis(1), &[xt.view(), yt.view()]).unwrap();
		let (data_test, data_train) = Self::_extract_part(&data, 10);
		let nx = xt.ncols();
		let xtrain = data_train.slice(s![.., 0..nx]);
		let ytrain = data_train.slice(s![.., nx..nx + 1]);

		// Cluster inputs
		let dataset = Dataset::from(xtrain);
		let cluster_index = GaussianMixtureModel::params(self.n_clusters)
			.with_rng(self.rng)
			.fit(&dataset)
			.expect("X training data clustering");

		// Fit GPs on clustered data
		// let data_test, data_train = self._extract_part(data, 10);

		// MixtureOfExperts{

		// }
	}

	fn _extract_part(
		data: &ArrayBase<impl Data<Elem = F>, Ix2>,
		quantile: usize,
	) -> (Array2<F>, Array2<F>) {
		let nsamples = data.nrows();
		let ndim = data.ncols();
		let indices = Array::range(0., nsamples as f32, quantile as f32).mapv(|v| v as usize);
		let data_test = data.select(Axis(0), indices.as_slice().unwrap());
		let indices2: Vec<usize> = (0..nsamples)
			.filter_map(|i| if i % quantile == 0 { None } else { Some(i) })
			.collect();
		let data_train = data.select(Axis(0), &indices2);
		(data_test, data_train)
	}
}

struct MixtureOfExperts<F: Float + Scalar + Lapack, R: Rng> {
	x: Array2<F>,
	y: Array2<F>,
	xt: Array2<F>,
	yt: Array2<F>,
	xtest: Array2<F>,
	ytest: Array2<F>,
	rng: R,
}

impl<F: Float + Scalar + Lapack, R: Rng> MixtureOfExperts<F, R> {
	pub fn params(n_clusters: usize) -> MixtureOfExpertsParams<F, Isaac64Rng> {
		MixtureOfExpertsParams::new(n_clusters)
	}
}

#[cfg(test)]
mod tests {
	extern crate openblas_src;
	use super::*;
	use approx::assert_abs_diff_eq;
	use ndarray::{arr2, array, Array2, Zip};
	use ndarray_rand::rand_distr::Uniform;
	use ndarray_rand::RandomExt;

	fn function_test_1d(x: &Array2<f64>) -> Array2<f64> {
		let mut y = Array2::zeros(x.dim());
		Zip::from(&mut y).and(x).apply(|yi, &xi| {
			if xi < 0.4 {
				*yi = xi * xi;
			} else if xi >= 0.4 && xi < 0.8 {
				*yi = 3. * xi + 1.;
			} else {
				*yi = f64::sin(10. * xi);
			}
		});
		y
	}

	#[test]
	fn test_moe() {
		let mut rng = Isaac64Rng::seed_from_u64(42);
		let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
		let yt = function_test_1d(&xt);
		let moe = MixtureOfExpertsParams::new(2).with_rng(rng).fit(&xt, &yt);
	}
}
