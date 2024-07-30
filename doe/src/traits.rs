use linfa::Float;
use ndarray::Array2;

/// Sampling method allowing to generate a DoE in a given sample space
///
/// A sampling method is able to generate a set of `ns` samples in a given sample space.
/// where the sample space is defined by `[lower_bound_xi, upper_bound_xi]^nx`
/// within `R^nx` where `nx` is the dimension of the sample space: x = (x_i) with i in [1, nx].
pub trait SamplingMethod<F: Float> {
    /// Returns the bounds of the sample space
    ///
    /// # Returns
    ///
    /// * A (nx, 2) matrix where the ith row is the interval of the ith components of a sample.
    fn sampling_space(&self) -> &Array2<F>;

    /// Generates a (ns, nx)-shaped array of samples belonging to `[0., 1.]^nx`
    ///
    /// # Parameters
    ///
    /// * `ns`: number of samples
    ///
    /// # Returns
    ///
    /// * A (ns, nx) matrix of samples where nx is the dimension of the sample space
    ///   each sample belongs to `[0., 1.]^nx` hypercube
    fn normalized_sample(&self, ns: usize) -> Array2<F>;

    /// Generates a (ns, nx)-shaped array of samples belonging to `[lower_bound_xi, upper_bound_xi]^nx`
    ///
    /// # Parameters
    ///
    /// * `ns`: number of samples
    ///
    /// # Returns
    ///
    /// * A (ns, nx) matrix where nx is the dimension of the sample space.
    ///   each sample belongs to `[lower_bound_xi, upper_bound_xi]^nx` where bounds
    ///   are defined as returned values of `sampling_space` function.
    fn sample(&self, ns: usize) -> Array2<F> {
        let xlimits = self.sampling_space();
        let lower = xlimits.column(0);
        let scaler = &xlimits.column(1) - &lower;
        self.normalized_sample(ns) * scaler + lower
    }
}
