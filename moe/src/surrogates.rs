use crate::errors::Result;
use egobox_gp::{correlation_models::*, mean_models::*, GaussianProcess, GpParams};
use linfa::prelude::{Dataset, Fit};
use ndarray::{Array2, ArrayView2};
use paste::paste;

#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "persistent")]
use crate::MoeError;
#[cfg(feature = "persistent")]
use std::fs;
#[cfg(feature = "persistent")]
use std::io::Write;
/// A trait for surrogate parameters to build surrogate once fitted.
pub trait SurrogateParams {
    /// Set initial theta
    fn initial_theta(&mut self, theta: Vec<f64>);
    /// Set the number of PLS components
    fn kpls_dim(&mut self, kpls_dim: Option<usize>);
    /// Set the nugget parameter to improve numerical stability
    fn nugget(&mut self, nugget: f64);
    /// Train the surrogate
    fn train(&self, x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Result<Box<dyn Surrogate>>;
}

/// A trait for a surrogate used as expert in the mixture.
#[cfg_attr(feature = "serializable", typetag::serde(tag = "type"))]
pub trait Surrogate: std::fmt::Display + Sync + Send {
    /// Predict output values at n points given as (n, xdim) matrix.
    fn predict_values(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>>;
    /// Predict variance values at n points given as (n, xdim) matrix.
    fn predict_variances(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>>;
    /// Predict derivatives at n points and return (n, xdim) matrix
    /// where each column is the partial derivatives wrt the ith component
    fn predict_derivatives(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>>;
    /// Predict derivatives of the variance at n points and return (n, xdim) matrix
    /// where each column is the partial derivatives wrt the ith component
    fn predict_variance_derivatives(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>>;
    /// Sample trajectories
    fn sample(&self, x: &ArrayView2<f64>, n_traj: usize) -> Result<Array2<f64>>;
    /// Save model in given file.
    #[cfg(feature = "persistent")]
    fn save(&self, path: &str) -> Result<()>;
}

/// A macro to declare GP surrogate using regression model and correlation model names.
///
/// Regression model is either `Constant`, `Linear` or `Quadratic`.
/// Correlation model is either `SquaredExponential`, `AbsoluteExponential`, `Matern32` or `Matern52`.
macro_rules! declare_surrogate {
    ($regr:ident, $corr:ident) => {
        paste! {

            #[doc = "GP surrogate parameters with `" $regr "` regression model and `" $corr "` correlation model. \n\nSee [egobox_gp::GpParams]"]
            #[derive(Clone, Debug)]
            pub struct [<Gp $regr $corr SurrogateParams>](
                GpParams<f64, [<$regr Mean>], [<$corr Corr>]>,
            );

            impl [<Gp $regr $corr SurrogateParams>] {
                /// Constructor
                pub fn new(gp_params: GpParams<f64, [<$regr Mean>], [<$corr Corr>]>) -> [<Gp $regr $corr SurrogateParams>] {
                    [<Gp $regr $corr SurrogateParams>](gp_params)
                }
            }

            impl SurrogateParams for [<Gp $regr $corr SurrogateParams>] {
                fn initial_theta(&mut self, theta: Vec<f64>) {
                    self.0 = self.0.clone().initial_theta(Some(theta));
                }

                fn kpls_dim(&mut self, kpls_dim: Option<usize>) {
                    self.0 = self.0.clone().kpls_dim(kpls_dim);
                }

                fn nugget(&mut self, nugget: f64) {
                    self.0 = self.0.clone().nugget(nugget);
                }

                fn train(
                    &self,
                    x: &ArrayView2<f64>,
                    y: &ArrayView2<f64>,
                ) -> Result<Box<dyn Surrogate>> {
                    Ok(Box::new([<Gp $regr $corr Surrogate>](
                        self.0.clone().fit(&Dataset::new(x.to_owned(), y.to_owned()))?,
                    )))
                }
            }

            #[doc = "GP surrogate with `" $regr "` regression model and `" $corr "` correlation model. \n\nSee [egobox_gp::GaussianProcess]"]
            #[derive(Clone, Debug)]
            #[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
            pub struct [<Gp $regr $corr Surrogate>](
                pub GaussianProcess<f64, [<$regr Mean>], [<$corr Corr>]>,
            );

            #[cfg_attr(feature = "serializable", typetag::serde)]
            impl Surrogate for [<Gp $regr $corr Surrogate>] {
                fn predict_values(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
                    Ok(self.0.predict_values(x)?)
                }
                fn predict_variances(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
                    Ok(self.0.predict_variances(x)?)
                }
                fn predict_derivatives(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
                    Ok(self.0.predict_derivatives(x))
                }
                fn predict_variance_derivatives(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
                    Ok(self.0.predict_variance_derivatives(x))
                }
                fn sample(&self, x: &ArrayView2<f64>, n_traj: usize) -> Result<Array2<f64>> {
                    Ok(self.0.sample(x, n_traj))
                }

                #[cfg(feature = "persistent")]
                fn save(&self, path: &str) -> Result<()> {
                    let mut file = fs::File::create(path).unwrap();
                    let bytes = match serde_json::to_string(self as &dyn Surrogate) {
                        Ok(b) => b,
                        Err(err) => return Err(MoeError::SaveError(err))
                    };
                    file.write_all(bytes.as_bytes())?;
                    Ok(())
                }

            }

            impl std::fmt::Display for [<Gp $regr $corr Surrogate>] {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{}_{}{}", stringify!($regr), stringify!($corr),
                        match self.0.kpls_dim() {
                            None => String::from(""),
                            Some(dim) => format!("_PLS({})", dim),
                        }
                    )
                }
            }
        }
    };
}

declare_surrogate!(Constant, SquaredExponential);
declare_surrogate!(Constant, AbsoluteExponential);
declare_surrogate!(Constant, Matern32);
declare_surrogate!(Constant, Matern52);
declare_surrogate!(Linear, SquaredExponential);
declare_surrogate!(Linear, AbsoluteExponential);
declare_surrogate!(Linear, Matern32);
declare_surrogate!(Linear, Matern52);
declare_surrogate!(Quadratic, SquaredExponential);
declare_surrogate!(Quadratic, AbsoluteExponential);
declare_surrogate!(Quadratic, Matern32);
declare_surrogate!(Quadratic, Matern52);

#[doc(hidden)]
// Create GP surrogate parameters with given regression and correlation models.
macro_rules! make_surrogate_params {
    ($regr:ident, $corr:ident) => {
        paste! {
            #[allow(unused_allocation)]
            Box::new([<Gp $regr $corr SurrogateParams>]::new(
                GaussianProcess::<f64, [<$regr Mean>], [<$corr Corr>] >::params(
                    [<$regr Mean>]::default(),
                    [<$corr Corr>]::default(),
                )
            ))
        }
    };
}

#[cfg(feature = "persistent")]
/// Load GP surrogate from given json file.
pub fn load(path: &str) -> Result<Box<dyn Surrogate>> {
    let data = fs::read_to_string(path)?;
    let gp: Box<dyn Surrogate> = serde_json::from_str(&data).unwrap();
    Ok(gp)
}

pub(crate) use make_surrogate_params;

#[cfg(feature = "persistent")]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use egobox_doe::{Lhs, SamplingMethod};
    #[cfg(not(feature = "blas"))]
    use linfa_linalg::norm::*;
    use ndarray::array;
    #[cfg(feature = "blas")]
    use ndarray_linalg::Norm;
    use ndarray_stats::DeviationExt;

    fn xsinx(x: &Array2<f64>) -> Array2<f64> {
        (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
    }

    #[test]
    fn test_save_load() {
        let xlimits = array![[0., 25.]];
        let xt = Lhs::new(&xlimits).sample(10);
        let yt = xsinx(&xt);
        let gp = make_surrogate_params!(Constant, SquaredExponential)
            .train(&xt.view(), &yt.view())
            .expect("GP fit error");
        gp.save("target/tests/save_gp.json").expect("GP not saved");
        let gp = load("target/tests/save_gp.json").expect("GP not loaded");
        let xv = Lhs::new(&xlimits).sample(20);
        let yv = xsinx(&xv);
        let ytest = gp.predict_values(&xv.view()).unwrap();
        let err = ytest.l2_dist(&yv).unwrap() / yv.norm_l2();
        assert_abs_diff_eq!(err, 0., epsilon = 2e-1);
    }

    #[test]
    fn test_load_fail() {
        let gp = load("notfound.json");
        assert!(gp.is_err());
    }
}
