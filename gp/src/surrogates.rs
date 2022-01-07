use crate::errors::Result;
use crate::{correlation_models::*, mean_models::*, GaussianProcess, GpParams};
use ndarray::{Array2, ArrayView2};
use paste::paste;

pub trait SurrogateParams {
    fn set_initial_theta(&mut self, theta: f64);
    fn set_kpls_dim(&mut self, kpls_dim: Option<usize>);
    fn set_nugget(&mut self, nugget: f64);
    fn fit(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<Box<dyn Surrogate>>;
}

pub trait Surrogate: std::fmt::Display {
    fn predict_values(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>>;
    fn predict_variances(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>>;
}

macro_rules! declare_surrogate {
    ($regr:ident, $corr:ident) => {
        paste! {
            #[derive(Clone, Copy)]
            pub struct [<Gp $regr $corr SurrogateParams>](
                GpParams<f64, [<$regr Mean>], [<$corr Kernel>]>,
            );

            impl [<Gp $regr $corr SurrogateParams>] {
                pub fn new(gp_params: GpParams<f64, [<$regr Mean>], [<$corr Kernel>]>) -> [<Gp $regr $corr SurrogateParams>] {
                    [<Gp $regr $corr SurrogateParams>](gp_params)
                }
            }

            impl SurrogateParams for [<Gp $regr $corr SurrogateParams>] {
                fn set_initial_theta(&mut self, theta: f64) {
                    self.0 = self.0.set_initial_theta(theta);
                }

                fn set_kpls_dim(&mut self, kpls_dim: Option<usize>) {
                    self.0 = self.0.set_kpls_dim(kpls_dim);
                }

                fn set_nugget(&mut self, nugget: f64) {
                    self.0 = self.0.set_nugget(nugget);
                }

                fn fit(
                    &self,
                    x: &Array2<f64>,
                    y: &Array2<f64>,
                ) -> Result<Box<dyn Surrogate>> {
                    Ok(Box::new([<Gp $regr $corr Surrogate>](
                        self.0.fit(x, y)?,
                    )))
                }
            }

            #[derive(Clone)]
            pub struct [<Gp $regr $corr Surrogate>](
                pub GaussianProcess<f64, [<$regr Mean>], [<$corr Kernel>]>,
            );

            impl Surrogate for [<Gp $regr $corr Surrogate>] {
                fn predict_values(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
                    self.0.predict_values(x)
                }
                fn predict_variances(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
                    self.0.predict_variances(x)
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
