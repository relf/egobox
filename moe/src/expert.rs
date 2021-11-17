use crate::errors::Result;
use gp::{correlation_models::*, mean_models::*, GaussianProcess, GpParams};
use ndarray::{Array2, ArrayView2};
use paste::paste;

pub trait ExpertParams {
    fn fit(&self, x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Result<Box<dyn Expert>>;
}

pub trait Expert: std::fmt::Display {
    fn predict_values(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>>;
    fn predict_variances(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>>;
}

macro_rules! declare_expert {
    ($regr:ident, $corr:ident) => {
        paste! {
            #[derive(Clone, Copy)]
            pub struct [<Gp $regr $corr ExpertParams>](
                GpParams<f64, [<$regr Mean>], [<$corr Kernel>]>,
            );

            impl [<Gp $regr $corr ExpertParams>] {
                pub fn new(gp_params: GpParams<f64, [<$regr Mean>], [<$corr Kernel>]>) -> [<Gp $regr $corr ExpertParams>] {
                    [<Gp $regr $corr ExpertParams>](gp_params)
                }
            }

            impl ExpertParams for [<Gp $regr $corr ExpertParams>] {
                fn fit(
                    &self,
                    x: &ArrayView2<f64>,
                    y: &ArrayView2<f64>,
                ) -> Result<Box<dyn Expert>> {
                    Ok(Box::new([<Gp $regr $corr Expert>](
                        self.0.fit(x, y)?,
                    )))
                }
            }

            #[derive(Clone)]
            pub struct [<Gp $regr $corr Expert>](
                GaussianProcess<f64, [<$regr Mean>], [<$corr Kernel>]>,
            );

            impl Expert for [<Gp $regr $corr Expert>] {
                fn predict_values(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
                    Ok(self.0.predict_values(x)?)
                }
                fn predict_variances(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
                    Ok(self.0.predict_variances(x)?)
                }
            }

            impl std::fmt::Display for [<Gp $regr $corr Expert>] {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{}_{}", stringify!($regr), stringify!($corr))
                }
            }
        }
    };
}

declare_expert!(Constant, SquaredExponential);
declare_expert!(Constant, AbsoluteExponential);
declare_expert!(Constant, Matern32);
declare_expert!(Constant, Matern52);
declare_expert!(Linear, SquaredExponential);
declare_expert!(Linear, AbsoluteExponential);
declare_expert!(Linear, Matern32);
declare_expert!(Linear, Matern52);
declare_expert!(Quadratic, SquaredExponential);
declare_expert!(Quadratic, AbsoluteExponential);
declare_expert!(Quadratic, Matern32);
declare_expert!(Quadratic, Matern52);

macro_rules! make_gp_params {
    ($regr:ident, $corr:ident) => {
        paste! {
            GaussianProcess::<f64, [<$regr Mean>], [<$corr Kernel>] >::params(
                [<$regr Mean>]::default(),
                [<$corr Kernel>]::default(),
            )
        }
    };
}

macro_rules! make_expert_params {
    ($regr:ident, $corr:ident) => {
        paste! {
            Ok(Box::new([<Gp $regr $corr ExpertParams>]::new(make_gp_params!($regr, $corr))) as Box<dyn ExpertParams>)
        }
    };
}

macro_rules! compute_error {
    ($regr:ident, $corr:ident, $dataset:ident) => {{
        trace!(
            "Expert {}_{} on dataset size = {}",
            stringify!($regr),
            stringify!($corr),
            $dataset.nsamples()
        );
        let params = make_gp_params!($regr, $corr);
        let mut errors = Vec::new();
        let input_dim = $dataset.records().shape()[1];
        let n_fold = std::cmp::min($dataset.nsamples(), 5 * input_dim);
        if (n_fold < 4 * input_dim && stringify!($regr) == "Quadratic") {
            f64::INFINITY // not enough points => huge error
        } else if (n_fold < 3 * input_dim && stringify!($regr) == "Linear") {
            f64::INFINITY // not enough points => huge error
        } else {
            for (gp, valid) in $dataset.iter_fold(n_fold, |train| {
                params.fit(&train.records(), &train.targets()).unwrap()
            }) {
                let pred = gp.predict_values(valid.records()).unwrap();
                let error = (valid.targets() - pred).norm_l2();
                errors.push(error);
            }
            let mean_err = errors.iter().fold(0.0, |acc, &item| acc + item) / errors.len() as f64;
            trace!("-> mean error = {}", mean_err);
            mean_err
        }
    }};
}

macro_rules! compute_accuracies_with_corr {
    ($allowed_mean_models:ident, $allowed_corr_models:ident, $dataset:ident, $map_accuracy:ident, $regr:ident, $corr:ident) => {{
        if $allowed_corr_models.contains(&stringify!($corr)) {
            $map_accuracy.push((
                format!("{}_{}", stringify!($regr), stringify!($corr)),
                compute_error!($regr, $corr, $dataset),
            ));
        }
    }};
}

macro_rules! compute_accuracies_with_regr {
    ($allowed_mean_models:ident, $allowed_corr_models:ident, $dataset:ident, $map_accuracy:ident, $regr:ident) => {{
        if $allowed_mean_models.contains(&stringify!($regr)) {
            compute_accuracies_with_corr!(
                $allowed_mean_models,
                $allowed_corr_models,
                $dataset,
                $map_accuracy,
                $regr,
                SquaredExponential
            );
            compute_accuracies_with_corr!(
                $allowed_mean_models,
                $allowed_corr_models,
                $dataset,
                $map_accuracy,
                $regr,
                AbsoluteExponential
            );
            compute_accuracies_with_corr!(
                $allowed_mean_models,
                $allowed_corr_models,
                $dataset,
                $map_accuracy,
                $regr,
                Matern32
            );
            compute_accuracies_with_corr!(
                $allowed_mean_models,
                $allowed_corr_models,
                $dataset,
                $map_accuracy,
                $regr,
                Matern52
            );
        }
    }};
}

macro_rules! compute_accuracies {
    ($allowed_mean_models:ident, $allowed_corr_models:ident, $dataset:ident, $map_accuracy:ident) => {{
        compute_accuracies_with_regr!(
            $allowed_mean_models,
            $allowed_corr_models,
            $dataset,
            $map_accuracy,
            Constant
        );
        compute_accuracies_with_regr!(
            $allowed_mean_models,
            $allowed_corr_models,
            $dataset,
            $map_accuracy,
            Linear
        );
        compute_accuracies_with_regr!(
            $allowed_mean_models,
            $allowed_corr_models,
            $dataset,
            $map_accuracy,
            Quadratic
        );
    }};
}

pub(crate) use compute_accuracies;
pub(crate) use compute_accuracies_with_corr;
pub(crate) use compute_accuracies_with_regr;
pub(crate) use compute_error;
pub(crate) use make_expert_params;
pub(crate) use make_gp_params;
