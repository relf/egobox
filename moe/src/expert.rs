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

macro_rules! make_surrogate_params {
    ($regr:ident, $corr:ident) => {
        paste! {
            Ok(Box::new([<Gp $regr $corr SurrogateParams>]::new(make_gp_params!($regr, $corr))) as Box<dyn GpSurrogateParams>)
        }
    };
}

macro_rules! compute_error {
    ($self:ident, $regr:ident, $corr:ident, $dataset:ident) => {{
        trace!(
            "Surrogate {}_{} on dataset size = {}",
            stringify!($regr),
            stringify!($corr),
            $dataset.nsamples()
        );
        let params = make_gp_params!($regr, $corr).set_kpls_dim($self.kpls_dim());
        let mut errors = Vec::new();
        let input_dim = $dataset.records().shape()[1];
        let n_fold = std::cmp::min($dataset.nsamples(), 5 * input_dim);
        if (n_fold < 4 * input_dim && stringify!($regr) == "Quadratic") {
            f64::INFINITY // not enough points => huge error
        } else if (n_fold < 3 * input_dim && stringify!($regr) == "Linear") {
            f64::INFINITY // not enough points => huge error
        } else {
            for (gp, valid) in $dataset.iter_fold(n_fold, |train| {
                params
                    .clone()
                    .set_kpls_dim($self.kpls_dim())
                    .fit(&train.records(), &train.targets())
                    .unwrap()
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
    ($self:ident, $allowed_corr_models:ident, $dataset:ident, $map_accuracy:ident, $regr:ident, $corr:ident) => {{
        if $allowed_corr_models.contains(&stringify!($corr)) {
            $map_accuracy.push((
                format!("{}_{}", stringify!($regr), stringify!($corr)),
                compute_error!($self, $regr, $corr, $dataset),
            ));
        }
    }};
}

macro_rules! compute_accuracies_with_regr {
    ($self:ident, $allowed_mean_models:ident, $allowed_corr_models:ident, $dataset:ident, $map_accuracy:ident, $regr:ident) => {{
        if $allowed_mean_models.contains(&stringify!($regr)) {
            compute_accuracies_with_corr!(
                $self,
                $allowed_corr_models,
                $dataset,
                $map_accuracy,
                $regr,
                SquaredExponential
            );
            compute_accuracies_with_corr!(
                $self,
                $allowed_corr_models,
                $dataset,
                $map_accuracy,
                $regr,
                AbsoluteExponential
            );
            compute_accuracies_with_corr!(
                $self,
                $allowed_corr_models,
                $dataset,
                $map_accuracy,
                $regr,
                Matern32
            );
            compute_accuracies_with_corr!(
                $self,
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
    ($self:ident, $allowed_mean_models:ident, $allowed_corr_models:ident, $dataset:ident, $map_accuracy:ident) => {{
        compute_accuracies_with_regr!(
            $self,
            $allowed_mean_models,
            $allowed_corr_models,
            $dataset,
            $map_accuracy,
            Constant
        );
        compute_accuracies_with_regr!(
            $self,
            $allowed_mean_models,
            $allowed_corr_models,
            $dataset,
            $map_accuracy,
            Linear
        );
        compute_accuracies_with_regr!(
            $self,
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
pub(crate) use make_gp_params;
pub(crate) use make_surrogate_params;
