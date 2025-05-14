//! Globalized bounded Nelder-Mead optimizer (Gbnm) - API

mod internal;

pub use internal::{gbnm, GbnmOptions};
use ndarray::Array1;

#[cfg(not(feature = "nlopt"))]
use crate::types::ObjFn;
#[cfg(feature = "nlopt")]
use nlopt::ObjFn;

use crate::InfillObjData;
#[derive(Debug, Clone)]
pub struct Options {
    pub max_restarts: usize,
    pub max_evals: usize,
    pub n_points: usize,
    pub max_iter: usize,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub ssigma: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_restarts: 8,
            max_evals: 5000,
            n_points: 100,
            max_iter: 300,
            alpha: 1.0,
            beta: 0.5,
            gamma: 2.0,
            epsilon: 1e-8,
            ssigma: 1e-8,
        }
    }
}

/// Result returned by the optimizer
#[derive(Debug)]
pub struct Result {
    pub x: Vec<f64>,
    pub fval: f64,
}

/// Run the optimizer
pub fn minimize<F>(
    fun: F,
    bounds: &[(f64, f64)],
    args: &mut InfillObjData<f64>,
    options: Options,
) -> std::result::Result<Result, &'static str>
where
    F: ObjFn<InfillObjData<f64>> + Sync,
{
    if bounds.is_empty() {
        return Err("Bounds cannot be empty");
    }
    let xmin: Array1<f64> = bounds.iter().map(|b| b.0).collect();
    let xmax: Array1<f64> = bounds.iter().map(|b| b.1).collect();

    let wrapped_fun = |x: &[f64], u: &mut InfillObjData<f64>| fun(x, None, u);

    let internal_options = GbnmOptions {
        max_restarts: options.max_restarts,
        max_evals: options.max_evals,
        n_points: options.n_points,
        max_iter: options.max_iter,
        alpha: options.alpha,
        beta: options.beta,
        gamma: options.gamma,
        epsilon: options.epsilon,
        ssigma: options.ssigma,
    };

    let result = gbnm(wrapped_fun, &xmin, &xmax, args.clone(), internal_options);

    Ok(Result {
        x: result.x.to_vec(),
        fval: result.fval,
    })
}
