use libm::{erf, erfc, exp, expm1, log, log1p};

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const INV_SQRT_2: f64 = 0.7071067811865475;
const SQRT_2PI: f64 = 2.5066282746310002;
const LOG_2PI_OVER_2: f64 = 0.9189385332046727; // log(2π)/2
const LOG_PI_OVER_2_ALL_OVER_2: f64 = 0.2257913526447274; // log(π/2)/2

fn normal_pdf(u: f64) -> f64 {
    exp(-0.5 * u * u) / SQRT_2PI
}

fn normal_cdf(u: f64) -> f64 {
    0.5 * (1.0 + erf(u / SQRT_2))
}

fn erfcx(u: f64) -> f64 {
    exp(u * u) * erfc(u)
}

fn log1mexp(x: f64) -> f64 {
    let log2 = log(2.0);
    if x > -log2 {
        log(-expm1(x))
    } else {
        log1p(-exp(x))
    }
}

pub fn log_ei_helper(u: f64) -> f64 {
    if u > -1.0 {
        log(normal_pdf(u) + u * normal_cdf(u))
    } else {
        let log_phi_u = -0.5 * u * u - LOG_2PI_OVER_2;

        let log_term = if u > -1. / f64::sqrt(1e-6) {
            let w = log(erfcx(-INV_SQRT_2 * u) * u.abs()) + LOG_PI_OVER_2_ALL_OVER_2;
            log1mexp(w)
        } else {
            -2.0 * log(u.abs())
        };

        log_phi_u + log_term
    }
}

pub fn d_log_ei_helper(u: f64) -> f64 {
    let phi = normal_pdf(u);
    let big_phi = normal_cdf(u);

    let numerator = big_phi + phi - u * phi;
    let denominator = log_ei_helper(u).exp();

    numerator / denominator
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use approx::assert_abs_diff_eq;
    use ndarray::Array1;
    use ndarray_npy::write_npy;

    use crate::utils::{d_log_ei_helper, log_ei_helper};

    #[test]
    fn test_log_ei_helper() {
        let vals = [-2.0, -1.0, 0.0, 1.0, 2.0];
        // values from trieste implementation
        let expected = vec![-4.7687836, -2.4851208, -0.9189385, 0.08002624, 0.69738346];
        for (expect, val) in zip(expected, vals) {
            assert_abs_diff_eq!(expect, log_ei_helper(val), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_d_log_ei() {
        let x = Array1::linspace(-10., 10., 100);
        write_npy("logei_x.npy", &x).expect("save x");

        let fx = x.mapv(log_ei_helper);
        write_npy("logei_fx.npy", &fx).expect("save fx");

        let dfx = x.mapv(d_log_ei_helper);
        write_npy("logei_dfx.npy", &dfx).expect("save dfx");

        let gradfx = x.mapv(|x| finite_diff_log_ei(x, 1e-6));
        write_npy("logei_gradfx.npy", &gradfx).expect("save dfx");
    }

    fn finite_diff_log_ei(u: f64, eps: f64) -> f64 {
        (log_ei_helper(u + eps) - log_ei_helper(u - eps)) / (2.0 * eps)
    }
}
