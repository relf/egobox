use crate::utils::misc::{norm_cdf, norm_pdf};
use libm::{erfc, exp, expm1, log, log1p};

const INV_SQRT_2: f64 = 0.7071067811865475;
const LOG_2PI_OVER_2: f64 = 0.9189385332046727; // log(2π)/2
const LOG_PI_OVER_2_ALL_OVER_2: f64 = 0.2257913526447274; // log(π/2)/2
const INV_SQRT_EPSILON: f64 = 1.0 / 1e-6;

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
        log(norm_pdf(u) + u * norm_cdf(u))
    } else {
        let log_phi_u = -0.5 * u * u - LOG_2PI_OVER_2;

        let log_term = if u > -INV_SQRT_EPSILON {
            let w = log(erfcx(-INV_SQRT_2 * u) * u.abs()) + LOG_PI_OVER_2_ALL_OVER_2;
            log1mexp(w)
        } else {
            -2.0 * log(u.abs())
        };

        log_phi_u + log_term
    }
}

fn log1mexp_prime(w: f64) -> f64 {
    // derivative wrt w: -e^w / (1 - e^w)
    -w.exp() / (1.0 - w.exp())
}

fn w_and_w_prime(u: f64) -> (f64, f64) {
    let z = -INV_SQRT_2 * u;
    let val_erfcx = erfcx(z);
    let erfcx_prime = 2.0 * z * val_erfcx - 2.0 / std::f64::consts::PI.sqrt();

    let w = log(val_erfcx * u.abs()) + LOG_PI_OVER_2_ALL_OVER_2;
    let w_prime = (erfcx_prime * -INV_SQRT_2 / val_erfcx) + 1.0 / u;
    (w, w_prime)
}

fn log1mexp_w_derivative(u: f64) -> f64 {
    let (w, w_prime) = w_and_w_prime(u);
    log1mexp_prime(w) * w_prime
}

pub fn d_log_ei_helper(u: f64) -> f64 {
    if u > -1.0 {
        let numerator = norm_cdf(u);
        let denominator = log_ei_helper(u).exp();
        numerator / denominator
    } else {
        let d_log_phi_u = -u;

        let d_log_term = if u > -INV_SQRT_EPSILON {
            log1mexp_w_derivative(u)
        } else {
            -2. / u
        };

        d_log_phi_u + d_log_term
    }
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
        write_npy("logei_fdifffx.npy", &gradfx).expect("save dfx");

        assert_abs_diff_eq!(dfx, gradfx, epsilon = 1e-3);
    }

    fn finite_diff_log_ei(u: f64, eps: f64) -> f64 {
        (log_ei_helper(u + eps) - log_ei_helper(u - eps)) / (2. * eps)
    }
}
