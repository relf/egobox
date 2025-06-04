use egobox_gp::{correlation_models::*, ThetaTuning};
use egobox_moe::CorrelationSpec;

/// Theta bounds
#[allow(dead_code)]
pub fn theta_bounds(dim: usize, spec: CorrelationSpec) -> (f64, f64) {
    let (theta_inf, theta_sup) = (0.1f64, 0.5f64);
    let (theta_inf, theta_sup) = if spec.contains(CorrelationSpec::SQUAREDEXPONENTIAL) {
        let influence: (f64, f64) = SquaredExponentialCorr::default().theta_influence_factors();
        (theta_inf.min(influence.0), theta_sup.max(influence.1))
    } else {
        (theta_inf, theta_sup)
    };
    let (theta_inf, theta_sup) = if spec.contains(CorrelationSpec::ABSOLUTEEXPONENTIAL) {
        let influence: (f64, f64) = AbsoluteExponentialCorr::default().theta_influence_factors();
        (theta_inf.min(influence.0), theta_sup.max(influence.1))
    } else {
        (theta_inf, theta_sup)
    };
    let (theta_inf, theta_sup) = if spec.contains(CorrelationSpec::MATERN32) {
        let influence: (f64, f64) = Matern32Corr::default().theta_influence_factors();
        (theta_inf.min(influence.0), theta_sup.max(influence.1))
    } else {
        (theta_inf, theta_sup)
    };
    let (theta_inf, theta_sup) = if spec.contains(CorrelationSpec::MATERN32) {
        let influence: (f64, f64) = Matern32Corr::default().theta_influence_factors();
        (theta_inf.min(influence.0), theta_sup.max(influence.1))
    } else {
        (theta_inf, theta_sup)
    };

    if dim < 10 {
        (
            ThetaTuning::<f64>::DEFAULT_BOUNDS.0,
            ThetaTuning::<f64>::DEFAULT_BOUNDS.1,
        )
    } else {
        let d = dim as f64;
        let s = 1. / f64::sqrt(12.); // uniform design on [0, 1]^d standard deviation
        let k = 9. / 5.; // uniform distribution kurtosis
        let interval = 1.96 * f64::sqrt(2. * (k + 1.) * d);
        let rmin = f64::sqrt(2. * d - interval);
        let rmax = f64::sqrt(2. * d + interval);
        dbg!("rmin = {}, rmax={}", rmin, rmax);
        let lmin = s * rmin * theta_inf;
        let lmax = s * rmax * theta_sup;
        dbg!("lmin = {}, lmax={}", lmin, lmax);
        let tlo = 1. / lmax;
        let tup = 1. / lmin;
        (tlo, tup)
    }
}

#[cfg(test)]
mod tests {
    use paste::paste;

    use super::*;

    #[allow(unused_macros)]
    macro_rules! test_theta_bounds {
        ($dim:expr, $corr:ident ) => {
            paste! {

                #[test]
                fn [<test_theta_bounds_  $corr:snake _ $dim:snake>]() {
                    let bounds = theta_bounds($dim, CorrelationSpec::[< $corr:upper >]);
                    println!("bounds({}) = {:?}", $dim, bounds);
                    assert!(bounds.0 <= bounds.1);
                }
            }
        };
    }

    test_theta_bounds!(1, SquaredExponential);
    test_theta_bounds!(10, SquaredExponential);
    test_theta_bounds!(50, SquaredExponential);
    test_theta_bounds!(100, SquaredExponential);
    test_theta_bounds!(200, SquaredExponential);
    test_theta_bounds!(1, AbsoluteExponential);
    test_theta_bounds!(10, AbsoluteExponential);
    test_theta_bounds!(50, AbsoluteExponential);
    test_theta_bounds!(100, AbsoluteExponential);
    test_theta_bounds!(200, AbsoluteExponential);
    test_theta_bounds!(1, Matern32);
    test_theta_bounds!(10, Matern32);
    test_theta_bounds!(50, Matern32);
    test_theta_bounds!(100, Matern32);
    test_theta_bounds!(200, Matern32);
    test_theta_bounds!(1, Matern52);
    test_theta_bounds!(10, Matern52);
    test_theta_bounds!(50, Matern52);
    test_theta_bounds!(100, Matern52);
    test_theta_bounds!(200, Matern52);
}
