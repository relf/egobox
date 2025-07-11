/// GBNM implementation translated from <https://github.com/ojdo/gbnm>
///
use ndarray::{s, Array1, Array2, Axis};
use ndarray_rand::*;

#[derive(Debug, Clone)]
pub struct GbnmOptions {
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

#[derive(Debug)]
#[allow(dead_code)]
pub struct GbnmResult {
    pub x: Array1<f64>,
    pub fval: f64,
    pub used_points: Array2<f64>,
    pub used_vals: Vec<f64>,
    pub reason: Vec<String>,
    pub n_eval: usize,
}

pub trait Func<U>: Fn(&[f64], &mut U) -> f64 {}
impl<T, U> Func<U> for T where T: Fn(&[f64], &mut U) -> f64 {}

pub fn gbnm<F: Func<U>, U: Clone>(
    fun: F,
    xmin: &Array1<f64>,
    xmax: &Array1<f64>,
    args: U,
    options: GbnmOptions,
) -> GbnmResult {
    let ndim = xmin.len();
    let xrange = xmax - xmin;

    let mut n_eval = 0;
    let mut used_points = Array2::zeros((ndim, 2 * options.max_restarts));
    let mut used_vals = vec![f64::INFINITY; 2 * options.max_restarts];
    let mut reason = vec!["".to_string(); options.max_restarts];

    for i_restart in 0..options.max_restarts {
        let initial_point = probabilistic_restart(&used_points, xmin, &xrange, options.n_points);
        let a = (0.02 + 0.08 * rand::random::<f64>())
            * xrange.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let (mut splx, mut fval, evals) =
            init_simplex(&initial_point, a, &fun, args.clone(), xmin, xmax);
        n_eval += evals;

        used_points
            .slice_mut(s![.., 2 * i_restart])
            .assign(&splx.column(0));
        used_vals[2 * i_restart] = fval[0];
        reason[i_restart] = "hit maxIter".into();

        for _ in 0..options.max_iter {
            let mut indices: Vec<_> = (0..=ndim).collect();
            indices.sort_by(|&i, &j| fval[i].partial_cmp(&fval[j]).unwrap());
            fval = indices.iter().map(|&i| fval[i]).collect();
            splx = splx.select(Axis(1), &indices);

            let x_worst = splx.column(ndim).to_owned();
            let x_best = splx.column(0).to_owned();
            let x_cent = splx.slice(s![.., ..ndim]).mean_axis(Axis(1)).unwrap();

            if standard_deviation(&fval) < options.epsilon {
                reason[i_restart] = "convergence (fVals similar)".into();
                break;
            }

            if max_relative_diff(&splx, &xrange) < options.ssigma {
                reason[i_restart] = "convergence (simplex small)".into();
                break;
            }

            if n_eval >= options.max_evals {
                reason[i_restart] = "hit maxEvals".into();
                break;
            }

            let xr = clip(
                &(x_cent.clone() + options.alpha * (&x_cent - &x_worst)),
                xmin,
                xmax,
            );
            let fr = fun(&xr.to_vec(), &mut args.clone());
            n_eval += 1;

            if fr < fval[0] {
                let xe = clip(
                    &(x_cent.clone() + options.gamma * (&xr - &x_cent)),
                    xmin,
                    xmax,
                );
                let fe = fun(&xe.to_vec(), &mut args.clone());
                n_eval += 1;
                if fe < fr {
                    splx.slice_mut(s![.., ndim]).assign(&xe);
                    fval[ndim] = fe;
                } else {
                    splx.slice_mut(s![.., ndim]).assign(&xr);
                    fval[ndim] = fr;
                }
            } else if fr <= fval[ndim - 1] {
                splx.slice_mut(s![.., ndim]).assign(&xr);
                fval[ndim] = fr;
            } else {
                if fr < fval[ndim] {
                    splx.slice_mut(s![.., ndim]).assign(&xr);
                    fval[ndim] = fr;
                }
                let xc = x_cent.clone() + options.beta * (&x_worst - &x_cent);
                let fc = fun(&xc.to_vec(), &mut args.clone());
                n_eval += 1;

                if fc > fval[ndim] {
                    for (k, mut column) in splx.columns_mut().into_iter().enumerate().skip(1) {
                        let xk = 0.5 * (&column + &x_best);
                        let fk = fun(&xk.to_vec(), &mut args.clone());
                        n_eval += 1;
                        column.assign(&xk);
                        fval[k] = fk;
                    }
                } else {
                    splx.slice_mut(s![.., ndim]).assign(&xc);
                    fval[ndim] = fc;
                }
            }
        }

        used_points
            .slice_mut(s![.., 2 * i_restart + 1])
            .assign(&splx.column(0));
        used_vals[2 * i_restart + 1] = fval[0];
        if n_eval >= options.max_evals {
            reason[i_restart] = "hit maxEvals".into();
            break;
        }
    }

    let (min_idx, &min_val) = used_vals
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let x = used_points.column(min_idx).to_owned();

    GbnmResult {
        x,
        fval: min_val,
        used_points,
        used_vals,
        reason,
        n_eval,
    }
}

fn probabilistic_restart(
    used: &Array2<f64>,
    xmin: &Array1<f64>,
    xrange: &Array1<f64>,
    n_points: usize,
) -> Array1<f64> {
    let ndim = xmin.len();
    let mut best_prob = f64::INFINITY;
    let mut best_point = xmin.clone();

    for _ in 0..n_points {
        let random_point =
            xmin + &(xrange * &Array1::from_shape_fn(ndim, |_| rand::random::<f64>()));
        let prob = gauss(&random_point, used, xrange);
        if prob < best_prob {
            best_prob = prob;
            best_point = random_point;
        }
    }
    best_point
}

fn init_simplex<F: Func<U>, U: Clone>(
    x0: &Array1<f64>,
    a: f64,
    fun: F,
    args: U,
    xmin: &Array1<f64>,
    xmax: &Array1<f64>,
) -> (Array2<f64>, Vec<f64>, usize) {
    let ndim = x0.len();
    let mut splx = Array2::<f64>::zeros((ndim, ndim + 1));
    let mut fval = vec![0.0; ndim + 1];

    let p = a * ((ndim as f64 + 1.0).sqrt() + ndim as f64 - 1.0) / (ndim as f64 * 2f64.sqrt());
    let q = a * ((ndim as f64 + 1.0).sqrt() - 1.0) / (ndim as f64 * 2f64.sqrt());

    splx.column_mut(0).assign(x0);

    for k in 0..ndim {
        splx.column_mut(k + 1).assign(x0);
        splx[(k, k + 1)] += p;
        for i in 0..ndim {
            if i != k {
                splx[(i, k + 1)] += q;
            }
        }
        let mut col = splx.column_mut(k + 1);
        col.zip_mut_with(xmin, |v, &min| *v = v.max(min));
        col.zip_mut_with(xmax, |v, &max| *v = v.min(max));
    }

    let mut n_eval = 0;
    for (k, column) in splx.columns().into_iter().enumerate() {
        fval[k] = fun(&column.to_vec(), &mut args.clone());
        n_eval += 1;
    }

    (splx, fval, n_eval)
}

fn gauss(x: &Array1<f64>, points: &Array2<f64>, xrange: &Array1<f64>) -> f64 {
    let glp = 0.01;
    let sigma = Array1::from_iter(xrange.iter().map(|&r| glp * r * r));
    let mut prob = 0.0;
    for point in points.axis_iter(Axis(1)) {
        if point.iter().all(|&v| v == 0.0) {
            break;
        }
        let delta = x - &point;
        let exp_term = delta
            .iter()
            .zip(sigma.iter())
            .map(|(&d, &s)| d * d / s)
            .sum::<f64>();
        prob += (-0.5 * exp_term).exp();
    }
    prob
}

fn standard_deviation(values: &[f64]) -> f64 {
    let mean = values.iter().copied().sum::<f64>() / values.len() as f64;
    (values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}

fn max_relative_diff(splx: &Array2<f64>, xrange: &Array1<f64>) -> f64 {
    let max_vals = splx.fold_axis(Axis(1), f64::MIN, |a, &b| a.max(b));
    let min_vals = splx.fold_axis(Axis(1), f64::MAX, |a, &b| a.min(b));
    max_vals
        .iter()
        .zip(min_vals.iter())
        .zip(xrange.iter())
        .map(|((&max, &min), &r)| (max - min) / r)
        .fold(0.0, |a, b| a.max(b))
}

fn clip(x: &Array1<f64>, xmin: &Array1<f64>, xmax: &Array1<f64>) -> Array1<f64> {
    x.iter()
        .zip(xmin.iter())
        .zip(xmax.iter())
        .map(|((&v, &min), &max)| v.max(min).min(max))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, array};

    #[test]
    fn test_gbnm_minimization() {
        // Example of using the Gbnm optimizer with a simple quadratic function
        let fun = |x: &[f64], _u: &mut Option<()>| x.iter().map(|&xi| xi * xi).sum();

        let xmin = arr1(&[-5.0, -5.0]);
        let xmax = arr1(&[5.0, 5.0]);

        let options = GbnmOptions {
            max_restarts: 1,
            max_evals: 100,
            n_points: 10,
            max_iter: 50,
            alpha: 1.0,
            beta: 0.5,
            gamma: 2.0,
            epsilon: 1e-6,
            ssigma: 1e-6,
        };

        let result = gbnm(fun, &xmin, &xmax, None, options);

        let expected = array![0., 0.];
        assert!(expected
            .iter()
            .zip(result.x.iter())
            .all(|(a, b)| (a - b).abs() < 0.05)); // The solution should be near the origin
        assert_abs_diff_eq!(result.fval, 0., epsilon = 1e-3); // We expect the minimum value to be near 0
        assert_abs_diff_eq!(result.x, array![0., 0.], epsilon = 5e-3); // The minimum should be at (0, 0)
    }

    // #[test]
    // fn test_gbnm_with_custom_function() {
    //     // Example with a custom function to minimize
    //     let fun = |x: &[f64], _u: &mut ()| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);

    //     let xmin = arr1(&[-10.0, -10.0]);
    //     let xmax = arr1(&[10.0, 10.0]);

    //     let options = GbnmOptions {
    //         max_restarts: 1,
    //         max_evals: 100,
    //         n_points: 20,
    //         max_iter: 50,
    //         alpha: 1.0,
    //         beta: 0.5,
    //         gamma: 2.0,
    //         epsilon: 1e-6,
    //         ssigma: 1e-6,
    //     };

    //     let result = gbnm(fun, &xmin, &xmax, (), options);

    //     assert_abs_diff_eq!(result.fval, 0., epsilon = 1e-4); // We expect the minimum value to be near 0
    //     assert_abs_diff_eq!(result.x, array![2., 3.], epsilon = 5e-3); // The minimum should be at (2, 3)
    // }

    #[test]
    fn test_multiple_restarts() {
        // Test with multiple restarts to verify different initializations
        let fun = |x: &[f64], _u: &mut ()| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);

        let xmin = arr1(&[-5.0, -5.0]);
        let xmax = arr1(&[5.0, 5.0]);

        let options = GbnmOptions {
            max_restarts: 3,
            max_evals: 100,
            n_points: 10,
            max_iter: 50,
            alpha: 1.0,
            beta: 0.5,
            gamma: 2.0,
            epsilon: 1e-6,
            ssigma: 1e-6,
        };

        let result = gbnm(fun, &xmin, &xmax, (), options);

        assert_abs_diff_eq!(result.fval, 0., epsilon = 1e-4); // We expect the minimum value to be near 0
        assert_abs_diff_eq!(result.x, array![1., 2.], epsilon = 5e-3); // The minimum should be at (1, 2)
    }

    #[test]
    fn test_max_evals() {
        // Test hitting max evaluations limit
        let fun = |x: &[f64], _u: &mut ()| x.iter().map(|&xi| xi * xi).sum();

        let xmin = arr1(&[-5.0, -5.0]);
        let xmax = arr1(&[5.0, 5.0]);

        let options = GbnmOptions {
            max_restarts: 1,
            max_evals: 10, // Set a small number to trigger max evaluations
            n_points: 5,
            max_iter: 50,
            alpha: 1.0,
            beta: 0.5,
            gamma: 2.0,
            epsilon: 1e-6,
            ssigma: 1e-6,
        };

        let result = gbnm(fun, &xmin, &xmax, (), options);

        assert_eq!(result.reason[0], "hit maxEvals".to_string()); // We should hit the max evaluations limit
    }

    /// This function has a known minimum at (2,3)(2,3).
    fn objective_function(x: &[f64], _u: &mut ()) -> f64 {
        (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2)
    }

    #[test]
    fn test_main() {
        let xmin = arr1(&[-5.0, -5.0]); // Lower bounds of the search space
        let xmax = arr1(&[5.0, 5.0]); // Upper bounds of the search space

        // Define the options for the optimizer
        let options = GbnmOptions {
            max_restarts: 3, // Number of restarts
            max_evals: 100,  // Maximum evaluations per restart
            n_points: 10,    // Number of points to sample for the initial simplex
            max_iter: 50,    // Maximum iterations per restart
            alpha: 1.0,      // Reflection coefficient
            beta: 0.5,       // Contraction coefficient
            gamma: 2.0,      // Expansion coefficient
            epsilon: 1e-6,   // Convergence threshold based on function values
            ssigma: 1e-6,    // Simplex size convergence threshold
        };

        // Run the optimization
        let result = gbnm(objective_function, &xmin, &xmax, (), options);

        // Print the results
        println!("Optimal x: {:?}", result.x);
        println!("Optimal f(x): {}", result.fval);
        println!("Reason for termination: {:?}", result.reason);
    }
}
