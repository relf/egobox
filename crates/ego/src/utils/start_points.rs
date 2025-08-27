use ndarray::{Array1, Array2, Axis, stack};
use std::cmp::Ordering;

pub fn start_points(x: &Array2<f64>, xl: &Array1<f64>, xu: &Array1<f64>) -> Array2<f64> {
    let n = x.nrows();
    let d = x.ncols();
    let xrange = xu - xl;

    // Collect all (i, j, distance) triplets
    let mut pairs = Vec::new();
    for i in 1..n {
        for j in 0..i {
            let diff = &x.row(i) - &x.row(j);
            let scaled = &diff / &xrange;
            let dist = scaled.dot(&scaled).sqrt();
            pairs.push((i, j, dist));
        }
    }

    // Sort pairs by distance
    pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

    // Determine good midpoints
    let mut xstart: Vec<Array1<f64>> = Vec::new();

    for (i, j, _) in pairs {
        // Consider xij middle of [xi, xj] to be a starting point
        let xi = x.row(i);
        let xj = x.row(j);
        let xij = (xi.to_owned() + xj) / 2.0;

        // Distance from xi (and xj)
        let d_ij = ((xi.to_owned() - &xij) / &xrange)
            .mapv(|v| v.powi(2))
            .sum()
            .sqrt();

        // xij good to be added
        let mut good = true;

        // Discard xij if there is a training point xk closer than xi and xj from it
        for k in 0..n {
            if k != i && k != j {
                let xk = x.row(k);
                let d_k = ((&xk - &xij) / &xrange).mapv(|v| v.powi(2)).sum().sqrt();
                if d_k < d_ij {
                    good = false;
                    break;
                }
            }
        }

        // Discard xij if there is already
        for xk in &xstart {
            let d_k = ((xk.to_owned() - &xij) / &xrange)
                .mapv(|v| v.powi(2))
                .sum()
                .sqrt();
            if d_k < d_ij {
                good = false;
                break;
            }
        }

        if good {
            xstart.push(xij);
        }
    }

    if xstart.is_empty() {
        Array2::<f64>::zeros((0, d))
    } else {
        let vstarts = xstart.iter().map(|p| p.view()).collect::<Vec<_>>();
        stack(Axis(0), &vstarts).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray_rand::{RandomExt, rand::SeedableRng, rand_distr::Uniform};
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    fn test_midpoint_between_two_distant_points() {
        let x = array![[0.1, 0.2], [0.9, 0.8]];
        let xl = array![0.0, 0.0];
        let xu = array![1.0, 1.0];

        let result = start_points(&x, &xl, &xu);
        assert_eq!(result.nrows(), 1);
        assert_eq!(result.shape(), &[1, 2]);

        let expected = array![[0.5, 0.5]];
        assert!(result.abs_diff_eq(&expected, 1e-12));
    }

    #[test]
    fn test_multiple_points() {
        let x = array![[0.1, 0.2], [0.9, 0.2], [0.5, 0.8]];
        let xl = array![0.0, 0.0];
        let xu = array![1.0, 1.0];

        let result = start_points(&x, &xl, &xu);
        println!("result={result}");
        assert!(result.nrows() >= 1);
    }

    #[test]
    fn test_n_start() {
        for n in 1..50 {
            let mut rng = Xoshiro256Plus::seed_from_u64(42);
            let xt = Array2::random_using((n, 2), Uniform::new(0., 1.), &mut rng);
            let xl = array![0.0, 0.0];
            let xu = array![1.0, 1.0];

            let result = start_points(&xt, &xl, &xu);
            println!("n={} n_start={}", n, result.len());
        }
    }
}
