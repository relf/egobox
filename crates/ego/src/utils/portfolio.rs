use std::collections::HashMap;

use linfa::traits::Transformer;

use linfa_clustering::Dbscan;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2, s};

/// Generate `num` points spaced evenly on a log scale between `start` and `end`.
pub fn logspace(start: f64, end: f64, num: usize) -> Array1<f64> {
    assert!(start > 0.0, "logspace requires start > 0");
    assert!(end > 0.0, "logspace requires end > 0");
    assert!(num >= 2, "logspace requires at least 2 points");

    let log_start = start.log10();
    let log_end = end.log10();
    Array1::from_iter((0..num).map(|i| {
        let t = i as f64 / (num as f64 - 1.0);
        10f64.powf(log_start + t * (log_end - log_start))
    }))
}

pub fn select_cluster_member(xdat: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
    let clusters = Dbscan::params(3)
        .tolerance(1e-1)
        .transform(&xdat.to_owned())
        .unwrap();
    let mut dict = HashMap::new();
    for (i, c) in clusters.iter().enumerate() {
        match c {
            None => (),
            Some(label) => dict
                .entry(label)
                .or_insert(vec![])
                .push(xdat.row(i).to_owned()),
        }
    }

    let mut res = Array2::zeros((dict.len(), xdat.ncols()));
    for (i, (_label, points)) in dict.iter().enumerate() {
        let centroid = points.first().unwrap().to_owned();
        res.slice_mut(s![i, ..]).assign(&centroid);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::*;

    #[test]
    fn test_logspace_endpoints() {
        let vals = logspace(0.1, 100.0, 5);
        assert!((vals[0] - 0.1).abs() < 1e-12);
        assert!((vals[4] - 100.0).abs() < 1e-12);
    }

    #[test]
    fn test_logspace_length() {
        let vals = logspace(0.1, 100.0, 13);
        println!("{vals}");
        assert_eq!(vals.len(), 13);
    }

    #[test]
    fn test_logspace_monotonic_increasing() {
        let vals = logspace(1e-3, 1e3, 20);
        for i in 1..vals.len() {
            assert!(vals[i] > vals[i - 1], "Values must be strictly increasing");
        }
    }

    #[test]
    fn test_logspace_known_values() {
        // logspace(1, 100, 3) should give [1, 10, 100]
        let vals = logspace(1.0, 100.0, 3);
        let expected = [1.0, 10.0, 100.0];
        for (v, e) in vals.iter().zip(expected.iter()) {
            assert!((*v - *e).abs() < 1e-12);
        }
    }

    #[test]
    fn test_clustering() {
        let x = array![
            [0.13],
            [0.70],
            [0.72],
            [0.14],
            [0.15],
            [0.71],
            [0.16],
            [0.73]
        ];
        let cluster_memberships = select_cluster_member(&x);
        println!("cluster_memberships = {cluster_memberships :?}");
    }
}
