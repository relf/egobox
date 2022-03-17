#![allow(dead_code)]
use doe::{SamplingMethod, LHS};
use ndarray::{array, s, Array, Array2, Axis, Zip};
use ndarray_rand::rand::SeedableRng;
use ndarray_stats::QuantileExt;
use rand_isaac::Isaac64Rng;

#[derive(Debug, Clone)]
pub enum Vspec {
    Cont(f64, f64),
    Int(i32, i32),
    Ord(Vec<i32>),
    Enum(Vec<String>),
}

fn unfold_xlimits_with_continuous_limits(spec: &[Vspec]) -> Array2<f64> {
    let mut res = vec![];
    spec.iter().for_each(|s| match s {
        Vspec::Cont(lb, ub) => res.extend([*lb, *ub]),
        Vspec::Int(lb, ub) => res.extend([*lb as f64, *ub as f64]),
        Vspec::Ord(v) => res.extend([v[0] as f64, v[(v.len() - 1)] as f64]),
        Vspec::Enum(v) => (0..v.len()).for_each(|_| res.extend([0., 1.])),
    });
    Array::from_shape_vec((res.len() / 2, 2), res).unwrap()
}

fn fold_with_enum_index(spec: &[Vspec], x: &Array2<f64>) -> Array2<f64> {
    let mut xfold = Array::zeros((x.nrows(), spec.len()));
    let mut unfold_index = 0;
    Zip::indexed(xfold.columns_mut()).for_each(|j, mut col| match &spec[j] {
        Vspec::Cont(_, _) | Vspec::Int(_, _) | Vspec::Ord(_) => {
            col.assign(&x.column(unfold_index));
            unfold_index += 1;
        }
        Vspec::Enum(v) => {
            let xenum = x.slice(s![.., j..j + v.len()]);
            let argmaxx = xenum.map_axis(Axis(1), |row| row.argmax().unwrap() as f64);
            col.assign(&argmaxx);
            unfold_index += v.len();
        }
    });
    xfold
}

fn take_closest(v: &[i32], val: f64) -> i32 {
    let idx = Array::from_vec(v.to_vec())
        .map(|refval| (val - *refval as f64).abs())
        .argmin()
        .unwrap();
    v[idx]
}

fn cast_to_discrete_values(spec: &[Vspec], x: &mut Array2<f64>) {
    let mut xcol = 0;
    spec.iter().for_each(|s| match s {
        Vspec::Cont(_, _) => xcol += 1,
        Vspec::Int(_, _) => {
            let xround = x.column(xcol).mapv(|v| v.round()).to_owned();
            x.column_mut(xcol).assign(&xround);
            xcol += 1;
        }
        Vspec::Ord(v) => {
            let xround = x
                .column(xcol)
                .mapv(|val| take_closest(v, val) as f64)
                .to_owned();
            x.column_mut(xcol).assign(&xround);
            xcol += 1;
        }
        Vspec::Enum(v) => {
            let mut xenum = x.slice_mut(s![.., xcol..xcol + v.len()]);
            let argmaxx = xenum.map_axis(Axis(1), |row| row.argmax().unwrap());
            Zip::from(xenum.rows_mut())
                .and(&argmaxx)
                .for_each(|mut row, &m| {
                    let mut xcast = Array::zeros(v.len());
                    xcast[m] = 1.;
                    row.assign(&xcast);
                });
            xcol += v.len();
        }
    });
}

pub struct MixintSampling {
    method: LHS<f64, Isaac64Rng>,
    spec: Vec<Vspec>,
    work_in_folded_space: bool,
}

impl MixintSampling {
    fn new(spec: Vec<Vspec>) -> Self {
        MixintSampling {
            method: LHS::new(&unfold_xlimits_with_continuous_limits(&spec)),
            spec: spec.clone(),
            work_in_folded_space: true,
        }
    }
}

impl SamplingMethod<f64> for MixintSampling {
    fn sampling_space(&self) -> &Array2<f64> {
        self.method.sampling_space()
    }

    fn normalized_sample(&self, ns: usize) -> Array2<f64> {
        self.method.normalized_sample(ns)
    }

    fn sample(&self, ns: usize) -> Array2<f64> {
        let mut doe = self.method.sample(ns);
        cast_to_discrete_values(&self.spec, &mut doe);
        if self.work_in_folded_space {
            fold_with_enum_index(&self.spec, &doe)
        } else {
            doe
        }
    }
}

// struct MixintSurrogate {}

pub struct MixintContext {
    spec: Vec<Vspec>,
}

impl MixintContext {
    fn new(spec: Vec<Vspec>) -> Self {
        MixintContext { spec }
    }

    fn create_sampling(&self, seed: Option<u64>) -> MixintSampling {
        let lhs = seed.map_or(
            LHS::new(&unfold_xlimits_with_continuous_limits(&self.spec)),
            |seed| {
                let rng = Isaac64Rng::seed_from_u64(seed);
                LHS::new(&unfold_xlimits_with_continuous_limits(&self.spec)).with_rng(rng)
            },
        );
        MixintSampling {
            method: lhs,
            spec: self.spec.clone(),
            work_in_folded_space: true,
        }
    }

    // fn create_surrogate() -> MixintSurrogate {
    //     MixintSurrogate {}
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test() {
        let specs = vec![
            Vspec::Cont(-10.0, 10.0),
            Vspec::Enum(vec![
                "blue".to_string(),
                "red".to_string(),
                "green".to_string(),
            ]),
            Vspec::Int(-10, 10),
            Vspec::Ord(vec![1, 3, 5, 8]),
        ];

        let mixi = MixintContext::new(specs);
        let mixi_lhs = mixi.create_sampling(Some(0));
        let actual = mixi_lhs.sample(10);
        let expected = array![
            [2.5506163720107278, 0.0, -9.0, 1.0],
            [-5.6951210599033315, 2.0, 4.0, 1.0],
            [8.00413910535675, 2.0, -5.0, 5.0],
            [7.204222718105676, 1.0, -3.0, 5.0],
            [4.937191086579546, 0.0, 4.0, 3.0],
            [-3.486137077103643, 2.0, -2.0, 5.0],
            [-6.013086019937296, 0.0, -8.0, 8.0],
            [1.434149013952382, 0.0, 7.0, 5.0],
            [-8.074280304556137, 1.0, 1.0, 3.0],
            [-1.4935174827024618, 1.0, 9.0, 8.0],
        ];
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }
}
