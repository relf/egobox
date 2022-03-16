use doe::{SamplingMethod, LHS};
use ndarray::{s, Array, Array2, Axis, Zip};
use ndarray_stats::QuantileExt;
use rand_isaac::Isaac64Rng;

#[derive(Debug, Clone)]
#[allow(dead_code)]
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
    println!("{:?}", x);
    spec.iter().for_each(|s| match s {
        Vspec::Cont(_, _) => xcol += 1,
        Vspec::Int(_, _) => {
            //let xround = x.column(j).mapv(|v| v.round()).to_owned();
            println!("{} {:?}", xcol, x.column(xcol));
            let xround = x.column(xcol).mapv(|v| v.round()).to_owned();
            println!("{:?}", xround);
            x.column_mut(xcol).assign(&xround);
            xcol += 1;
        }
        Vspec::Ord(v) => {
            println!("{} {:?}", xcol, x.column(xcol));
            let xround = x
                .column(xcol)
                .mapv(|val| take_closest(v, val) as f64)
                .to_owned();
            println!("{:?}", xround);
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

struct MixintSampling {
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

struct MixintContext {
    spec: Vec<Vspec>,
}

impl MixintContext {
    fn new(spec: Vec<Vspec>) -> Self {
        MixintContext { spec }
    }

    fn create_sampling(&self) -> MixintSampling {
        MixintSampling {
            method: LHS::new(&unfold_xlimits_with_continuous_limits(&self.spec)),
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
        let mixi_lhs = mixi.create_sampling();
        let doe = mixi_lhs.sample(10);
        println!("{:?}", doe);
    }
}
