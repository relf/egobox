//! This library implements continuous relaxation functions,
//! it is a port of [SMT mixed integer module](https://smt.readthedocs.io/en/latest/_src_docs/applications/mixed_integer.html)

#![allow(dead_code)]
use egobox_doe::{Lhs, SamplingMethod};
use egobox_moe::{Moe, MoeFit, MoeParams, RegressionSpec, Result, Surrogate};
use linfa::{traits::Fit, Dataset};
use ndarray::{s, Array, Array2, ArrayView2, Axis, Zip};
use ndarray_rand::rand::SeedableRng;
use ndarray_stats::QuantileExt;
use rand_isaac::Isaac64Rng;

#[cfg(feature = "persistent")]
use egobox_moe::MoeError;
#[cfg(feature = "persistent")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "persistent")]
use std::fs;
#[cfg(feature = "persistent")]
use std::io::Write;

/// An enumeration to define the type of an input variable component
/// with its domain definition
#[derive(Debug, Clone)]
#[cfg_attr(feature = "persistent", derive(Serialize, Deserialize))]
pub enum Xtype {
    /// Continuous variable in [lower bound, upper bound]
    Cont(f64, f64),
    /// Integer variable in lower bound .. upper bound
    Int(i32, i32),
    /// An Ordered variable in { int_1, int_2, ... int_n }
    Ord(Vec<i32>),
    /// An Enum variable in { str_1, str_2, ..., str_n }
    Enum(Vec<String>),
}

/// Expand xlimits to add continuous dimensions for enumerate x features
/// Each level of an enumerate gives a new continuous dimension in [0, 1].
/// Each integer dimensions are relaxed continuously.
pub fn unfold_xlimits_with_continuous_limits(xtypes: &[Xtype]) -> Array2<f64> {
    let mut res = vec![];
    xtypes.iter().for_each(|s| match s {
        Xtype::Cont(lb, ub) => res.extend([*lb, *ub]),
        Xtype::Int(lb, ub) => res.extend([*lb as f64, *ub as f64]),
        Xtype::Ord(v) => res.extend([v[0] as f64, v[(v.len() - 1)] as f64]),
        Xtype::Enum(v) => (0..v.len()).for_each(|_| res.extend([0., 1.])),
    });
    Array::from_shape_vec((res.len() / 2, 2), res).unwrap()
}

/// Reduce categorical inputs from discrete unfolded space to
/// initial x dimension space where categorical x dimensions are valued by the index
/// in the corresponding enumerate list.
/// For instance, if an input dimension is typed ["blue", "red", "green"] a sample/row of
/// the input x may contain the mask [..., 0, 0, 1, ...] which will be contracted in [..., 2, ...]
/// meaning the "green" value.
/// This function is the opposite of unfold_with_enum_mask().
pub fn fold_with_enum_index(xtypes: &[Xtype], x: &ArrayView2<f64>) -> Array2<f64> {
    let mut xfold = Array::zeros((x.nrows(), xtypes.len()));
    let mut unfold_index = 0;
    Zip::indexed(xfold.columns_mut()).for_each(|j, mut col| match &xtypes[j] {
        Xtype::Cont(_, _) | Xtype::Int(_, _) | Xtype::Ord(_) => {
            col.assign(&x.column(unfold_index));
            unfold_index += 1;
        }
        Xtype::Enum(v) => {
            let xenum = x.slice(s![.., j..j + v.len()]);
            let argmaxx = xenum.map_axis(Axis(1), |row| row.argmax().unwrap() as f64);
            col.assign(&argmaxx);
            unfold_index += v.len();
        }
    });
    xfold
}

/// Compute dimension when all variables are continuously relaxed
fn compute_unfolded_dimension(xtypes: &[Xtype]) -> usize {
    xtypes
        .iter()
        .map(|s| match s {
            Xtype::Enum(v) => v.len(),
            _ => 1,
        })
        .reduce(|acc, l| -> usize { acc + l })
        .unwrap()
}

/// Expand categorical inputs from initial x dimension space where categorical x dimensions
/// are valued by the index in the corresponding enumerate list to the discrete unfolded space.
/// For instance, if an input dimension is typed ["blue", "red", "green"] a sample/row of
/// the input x may contain [..., 2, ...] which will be expanded in [..., 0, 0, 1, ...].
/// This function is the opposite of fold_with_enum_index().
fn unfold_with_enum_mask(xtypes: &[Xtype], x: &ArrayView2<f64>) -> Array2<f64> {
    let mut xunfold = Array::zeros((x.nrows(), compute_unfolded_dimension(xtypes)));
    let mut unfold_index = 0;
    xtypes.iter().for_each(|s| match s {
        Xtype::Cont(_, _) | Xtype::Int(_, _) | Xtype::Ord(_) => {
            xunfold
                .column_mut(unfold_index)
                .assign(&x.column(unfold_index));
            unfold_index += 1;
        }
        Xtype::Enum(v) => {
            let mut unfold = Array::zeros((x.nrows(), v.len()));
            Zip::from(unfold.rows_mut())
                .and(x.rows())
                .for_each(|mut row, xrow| {
                    let index = xrow[[unfold_index]] as usize;
                    row[[index]] = 1.;
                });
            xunfold
                .slice_mut(s![.., unfold_index..unfold_index + v.len()])
                .assign(&unfold);
            unfold_index += v.len();
        }
    });
    xunfold
}

fn take_closest(v: &[i32], val: f64) -> i32 {
    let idx = Array::from_vec(v.to_vec())
        .map(|refval| (val - *refval as f64).abs())
        .argmin()
        .unwrap();
    v[idx]
}

/// Project continuously relaxed values to their closer assessable values.
/// Note: categorical (or enum) x dimensions are still expanded that is
/// there are still as many columns as categorical possible values for the given x dimension.
/// For instance, if an input dimension is typed ["blue", "red", "green"] in xlimits a sample/row of
/// the input x may contain the values (or mask) [..., 0, 0, 1, ...] to specify "green" for
/// this original dimension.
pub fn cast_to_discrete_values(xtypes: &[Xtype], x: &mut Array2<f64>) {
    let mut xcol = 0;
    xtypes.iter().for_each(|s| match s {
        Xtype::Cont(_, _) => xcol += 1,
        Xtype::Int(_, _) => {
            let xround = x.column(xcol).mapv(|v| v.round()).to_owned();
            x.column_mut(xcol).assign(&xround);
            xcol += 1;
        }
        Xtype::Ord(v) => {
            let xround = x
                .column(xcol)
                .mapv(|val| take_closest(v, val) as f64)
                .to_owned();
            x.column_mut(xcol).assign(&xround);
            xcol += 1;
        }
        Xtype::Enum(v) => {
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

pub fn get_cast_to_discrete_values(xtypes: &[Xtype], x: &Array2<f64>) -> Array2<f64> {
    let mut xcast = x.to_owned();
    cast_to_discrete_values(xtypes, &mut xcast);
    xcast
}

pub fn cast_to_enum_value(xtypes: &[Xtype], i: usize, enum_i: usize) -> Option<String> {
    if let Xtype::Enum(v) = xtypes[i].clone() {
        return Some(v[enum_i].clone());
    }
    None
}

// TODO
// pub fn cast_to_mixint(xtypes: &[Xtype], x: &Vec<Vec<Xval>>) -> Array2<f64> {
//     let mut res = Array::zeros((xtypes.len(), x[1].len()));
//     res.outer_iter().for_each(|mut row| {
//         Zip::from(row)
//             .and(xtypes)
//             .for_each(|val, &xtype| match xtype {
//                 Cont(_, _) => (),
//                 Int(_, _) => *val = v,
//             });
//     });
//     res
// }

pub struct MixintSampling {
    lhs: Lhs<f64, Isaac64Rng>,
    xtypes: Vec<Xtype>,
    /// whether data are in given in folded space (enum indexes) or not (enum masks)
    output_in_folded_space: bool,
}

impl MixintSampling {
    pub fn new(xtypes: Vec<Xtype>) -> Self {
        MixintSampling {
            lhs: Lhs::new(&unfold_xlimits_with_continuous_limits(&xtypes)),
            xtypes: xtypes.clone(),
            output_in_folded_space: false,
        }
    }

    pub fn work_in_folded_space(&mut self, output_in_folded_space: bool) -> &mut Self {
        self.output_in_folded_space = output_in_folded_space;
        self
    }
}

impl SamplingMethod<f64> for MixintSampling {
    fn sampling_space(&self) -> &Array2<f64> {
        self.lhs.sampling_space()
    }

    fn normalized_sample(&self, ns: usize) -> Array2<f64> {
        self.lhs.normalized_sample(ns)
    }

    fn sample(&self, ns: usize) -> Array2<f64> {
        let mut doe = self.lhs.sample(ns);
        cast_to_discrete_values(&self.xtypes, &mut doe);
        if self.output_in_folded_space {
            fold_with_enum_index(&self.xtypes, &doe.view())
        } else {
            doe
        }
    }
}

pub type SurrogateParams = MoeParams<f64, Isaac64Rng>;

pub struct MixintMoeParams {
    moe_params: SurrogateParams,
    xtypes: Vec<Xtype>,
    /// whether x data are in given in folded space (enum indexes) or not (enum masks)
    work_in_folded_space: bool,
}

impl MixintMoeParams {
    pub fn new(xtypes: &[Xtype], moe_params: &SurrogateParams) -> Self {
        MixintMoeParams {
            moe_params: moe_params.clone(),
            xtypes: xtypes.to_vec(),
            work_in_folded_space: false,
        }
    }

    pub fn work_in_folded_space(&mut self, wfs: bool) -> &mut Self {
        self.work_in_folded_space = wfs;
        self
    }

    pub fn xtypes(&self) -> &[Xtype] {
        &self.xtypes
    }
}

impl MoeFit for MixintMoeParams {
    fn train(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<Box<dyn Surrogate>> {
        Ok(Box::new(self.fit(x, y)) as Box<dyn Surrogate>)
    }
}

impl MixintMoeParams {
    fn fit(&self, x: &Array2<f64>, y: &Array2<f64>) -> MixintMoe {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, &x.view())
        } else {
            x.to_owned()
        };
        cast_to_discrete_values(&self.xtypes, &mut xcast);
        MixintMoe {
            moe: self
                .moe_params
                .clone()
                .regression_spec(RegressionSpec::CONSTANT)
                .fit(&Dataset::new(xcast, y.to_owned()))
                .unwrap(),
            xtypes: self.xtypes.clone(),
            work_in_folded_space: self.work_in_folded_space,
        }
    }
}

#[cfg_attr(feature = "persistent", derive(Serialize, Deserialize))]
pub struct MixintMoe {
    moe: Moe,
    xtypes: Vec<Xtype>,
    work_in_folded_space: bool,
}

impl std::fmt::Display for MixintMoe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.moe)
    }
}

#[cfg_attr(feature = "persistent", typetag::serde)]
impl Surrogate for MixintMoe {
    fn predict_values(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, x)
        } else {
            x.to_owned()
        };
        cast_to_discrete_values(&self.xtypes, &mut xcast);
        self.moe.predict_values(&xcast)
    }

    fn predict_variances(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, x)
        } else {
            x.to_owned()
        };
        cast_to_discrete_values(&self.xtypes, &mut xcast);
        self.moe.predict_variances(&xcast)
    }

    /// Save Moe model in given file.
    #[cfg(feature = "persistent")]
    fn save(&self, path: &str) -> Result<()> {
        let mut file = fs::File::create(path).unwrap();
        let bytes = match serde_json::to_string(self) {
            Ok(b) => b,
            Err(err) => return Err(MoeError::SaveError(err)),
        };
        file.write_all(bytes.as_bytes())?;
        Ok(())
    }
}

pub struct MixintContext {
    xtypes: Vec<Xtype>,
    work_in_folded_space: bool,
}

impl MixintContext {
    pub fn new(xtypes: &[Xtype]) -> Self {
        MixintContext {
            xtypes: xtypes.to_vec(),
            work_in_folded_space: true,
        }
    }

    pub fn get_unfolded_dim(&self) -> usize {
        compute_unfolded_dimension(&self.xtypes)
    }

    pub fn create_sampling(&self, seed: Option<u64>) -> MixintSampling {
        let lhs = seed.map_or(
            Lhs::new(&unfold_xlimits_with_continuous_limits(&self.xtypes)),
            |seed| {
                let rng = Isaac64Rng::seed_from_u64(seed);
                Lhs::new(&unfold_xlimits_with_continuous_limits(&self.xtypes)).with_rng(rng)
            },
        );
        MixintSampling {
            lhs,
            xtypes: self.xtypes.clone(),
            output_in_folded_space: self.work_in_folded_space,
        }
    }

    pub fn create_surrogate(
        &self,
        moe_params: &SurrogateParams,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> MixintMoe {
        let mut params = MixintMoeParams::new(&self.xtypes, moe_params);
        let params = params.work_in_folded_space(self.work_in_folded_space);
        params.fit(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use egobox_moe::CorrelationSpec;
    use ndarray::array;

    #[test]
    fn test_mixint_lhs() {
        let xtypes = vec![
            Xtype::Cont(-10.0, 10.0),
            Xtype::Enum(vec![
                "blue".to_string(),
                "red".to_string(),
                "green".to_string(),
            ]),
            Xtype::Int(-10, 10),
            Xtype::Ord(vec![1, 3, 5, 8]),
        ];

        let mixi = MixintContext::new(&xtypes);
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

    #[test]
    fn test_mixint_moe_1d() {
        let xtypes = vec![Xtype::Int(0, 4)];

        let mixi = MixintContext::new(&xtypes);

        let moe_params = SurrogateParams::new(1);
        let xt = array![[0.], [2.], [3.0], [4.]];
        let yt = array![[0.], [1.5], [0.9], [1.]];
        let mixi_moe = mixi.create_surrogate(&moe_params, &xt, &yt);

        let num = 5;
        let xtest = Array::linspace(0.0, 4.0, num).insert_axis(Axis(1));
        let ytest = mixi_moe
            .predict_values(&xtest.view())
            .expect("Predict val fail");
        let yvar = mixi_moe
            .predict_variances(&xtest.view())
            .expect("Predict var fail");
        println!("{:?}", ytest);
        assert_abs_diff_eq!(
            array![[0.], [0.8296067096163109], [1.5], [0.9], [1.]],
            ytest,
            epsilon = 1e-6
        );
        println!("{:?}", yvar);
        assert_abs_diff_eq!(
            array![[0.], [0.35290670137172425], [0.], [0.], [0.]],
            yvar,
            epsilon = 1e-6
        );
    }

    fn ftest(x: &Array2<f64>) -> Array2<f64> {
        let mut y = (x.column(0).to_owned() * x.column(0)).insert_axis(Axis(1));
        y = &y + (x.column(1).to_owned() * x.column(1)).insert_axis(Axis(1));
        y = &y * (x.column(2).insert_axis(Axis(1)).mapv(|v| v + 1.));
        y
    }

    #[test]
    fn test_mixint_moe_3d() {
        let xtypes = vec![
            Xtype::Int(0, 5),
            Xtype::Cont(0., 4.),
            Xtype::Enum(vec![
                "blue".to_string(),
                "red".to_string(),
                "green".to_string(),
                "yellow".to_string(),
            ]),
        ];

        let mixi = MixintContext::new(&xtypes);
        let mixi_lhs = mixi.create_sampling(Some(0));

        let n = mixi.get_unfolded_dim() * 5;
        let xt = mixi_lhs.sample(n);
        let yt = ftest(&xt);

        let moe_params =
            SurrogateParams::new(1).correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL);
        let mixi_moe = mixi.create_surrogate(&moe_params, &xt, &yt);

        let ntest = 10;
        let mixi_lhs = mixi.create_sampling(Some(42));

        let xtest = mixi_lhs.sample(ntest);
        let ytest = mixi_moe
            .predict_values(&xtest.view())
            .expect("Predict val fail");
        let ytrue = ftest(&xtest);
        assert_abs_diff_eq!(ytrue, ytest, epsilon = 1.5);
    }
}
