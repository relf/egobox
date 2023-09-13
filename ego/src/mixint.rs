//! This library implements continuous relaxation functions,
//! it is a port of [SMT mixed integer module](https://smt.readthedocs.io/en/latest/_src_docs/applications/mixed_integer.html)

#![allow(dead_code)]
use crate::errors::{EgoError, Result};
use crate::types::{SurrogateBuilder, XType};
use egobox_doe::{Lhs, SamplingMethod};
use egobox_moe::{
    Clustered, ClusteredSurrogate, Clustering, CorrelationSpec, Moe, MoeParams, RegressionSpec,
    Surrogate,
};
use linfa::traits::{Fit, PredictInplace};
use linfa::{DatasetBase, ParamGuard};
use ndarray::{s, Array, Array2, ArrayBase, ArrayView2, Axis, Data, DataMut, Ix2, Zip};
use ndarray_rand::rand::SeedableRng;
use ndarray_stats::QuantileExt;
use rand_xoshiro::Xoshiro256Plus;

use serde::{Deserialize, Serialize};

#[cfg(feature = "persistent")]
use egobox_moe::MoeError;
#[cfg(feature = "persistent")]
use std::fs;
#[cfg(feature = "persistent")]
use std::io::Write;

/// Expand xlimits to add continuous dimensions for enumeration x features.
///
/// Each level of an enumerate gives a new continuous dimension in [0, 1].
/// Each integer dimensions are relaxed continuously.
pub fn unfold_xtypes_as_continuous_limits(xtypes: &[XType]) -> Array2<f64> {
    let mut xlimits: Vec<f64> = vec![];
    let mut dim = 0;
    xtypes.iter().for_each(|xtype| match xtype {
        XType::Cont(lower, upper) => {
            dim += 1;
            xlimits.extend([*lower, *upper]);
        }
        XType::Int(lower, upper) => {
            dim += 1;
            xlimits.extend([*lower as f64, *upper as f64]);
        }
        XType::Ord(v) => {
            dim += 1;
            xlimits.extend([
                (v.iter().fold(f64::INFINITY, |a, &b| a.min(b))),
                (v.iter().fold(-f64::INFINITY, |a, &b| a.max(b))),
            ]);
        }
        XType::Enum(v) => {
            dim += v.len();
            v.iter().for_each(|_| {
                xlimits.extend([0., 1.]);
            })
        }
    });
    Array::from_shape_vec((dim, 2), xlimits).unwrap()
}

/// Reduce categorical inputs from discrete unfolded space to
/// initial x dimension space where categorical x dimensions are valued by the index
/// in the corresponding enumerate list.
///
/// For instance, if an input dimension is typed ["blue", "red", "green"] a sample/row of
/// the input x may contain the mask [..., 0, 0, 1, ...] which will be contracted in [..., 2, ...]
/// meaning the "green" value.
/// This function is the opposite of unfold_with_enum_mask().
pub fn fold_with_enum_index(
    xtypes: &[XType],
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    let mut xfold = Array::zeros((x.nrows(), xtypes.len()));
    let mut unfold_index = 0;
    Zip::indexed(xfold.columns_mut()).for_each(|j, mut col| match &xtypes[j] {
        XType::Cont(_, _) | XType::Int(_, _) | XType::Ord(_) => {
            col.assign(&x.column(unfold_index));
            unfold_index += 1;
        }
        XType::Enum(v) => {
            let xenum = x.slice(s![.., j..j + v.len()]);
            let argmaxx = xenum.map_axis(Axis(1), |row| row.argmax().unwrap() as f64);
            col.assign(&argmaxx);
            unfold_index += v.len();
        }
    });
    xfold
}

/// Compute dimension when all variables are continuously relaxed
fn compute_unfolded_dimension(xtypes: &[XType]) -> usize {
    xtypes
        .iter()
        .map(|s| match s {
            XType::Enum(v) => v.len(),
            _ => 1,
        })
        .reduce(|acc, l| -> usize { acc + l })
        .unwrap()
}

/// Expand categorical inputs from initial x dimension space where categorical x dimensions
/// are valued by the index in the corresponding enumerate list to the discrete unfolded space.
///
/// For instance, if an input dimension is typed ["blue", "red", "green"] a sample/row of
/// the input x may contain [..., 2, ...] which will be expanded in [..., 0, 0, 1, ...].
/// This function is the opposite of fold_with_enum_index().
fn unfold_with_enum_mask(
    xtypes: &[XType],
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    let mut xunfold = Array::zeros((x.nrows(), compute_unfolded_dimension(xtypes)));
    let mut unfold_index = 0;
    xtypes.iter().for_each(|s| match s {
        XType::Cont(_, _) | XType::Int(_, _) | XType::Ord(_) => {
            xunfold
                .column_mut(unfold_index)
                .assign(&x.column(unfold_index));
            unfold_index += 1;
        }
        XType::Enum(v) => {
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

/// Find closest value to `val` in given slice `v`.
fn take_closest(v: &[f64], val: f64) -> f64 {
    let idx = Array::from_vec(v.to_vec())
        .map(|refval| (val - *refval).abs())
        .argmin()
        .unwrap();
    v[idx]
}

/// Project continuously relaxed values to their closer assessable values.
///
/// See cast_to_discrete_values
fn cast_to_discrete_values_mut(xtypes: &[XType], x: &mut ArrayBase<impl DataMut<Elem = f64>, Ix2>) {
    let mut xcol = 0;
    xtypes.iter().for_each(|s| match s {
        XType::Cont(_, _) => xcol += 1,
        XType::Int(_, _) => {
            let xround = x.column(xcol).mapv(|v| v.round()).to_owned();
            x.column_mut(xcol).assign(&xround);
            xcol += 1;
        }
        XType::Ord(v) => {
            let xround = x.column(xcol).mapv(|val| take_closest(v, val)).to_owned();
            x.column_mut(xcol).assign(&xround);
            xcol += 1;
        }
        XType::Enum(v) => {
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

/// Project continuously relaxed values to their closer assessable values.
///
/// Note: categorical (or enum) x dimensions are still expanded that is
/// there are still as many columns as categorical possible values for the given x dimension.
/// For instance, if an input dimension is typed ["blue", "red", "green"] in xlimits a sample/row of
/// the input x may contain the values (or mask) [..., 0, 0, 1, ...] to specify "green" for
/// this original dimension.
pub fn cast_to_discrete_values(
    xtypes: &[XType],
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    let mut xcast = x.to_owned();
    cast_to_discrete_values_mut(xtypes, &mut xcast);
    xcast
}

/// A decorator of LHS sampling that takes into account XType specifications
/// casting continuous LHS result from floats to discrete types.
#[derive(Serialize, Deserialize)]
pub struct MixintSampling {
    /// The continuous LHS sampling method
    lhs: Lhs<f64, Xoshiro256Plus>,
    /// The input specifications
    xtypes: Vec<XType>,
    /// whether data are in given in folded space (enum indexes) or not (enum masks)
    /// i.e for "blue" in ["red", "green", "blue"] either \[2\] or [0, 0, 1]
    output_in_folded_space: bool,
}

impl MixintSampling {
    /// Constructor using `xtypes` specifications
    pub fn new(xtypes: Vec<XType>) -> Self {
        MixintSampling {
            lhs: Lhs::new(&unfold_xtypes_as_continuous_limits(&xtypes)),
            xtypes: xtypes.clone(),
            output_in_folded_space: false,
        }
    }

    /// Sets whether we want to work in folded space
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
        cast_to_discrete_values_mut(&self.xtypes, &mut doe);
        if self.output_in_folded_space {
            fold_with_enum_index(&self.xtypes, &doe.view())
        } else {
            doe
        }
    }
}

/// Moe type builder for mixed-integer Egor optimizer
pub type MoeBuilder = MoeParams<f64, Xoshiro256Plus>;
/// A decorator of Moe surrogate builder that takes into account XType specifications
///
/// It allows to implement continuous relaxation over continuous Moe builder.
#[derive(Clone, Serialize, Deserialize)]
pub struct MixintMoeValidParams {
    /// The surrogate factory
    surrogate_builder: MoeParams<f64, Xoshiro256Plus>,
    /// The input specifications
    xtypes: Vec<XType>,
    /// whether data are in given in folded space (enum indexes) or not (enum masks)
    /// i.e for "blue" in ["red", "green", "blue"] either \[2\] or [0, 0, 1]
    work_in_folded_space: bool,
}

impl MixintMoeValidParams {
    /// Sets whether we want to work in folded space
    pub fn work_in_folded_space(&self) -> bool {
        self.work_in_folded_space
    }

    /// Sets the specification
    pub fn xtypes(&self) -> &[XType] {
        &self.xtypes
    }
}

/// Parameters for mixture of experts surrogate model
#[derive(Clone, Serialize, Deserialize)]
pub struct MixintMoeParams(MixintMoeValidParams);

impl MixintMoeParams {
    /// Constructor given  `xtypes` specifications and given surrogate builder
    pub fn new(xtypes: &[XType], surrogate_builder: &MoeBuilder) -> Self {
        MixintMoeParams(MixintMoeValidParams {
            surrogate_builder: surrogate_builder.clone(),
            xtypes: xtypes.to_vec(),
            work_in_folded_space: false,
        })
    }

    /// Sets whether we want to work in folded space
    pub fn work_in_folded_space(&mut self, wfs: bool) -> &mut Self {
        self.0.work_in_folded_space = wfs;
        self
    }

    /// Sets the specification
    pub fn xtypes(&self) -> &[XType] {
        &self.0.xtypes
    }
}

impl MixintMoeValidParams {
    fn _train(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<MixintMoe> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, &xt.view())
        } else {
            xt.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        let mixmoe = MixintMoe {
            moe: self
                .surrogate_builder
                .clone()
                .check()?
                .train(&xcast, &yt.to_owned())
                .unwrap(),
            xtypes: self.xtypes.clone(),
            work_in_folded_space: self.work_in_folded_space,
        };
        Ok(mixmoe)
    }

    fn _train_on_clusters(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        clustering: &egobox_moe::Clustering,
    ) -> Result<MixintMoe> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, &xt.view())
        } else {
            xt.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        let mixmoe = MixintMoe {
            moe: self
                .surrogate_builder
                .clone()
                .check_ref()?
                .train_on_clusters(&xcast, &yt.to_owned(), clustering)
                .unwrap(),
            xtypes: self.xtypes.clone(),
            work_in_folded_space: self.work_in_folded_space,
        };
        Ok(mixmoe)
    }
}

impl SurrogateBuilder for MixintMoeParams {
    fn new_with_xtypes_rng(xtypes: &[XType]) -> Self {
        MixintMoeParams::new(xtypes, &MoeParams::new())
    }

    /// Sets the allowed regression models used in gaussian processes.
    fn set_regression_spec(&mut self, regression_spec: RegressionSpec) {
        self.0 = MixintMoeValidParams {
            surrogate_builder: self
                .0
                .surrogate_builder
                .clone()
                .regression_spec(regression_spec),
            xtypes: self.0.xtypes.clone(),
            work_in_folded_space: self.0.work_in_folded_space,
        }
    }

    /// Sets the allowed correlation models used in gaussian processes.
    fn set_correlation_spec(&mut self, correlation_spec: CorrelationSpec) {
        self.0 = MixintMoeValidParams {
            surrogate_builder: self
                .0
                .surrogate_builder
                .clone()
                .correlation_spec(correlation_spec),
            xtypes: self.0.xtypes.clone(),
            work_in_folded_space: self.0.work_in_folded_space,
        }
    }

    /// Sets the number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
    fn set_kpls_dim(&mut self, kpls_dim: Option<usize>) {
        self.0 = MixintMoeValidParams {
            surrogate_builder: self.0.surrogate_builder.clone().kpls_dim(kpls_dim),
            xtypes: self.0.xtypes.clone(),
            work_in_folded_space: self.0.work_in_folded_space,
        }
    }

    /// Sets the number of clusters used by the mixture of surrogate experts.
    fn set_n_clusters(&mut self, n_clusters: usize) {
        self.0 = MixintMoeValidParams {
            surrogate_builder: self.0.surrogate_builder.clone().n_clusters(n_clusters),
            xtypes: self.0.xtypes.clone(),
            work_in_folded_space: self.0.work_in_folded_space,
        }
    }

    fn train(
        &self,
        xt: &ArrayView2<f64>,
        yt: &ArrayView2<f64>,
    ) -> Result<Box<dyn ClusteredSurrogate>> {
        let mixmoe = self.check_ref()?._train(xt, yt)?;
        Ok(mixmoe).map(|mixmoe| Box::new(mixmoe) as Box<dyn ClusteredSurrogate>)
    }

    fn train_on_clusters(
        &self,
        xt: &ArrayView2<f64>,
        yt: &ArrayView2<f64>,
        clustering: &Clustering,
    ) -> Result<Box<dyn ClusteredSurrogate>> {
        let mixmoe = self.check_ref()?._train_on_clusters(xt, yt, clustering)?;
        Ok(mixmoe).map(|mixmoe| Box::new(mixmoe) as Box<dyn ClusteredSurrogate>)
    }
}

impl<D: Data<Elem = f64>> Fit<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>, EgoError>
    for MixintMoeValidParams
{
    type Object = MixintMoe;

    fn fit(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
    ) -> Result<Self::Object> {
        let x = dataset.records();
        let y = dataset.targets();
        self._train(x, y)
    }
}

impl ParamGuard for MixintMoeParams {
    type Checked = MixintMoeValidParams;
    type Error = EgoError;

    fn check_ref(&self) -> Result<&Self::Checked> {
        Ok(&self.0)
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}

impl From<MixintMoeValidParams> for MixintMoeParams {
    fn from(item: MixintMoeValidParams) -> Self {
        MixintMoeParams(item)
    }
}

/// The Moe model that takes into account XType specifications
#[derive(Serialize, Deserialize)]
pub struct MixintMoe {
    /// the decorated Moe
    moe: Moe,
    /// The input specifications
    xtypes: Vec<XType>,
    /// whether data are in given in folded space (enum indexes) or not (enum masks)
    /// i.e for "blue" in ["red", "green", "blue"] either \[2\] or [0, 0, 1]
    work_in_folded_space: bool,
}

impl std::fmt::Display for MixintMoe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = if crate::utils::no_discrete(&self.xtypes) {
            ""
        } else {
            "MixInt"
        };
        write!(f, "{}{}", prefix, &self.moe)
    }
}

impl Clustered for MixintMoe {
    fn n_clusters(&self) -> usize {
        self.moe.n_clusters()
    }

    fn recombination(&self) -> egobox_moe::Recombination<f64> {
        self.moe.recombination()
    }

    /// Convert to clustering
    fn to_clustering(&self) -> Clustering {
        Clustering::new(self.moe.gmx().clone(), self.moe.recombination())
    }
}

#[typetag::serde]
impl Surrogate for MixintMoe {
    fn predict_values(&self, x: &ArrayView2<f64>) -> egobox_moe::Result<Array2<f64>> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, x)
        } else {
            x.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        self.moe.predict_values(&xcast)
    }

    fn predict_variances(&self, x: &ArrayView2<f64>) -> egobox_moe::Result<Array2<f64>> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, x)
        } else {
            x.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        self.moe.predict_variances(&xcast)
    }

    fn predict_derivatives(&self, x: &ArrayView2<f64>) -> egobox_moe::Result<Array2<f64>> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, x)
        } else {
            x.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        self.moe.predict_derivatives(&xcast)
    }

    fn predict_variance_derivatives(&self, x: &ArrayView2<f64>) -> egobox_moe::Result<Array2<f64>> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, x)
        } else {
            x.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        self.moe.predict_variance_derivatives(&xcast)
    }

    fn sample(&self, x: &ArrayView2<f64>, n_traj: usize) -> egobox_moe::Result<Array2<f64>> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, x)
        } else {
            x.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        self.moe.sample(&xcast.view(), n_traj)
    }

    /// Save Moe model in given file.
    #[cfg(feature = "persistent")]
    fn save(&self, path: &str) -> egobox_moe::Result<()> {
        let mut file = fs::File::create(path).unwrap();
        let bytes = match serde_json::to_string(self) {
            Ok(b) => b,
            Err(err) => return Err(MoeError::SaveError(err)),
        };
        file.write_all(bytes.as_bytes())?;
        Ok(())
    }
}

impl ClusteredSurrogate for MixintMoe {}

impl<D: Data<Elem = f64>> PredictInplace<ArrayBase<D, Ix2>, Array2<f64>> for MixintMoe {
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array2<f64>) {
        assert_eq!(
            x.nrows(),
            y.nrows(),
            "The number of data points must match the number of output targets."
        );

        let values = self.moe.predict_values(x).expect("MixintMoE prediction");
        *y = values;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array2<f64> {
        Array2::zeros((x.nrows(), self.moe.output_dim()))
    }
}

struct MoeVariancePredictor<'a>(&'a Moe);
impl<'a, D: Data<Elem = f64>> PredictInplace<ArrayBase<D, Ix2>, Array2<f64>>
    for MoeVariancePredictor<'a>
{
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array2<f64>) {
        assert_eq!(
            x.nrows(),
            y.nrows(),
            "The number of data points must match the number of output targets."
        );

        let values = self
            .0
            .predict_variances(x)
            .expect("MixintMoE variances prediction");
        *y = values;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array2<f64> {
        Array2::zeros((x.nrows(), self.0.output_dim()))
    }
}

/// A factory to build consistent sampling method and surrogate regarding
/// XType specifications
pub struct MixintContext {
    /// The input specifications
    xtypes: Vec<XType>,
    /// whether data are in given in folded space (enum indexes) or not (enum masks)
    /// i.e for "blue" in ["red", "green", "blue"] either \[2\] or [0, 0, 1]
    work_in_folded_space: bool,
}

impl MixintContext {
    /// Constructor with given `xtypes` specification
    pub fn new(xtypes: &[XType]) -> Self {
        MixintContext {
            xtypes: xtypes.to_vec(),
            work_in_folded_space: true,
        }
    }

    /// Compute input dim once unfolded due to continupous relaxation
    pub fn get_unfolded_dim(&self) -> usize {
        compute_unfolded_dimension(&self.xtypes)
    }

    /// Create a mixed integer LHS
    pub fn create_sampling(&self, seed: Option<u64>) -> MixintSampling {
        let lhs = seed.map_or(
            Lhs::new(&unfold_xtypes_as_continuous_limits(&self.xtypes)),
            |seed| {
                let rng = Xoshiro256Plus::seed_from_u64(seed);
                Lhs::new(&unfold_xtypes_as_continuous_limits(&self.xtypes)).with_rng(rng)
            },
        );
        MixintSampling {
            lhs,
            xtypes: self.xtypes.clone(),
            output_in_folded_space: self.work_in_folded_space,
        }
    }

    /// Create a mixed integer mixture of experts surrogate
    pub fn create_surrogate(
        &self,
        surrogate_builder: &MoeBuilder,
        dataset: &DatasetBase<Array2<f64>, Array2<f64>>,
    ) -> Result<MixintMoe> {
        let mut params = MixintMoeParams::new(&self.xtypes, surrogate_builder);
        let params = params.work_in_folded_space(self.work_in_folded_space);
        params.fit(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use egobox_moe::CorrelationSpec;
    use linfa::Dataset;
    use ndarray::array;

    #[test]
    fn test_mixint_lhs() {
        let xtypes = vec![
            XType::Cont(-10.0, 10.0),
            XType::Enum(vec![
                "blue".to_string(),
                "red".to_string(),
                "green".to_string(),
            ]),
            XType::Int(-10, 10),
            XType::Ord(vec![1., 3., 5., 8.]),
        ];

        let mixi = MixintContext::new(&xtypes);
        let mixi_lhs = mixi.create_sampling(Some(0));

        let actual = mixi_lhs.sample(10);
        let expected = array![
            [-1.4333973977708876, 2.0, 10.0, 5.0],
            [-3.012628232178356, 2.0, 1.0, 8.0],
            [5.021749742902275, 2.0, -7.0, 5.0],
            [3.5274476356096063, 0.0, -2.0, 1.0],
            [7.6267468405479235, 0.0, 3.0, 1.0],
            [-4.8307414212425375, 1.0, -4.0, 3.0],
            [-6.60764078463793, 0.0, -3.0, 8.0],
            [-8.291614427265058, 0.0, 7.0, 3.0],
            [8.455590615130134, 1.0, 6.0, 5.0],
            [0.9688101123304538, 0.0, -9.0, 5.0]
        ];
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }

    #[test]
    fn test_mixint_moe_1d() {
        let xtypes = vec![XType::Int(0, 4)];

        let mixi = MixintContext::new(&xtypes);

        let surrogate_builder = MoeBuilder::new();
        let xt = array![[0.], [2.], [3.0], [4.]];
        let yt = array![[0.], [1.5], [0.9], [1.]];
        let ds = Dataset::new(xt, yt);
        let mixi_moe = mixi
            .create_surrogate(&surrogate_builder, &ds)
            .expect("Mixint surrogate creation");

        let num = 5;
        let xtest = Array::linspace(0.0, 4.0, num).insert_axis(Axis(1));
        let ytest = mixi_moe
            .predict_values(&xtest.view())
            .expect("Predict val fail");
        let yvar = mixi_moe
            .predict_variances(&xtest.view())
            .expect("Predict var fail");
        println!("{ytest:?}");
        assert_abs_diff_eq!(
            array![[0.], [0.8297580023632912], [1.5], [0.9], [1.]],
            ytest,
            epsilon = 1e-3
        );
        println!("{yvar:?}");
        assert_abs_diff_eq!(
            array![[0.], [0.35302771336218247], [0.], [0.], [0.]],
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
            XType::Int(0, 5),
            XType::Cont(0., 4.),
            XType::Enum(vec![
                "blue".to_string(),
                "red".to_string(),
                "green".to_string(),
                "yellow".to_string(),
            ]),
        ];

        let mixi = MixintContext::new(&xtypes);
        let mixi_lhs = mixi.create_sampling(Some(0));

        let n = mixi.get_unfolded_dim() * 10;
        let xt = mixi_lhs.sample(n);
        let yt = ftest(&xt);

        let surrogate_builder =
            MoeBuilder::new().correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL);
        let ds = Dataset::new(xt, yt);
        let mixi_moe = mixi
            .create_surrogate(&surrogate_builder, &ds)
            .expect("Mixint surrogate creation");

        let ntest = 10;
        let mixi_lhs = mixi.create_sampling(Some(42));

        let xtest = mixi_lhs.sample(ntest);
        let ytest = mixi_moe
            .predict_values(&xtest.view())
            .expect("Predict val fail");
        let ytrue = ftest(&xtest);
        assert_abs_diff_eq!(ytrue, ytest, epsilon = 2.0);
    }
}
