use crate::gaussian_mixture::GaussianMixture;
use crate::{FullGpSurrogate, GpSurrogate};
use bitflags::bitflags;
#[allow(unused_imports)]
use egobox_gp::correlation_models::{
    AbsoluteExponentialCorr, Matern32Corr, Matern52Corr, SquaredExponentialCorr,
};
#[allow(unused_imports)]
use egobox_gp::mean_models::{ConstantMean, LinearMean, QuadraticMean};
use linfa::Float;
use std::fmt::Display;

#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

/// Enumeration of recombination modes handled by the mixture
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub enum Recombination<F: Float> {
    /// prediction is taken from the expert with highest responsability
    /// resulting in a model with discontinuities
    Hard,
    /// Prediction is a combination experts prediction wrt their responsabilities,
    /// an optional heaviside factor might be used control steepness of the change between
    /// experts regions.
    Smooth(Option<F>),
}

impl<F: Float> Display for Recombination<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let recomb = match self {
            Recombination::Hard => "Hard".to_string(),
            Recombination::Smooth(Some(f)) => format!("Smooth({f})"),
            Recombination::Smooth(None) => "Smooth".to_string(),
        };
        write!(f, "Mixture[{}]", &recomb)
    }
}

bitflags! {
    /// Flags to specify tested regression models during experts selection (see [`regression_spec()`](SgpParams::regression_spec)).
    ///
    /// Flags can be combine with bit-wise `or` operator to select two or more models.
    /// ```ignore
    /// let spec = RegressionSpec::CONSTANT | RegressionSpec::LINEAR;
    /// ```
    ///
    /// See [bitflags::bitflags]
    #[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone, Copy)]
    #[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
    pub struct RegressionSpec: u8 {
        /// Constant regression
        const CONSTANT = 0x01;
        /// Linear regression
        const LINEAR = 0x02;
        /// 2-degree polynomial regression
        const QUADRATIC = 0x04;
        /// All regression models available
        const ALL = RegressionSpec::CONSTANT.bits()
                    | RegressionSpec::LINEAR.bits()
                    | RegressionSpec::QUADRATIC.bits();
    }
}

bitflags! {
    /// Flags to specify tested correlation models during experts selection (see [`correlation_spec()`](SgpParams::correlation_spec)).
    ///
    /// Flags can be combine with bit-wise `or` operator to select two or more models.
    /// ```ignore
    /// let spec = CorrelationSpec::MATERN32 | CorrelationSpec::Matern52;
    /// ```
    ///
    /// See [bitflags::bitflags]
    #[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone, Copy)]
    #[cfg_attr(feature = "serializable", derive(Serialize, Deserialize), serde(transparent))]
    pub struct CorrelationSpec: u8 {
        /// Squared exponential correlation model
        const SQUAREDEXPONENTIAL = 0x01;
        /// Absolute exponential correlation model
        const ABSOLUTEEXPONENTIAL = 0x02;
        /// Matern 3/2 correlation model
        const MATERN32 = 0x04;
        /// Matern 5/2 correlation model
        const MATERN52 = 0x08;
        /// All correlation models available
        const ALL = CorrelationSpec::SQUAREDEXPONENTIAL.bits()
                    | CorrelationSpec::ABSOLUTEEXPONENTIAL.bits()
                    | CorrelationSpec::MATERN32.bits()
                    | CorrelationSpec::MATERN52.bits();
    }
}

/// A trait to represent clustered structure
pub trait Clustered {
    fn n_clusters(&self) -> usize;
    fn recombination(&self) -> Recombination<f64>;

    fn to_clustering(&self) -> Clustering;
}

/// A structure for clustering
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct Clustering {
    /// Recombination between the clusters
    pub(crate) recombination: Recombination<f64>,
    /// Clusters
    pub(crate) gmx: GaussianMixture<f64>,
}

impl Clustering {
    pub fn new(gmx: GaussianMixture<f64>, recombination: Recombination<f64>) -> Self {
        Clustering { gmx, recombination }
    }

    pub fn recombination(&self) -> Recombination<f64> {
        self.recombination
    }
    pub fn gmx(&self) -> &GaussianMixture<f64> {
        &self.gmx
    }
}

/// A trait for Gp surrogates using clustering
pub trait ClusteredGpSurrogate: Clustered + GpSurrogate {}

/// A trait for Gp surrogates with derivatives using clustering
pub trait ClusteredSurrogate: Clustered + FullGpSurrogate {}
