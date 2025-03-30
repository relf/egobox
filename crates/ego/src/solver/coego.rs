use crate::types::*;
use crate::EgorSolver;

use serde::de::DeserializeOwned;

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + DeserializeOwned,
    C: CstrFn,
{
    /// Set active components to xcoop using xopt values
    /// active and values must have the same size
    pub(crate) fn setx(xcoop: &mut [f64], active: &[usize], values: &[f64]) {
        std::iter::zip(active, values).for_each(|(&i, &xi)| xcoop[i] = xi)
    }
}
