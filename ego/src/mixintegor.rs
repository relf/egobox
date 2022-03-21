use crate::egor::*;
use crate::errors::*;
use crate::mixint::*;
use crate::types::*;
use moe::{Moe, MoeFit};
use ndarray::Array;
use rand_isaac::Isaac64Rng;

struct MixintEgor<'a, O: GroupFunc> {
    pub egor: Egor<'a, O, Isaac64Rng>,
    xtypes: Vec<Xtype>,
}

impl<'a, O: GroupFunc> MixintEgor<'a, O> {
    pub fn new(f: O, moe_params: Option<&'a dyn MoeFit>, xtypes: &[Xtype]) -> Self {
        let xlimits = unfold_xlimits_with_continuous_limits(xtypes);
        let mut egor = Egor::new(f, &xlimits);
        let egor = egor.moe_params(moe_params);
        MixintEgor {
            egor: egor.clone(),
            xtypes: xtypes.to_vec(),
        }
    }

    pub fn minimize(&mut self) -> Result<OptimResult<f64>> {
        let moe_params = Moe::params(self.egor.n_clusters.unwrap_or(1))
            .set_kpls_dim(self.egor.kpls_dim)
            .set_regression_spec(self.egor.regression_spec)
            .set_correlation_spec(self.egor.correlation_spec);

        Ok(OptimResult {
            x_opt: Array::zeros(0),
            y_opt: Array::zeros(0),
        })
    }
}

#[cfg(test)]
mod tests {}
