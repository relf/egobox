use crate::egor::*;
use crate::errors::*;
use crate::mixint::*;
use crate::types::*;
use moe::{Moe, MoeParams};
use ndarray::Array;
use rand_isaac::Isaac64Rng;

struct MixintEgor<'a, O: GroupFunc> {
    xtypes: Vec<Xtype>,
    pub egor: Egor<'a, O, Isaac64Rng>,
}

impl<'a, O: GroupFunc> MixintEgor<'a, O> {
    pub fn new(moe_params: &'a MixintMoeParams, f: O) -> Self {
        let xlimits = unfold_xlimits_with_continuous_limits(moe_params.xtypes());
        let mut egor = Egor::new(f, &xlimits);
        let egor = egor.moe_params(Some(moe_params));
        MixintEgor {
            xtypes: moe_params.xtypes().to_vec(),
            egor: egor.clone(),
        }
    }

    pub fn minimize(&mut self) -> Result<OptimResult<f64>> {
        let res = self.egor.minimize();
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixint::Xtype;
    use moe::MoeParams;
    use ndarray::{array, Array2, ArrayView2};
    use ndarray_linalg::Norm;

    fn mixsinx(x: &ArrayView2<f64>) -> Array2<f64> {
        if (x.mapv(|v| v.round()).norm_l2() - x.norm_l2()).abs() < 1e-6 {
            (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
        } else {
            panic!("Error: mixsinx works only on integer, got {:?}", x)
        }
    }

    #[test]
    fn test_mixintegor() {
        let n_eval = 10;
        let doe = array![[0.], [10.], [20.]];
        let xtypes = vec![Xtype::Int(0, 25)];

        let moe_params = MoeParams::default();
        let moe_params = MixintMoeParams::new(&xtypes, &moe_params);
        let mut mixintegor = MixintEgor::new(&moe_params, mixsinx);
        mixintegor.egor.doe(Some(doe)).n_eval(n_eval);

        let res = mixintegor.minimize();
        println!("{:?}", res)
    }
}
