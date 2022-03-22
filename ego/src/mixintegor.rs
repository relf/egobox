use crate::egor::*;
use crate::errors::*;
use crate::mixint::*;
use crate::types::*;
use ndarray::{Array2, Axis};
use rand_isaac::Isaac64Rng;

pub struct MixintEgor<'a, O: GroupFunc> {
    xtypes: Vec<Xtype>,
    egor: Egor<'a, O, Isaac64Rng>,
}

impl<'a, O: GroupFunc> MixintEgor<'a, O> {
    pub fn new(mix_params: &'a MixintMoeParams, evaluator: &'a MixintEvaluator, f: O) -> Self {
        let xlimits = unfold_xlimits_with_continuous_limits(mix_params.xtypes());
        let egor = Egor::new(f, &xlimits)
            .moe_params(Some(mix_params))
            .evaluator(Some(evaluator))
            .clone();
        MixintEgor {
            xtypes: mix_params.xtypes().to_vec(),
            egor,
        }
    }

    pub fn minimize(&self) -> Result<OptimResult<f64>> {
        let res = self.egor.minimize();
        res.map(|opt| -> OptimResult<f64> {
            let x_opt = opt.x_opt.to_owned().insert_axis(Axis(0));
            let x_opt = get_cast_to_discrete_values(&self.xtypes, &x_opt);
            let x_opt = fold_with_enum_index(&self.xtypes, &x_opt);
            OptimResult {
                x_opt: x_opt.row(0).to_owned(),
                y_opt: opt.y_opt.to_owned(),
            }
        })
    }
}

pub struct MixintEvaluator {
    xtypes: Vec<Xtype>,
}

impl Evaluator for MixintEvaluator {
    fn eval(&self, x: &Array2<f64>) -> Array2<f64> {
        let fold = fold_with_enum_index(&self.xtypes, x);
        get_cast_to_discrete_values(&self.xtypes, &fold)
    }
}

impl MixintEvaluator {
    pub fn new(xtypes: &[Xtype]) -> Self {
        MixintEvaluator {
            xtypes: xtypes.to_vec(),
        }
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
    fn test_mixintegor_ei() {
        let n_eval = 10;
        let doe = array![[0.], [7.], [25.]];
        let xtypes = vec![Xtype::Int(0, 25)];

        let moe_params = MoeParams::default();
        let moe_params = MixintMoeParams::new(&xtypes, &moe_params);
        let evaluator = MixintEvaluator::new(&xtypes);
        let mut mixintegor = MixintEgor::new(&moe_params, &evaluator, mixsinx);
        mixintegor
            .egor
            .doe(Some(doe))
            .n_eval(n_eval)
            .expect(Some(ApproxValue {
                value: -15.1,
                tolerance: 1e-1,
            }))
            .infill_strategy(InfillStrategy::EI);

        let res = mixintegor.minimize();
        println!("{:?}", res)
    }

    #[test]
    fn test_mixintegor_wb2() {
        let n_eval = 10;
        let xtypes = vec![Xtype::Int(0, 25)];

        let moe_params = MoeParams::default();
        let moe_params = MixintMoeParams::new(&xtypes, &moe_params);
        let evaluator = MixintEvaluator::new(&xtypes);
        let mut mixintegor = MixintEgor::new(&moe_params, &evaluator, mixsinx);
        mixintegor
            .egor
            .n_eval(n_eval)
            .expect(Some(ApproxValue {
                value: -15.1,
                tolerance: 1e-1,
            }))
            .infill_strategy(InfillStrategy::WB2);

        let res = mixintegor.minimize();
        println!("{:?}", res)
    }
}
