use crate::egor::*;
use crate::errors::*;
use crate::mixint::*;
use crate::types::*;
use ndarray::{Array2, Axis};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;

/// The MixintEgor structure wraps the [Egor] structure to implement
/// continuous relaxation allowing to manage function optimization which
/// takes discrete input variables.  
pub struct MixintEgor<'a, O: GroupFunc, R: Rng + Clone> {
    /// Specifications of the x input variables being either coninuous (float) or discrete (integer)
    xtypes: Vec<Xtype>,
    /// The EGO algorithm. the object is accessible to be parametirizable using Egor API (see [`Egor`])
    pub egor: Egor<'a, O, R>,
}

impl<'a, O: GroupFunc> MixintEgor<'a, O, Isaac64Rng> {
    /// Constructor of MixintEgor optimizer.
    ///
    /// the function `f` under optimization accepts mixed integer variables
    /// returning objective to minimize and constraints to be negative.
    /// Mixture of experts parameters `mix_params` are used to model objective
    /// and constraints.
    /// Mixed integer preprocessor manages continous input transformation (cast to discrete values)
    /// before calling `f`.
    pub fn new(
        f: O,
        mix_params: &'a MixintMoeParams,
        pre_proc: &'a MixintPreProcessor,
    ) -> MixintEgor<'a, O, Isaac64Rng> {
        Self::new_with_rng(f, mix_params, pre_proc, Isaac64Rng::from_entropy())
    }
}

impl<'a, O: GroupFunc, R: Rng + Clone> MixintEgor<'a, O, R> {
    /// Constructor enabling random generator specification
    /// See [MixintEgor::new]
    pub fn new_with_rng(
        f: O,
        mix_params: &'a MixintMoeParams,
        pre_proc: &'a MixintPreProcessor,
        rng: R,
    ) -> Self {
        let xlimits = unfold_xlimits_with_continuous_limits(mix_params.xtypes());
        let egor = Egor::new_with_rng(f, &xlimits, rng)
            .surrogate_builder(Some(mix_params))
            .pre_proc(Some(pre_proc))
            .clone();
        MixintEgor {
            xtypes: mix_params.xtypes().to_vec(),
            egor,
        }
    }

    /// Minimize with regard to discrete input variables at construction using [Xtype]
    pub fn minimize(&self) -> Result<OptimResult<f64>> {
        let res = self.egor.minimize();
        res.map(|opt| -> OptimResult<f64> {
            let x_opt = opt.x_opt.to_owned().insert_axis(Axis(0));
            let x_opt = cast_to_discrete_values(&self.xtypes, &x_opt);
            let x_opt = fold_with_enum_index(&self.xtypes, &x_opt.view());
            let res = OptimResult {
                x_opt: x_opt.row(0).to_owned(),
                y_opt: opt.y_opt,
            };
            log::info!(
                "Mixint Optim Result: min f(x)={} at x={}  ",
                res.y_opt,
                res.x_opt
            );
            res
        })
    }
}

/// A PreProcessor for the function under optimization taking into account
/// discrete input variables specification.
pub struct MixintPreProcessor {
    xtypes: Vec<Xtype>,
}

impl PreProcessor for MixintPreProcessor {
    /// cast continuous input as discrete input following types spec
    fn run(&self, x: &Array2<f64>) -> Array2<f64> {
        let fold = fold_with_enum_index(&self.xtypes, &x.view());
        cast_to_discrete_values(&self.xtypes, &fold)
    }
}

impl MixintPreProcessor {
    /// Constrcutor with given `xtypes` specification
    pub fn new(xtypes: &[Xtype]) -> Self {
        MixintPreProcessor {
            xtypes: xtypes.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixint::Xtype;
    use approx::assert_abs_diff_eq;
    use egobox_moe::MoeParams;
    use ndarray::{array, Array2, ArrayView2};

    #[cfg(not(feature = "blas"))]
    use linfa_linalg::norm::*;
    #[cfg(feature = "blas")]
    use ndarray_linalg::Norm;

    use serial_test::serial;

    fn mixsinx(x: &ArrayView2<f64>) -> Array2<f64> {
        if (x.mapv(|v| v.round()).norm_l2() - x.norm_l2()).abs() < 1e-6 {
            (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
        } else {
            panic!("Error: mixsinx works only on integer, got {:?}", x)
        }
    }

    #[test]
    #[serial]
    fn test_mixintegor_ei() {
        let n_eval = 30;
        let doe = array![[0.], [7.], [25.]];
        let xtypes = vec![Xtype::Int(0, 25)];

        let surrogate_builder = MoeParams::default();
        let surrogate_builder = MixintMoeParams::new(&xtypes, &surrogate_builder);
        let pre_proc = MixintPreProcessor::new(&xtypes);
        let mut mixintegor = MixintEgor::new(mixsinx, &surrogate_builder, &pre_proc);
        mixintegor
            .egor
            .doe(Some(doe))
            .n_eval(n_eval)
            .expect(Some(ApproxValue {
                value: -15.1,
                tolerance: 1e-1,
            }))
            .infill_strategy(InfillStrategy::EI);

        let res = mixintegor.minimize().unwrap();
        assert_abs_diff_eq!(array![18.], res.x_opt, epsilon = 2.);
    }

    #[test]
    #[serial]
    fn test_mixintegor_wb2() {
        let n_eval = 30;
        let xtypes = vec![Xtype::Int(0, 25)];

        let surrogate_builder = MoeParams::default();
        let surrogate_builder = MixintMoeParams::new(&xtypes, &surrogate_builder);
        let pre_proc = MixintPreProcessor::new(&xtypes);
        let mut mixintegor = MixintEgor::new(mixsinx, &surrogate_builder, &pre_proc);
        mixintegor
            .egor
            .n_eval(n_eval)
            .infill_strategy(InfillStrategy::WB2);

        let res = mixintegor.minimize().unwrap();
        assert_abs_diff_eq!(&array![18.], &res.x_opt, epsilon = 3.);
    }
}
