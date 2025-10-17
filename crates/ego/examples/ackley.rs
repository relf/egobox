use clap::Parser;

use egobox_ego::{
    EgorBuilder, InfillOptimizer, InfillStrategy, OptimResult, QEiStrategy, Result, RunInfo,
};
use egobox_moe::{CorrelationSpec, RegressionSpec};
use ndarray::{Array, Array2, ArrayView2, Zip, array};

/// Ackley test function: min f(x)=0 at x=(0, 0, ..., 0)
fn ackley(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .par_for_each(|mut yi, xi| yi.assign(&array![argmin_testfunctions::ackley(&xi.to_vec(),)]));
    y
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 100)]
    dim: usize,
    #[arg(short, long, default_value = "./ackley")]
    outdir: String,
    #[arg(short, long, default_value_t = 1)]
    rep: usize,
}

const BUDGET: usize = 500;

fn run_egor(dim: usize, outdir: &String, num: usize) -> Result<OptimResult<f64>> {
    let n_doe = dim + 1;
    let max_iters = BUDGET - n_doe;

    let data = [-32.768, 32.768].repeat(dim);
    let xlimits = Array::from_shape_vec((dim, 2), data).unwrap();

    EgorBuilder::optimize(ackley)
        .configure(|config| {
            config
                .n_doe(n_doe)
                .configure_gp(|gp| {
                    gp.regression_spec(RegressionSpec::CONSTANT)
                        .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
                })
                .infill_strategy(InfillStrategy::EI)
                .infill_optimizer(InfillOptimizer::Cobyla)
                .coego(egobox_ego::CoegoStatus::Enabled(5))
                .trego(true)
                .q_points(10)
                .q_optmod(2)
                .qei_strategy(QEiStrategy::KrigingBeliever)
                .n_start(150)
                .outdir(outdir)
                .max_iters(max_iters)
        })
        .min_within(&xlimits)
        .run_info(RunInfo {
            fname: "ackley".to_string(),
            num,
        })
        .run()
}

fn main() -> Result<()> {
    let args = Args::parse();

    let dim = args.dim;
    let outdir = args.outdir;
    let rep = args.rep;

    for num in 1..=rep {
        println!(">>>> Run {} of 30", num);
        let outdir = format!("{}/run{:0>2}", outdir, num);
        let res = run_egor(dim, &outdir, num)?;

        println!("Ackley minimum y = {} at x = {}", res.y_opt, res.x_opt);
    }

    Ok(())
}
