use clap::Parser;

use egobox_ego::{EgorBuilder, InfillOptimizer, InfillStrategy, QEiStrategy, RunInfo};
use egobox_moe::{CorrelationSpec, RegressionSpec};
use ndarray::{Array, Array2, ArrayView2, Zip, array};

/// Ackley test function: min f(x)=0 at x=(0, 0, 0)
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
    #[arg(short, long, default_value_t = 10)]
    dim: usize,
    #[arg(short, long, default_value = "./history")]
    outdir: String,
}

fn main() {
    let args = Args::parse();

    let ndim = args.dim;
    let outdir = args.outdir;

    let data = [-32.768, 32.768].repeat(ndim);
    let xlimits = Array::from_shape_vec((ndim, 2), data).unwrap();

    let res = EgorBuilder::optimize(ackley)
        .configure(|config| {
            config
                .n_doe(200)
                .configure_gp(|gp| {
                    gp.regression_spec(RegressionSpec::CONSTANT)
                        .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
                })
                .infill_strategy(InfillStrategy::EI)
                .infill_optimizer(InfillOptimizer::Slsqp)
                .coego(egobox_ego::CoegoStatus::Enabled(5))
                .q_points(10)
                .q_optmod(2)
                .qei_strategy(QEiStrategy::KrigingBeliever)
                .n_start(50)
                .outdir(outdir)
                .max_iters(30)
        })
        .min_within(&xlimits)
        .run_info(RunInfo {
            fname: "ackley".to_string(),
            num: 1,
        })
        .run()
        .expect("Minimize failure");
    println!("Ackley minimum y = {} at x = {}", res.y_opt, res.x_opt);
}
