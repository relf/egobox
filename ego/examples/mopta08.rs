use clap::Parser;
use egobox_ego::{EgorBuilder, GroupFunc, InfillOptimizer};
use egobox_moe::{RegressionSpec, CorrelationSpec};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufRead, BufReader};
use std::process::Command;

const DIM_X: usize = 124;
const DIM_Y: usize = 69;

/// Mopta08 reference solution
const REFSOL: [f64; DIM_X] = [
    0.424901536845442,
    0.000000000000000,
    -0.000000000000000,
    0.000000000000000,
    0.000000000000000,
    0.070368163337397,
    0.191758133431455,
    0.659581587650616,
    0.312511963893922,
    1.000000000000000,
    0.000000000000000,
    -0.000000000000000,
    0.000000000000000,
    0.000000000000000,
    -0.000000000000000,
    0.500523655520237,
    0.000000000000000,
    0.000000000000000,
    0.000000000000000,
    -0.000000000000000,
    -0.000000000000000,
    0.007323697294187,
    -0.000000000000000,
    0.000000000000000,
    0.000000000000000,
    0.691424942460969,
    0.549284506769106,
    -0.000000000000000,
    -0.000000000000000,
    0.634896580136233,
    0.246440520422731,
    0.312017956345506,
    0.000000000000000,
    1.000000000000000,
    1.000000000000000,
    0.317952882489170,
    0.855170249994022,
    0.658962493142454,
    0.567339401149320,
    0.191865333482370,
    0.906569943802102,
    0.612762717451940,
    0.033974026079705,
    0.250858210690466,
    -0.000000000000000,
    0.000000000000000,
    0.397548021112299,
    0.050453268214836,
    0.144718492777280,
    0.065661350680142,
    -0.000000000000000,
    -0.000000000000000,
    0.401813779445987,
    -0.000000000000000,
    0.000000000000000,
    0.181619715253390,
    0.000000000000000,
    -0.000000000000000,
    0.114195768540203,
    0.393419805218654,
    0.047400618876390,
    0.000000000000000,
    0.877814656956643,
    0.767294038418826,
    0.000000000000000,
    0.853786893241821,
    0.603638877543724,
    0.330387862928772,
    0.000000000000000,
    0.204560994583241,
    0.221383426733140,
    0.000000000000000,
    0.000000000000000,
    0.094398339150782,
    -0.000000000000000,
    0.308945500999398,
    -0.000000000000000,
    0.320954495292004,
    0.735573944494521,
    0.000000000000000,
    0.510228205500305,
    0.558204538496431,
    0.527207739609047,
    0.827802688106254,
    -0.000000000000000,
    0.665687979989158,
    0.724380930097933,
    0.287131241910361,
    0.414606427865539,
    -0.000000000000000,
    0.419521821597494,
    0.388002834344305,
    0.029815463457816,
    0.631619810505481,
    -0.000000000000000,
    0.000000000000000,
    0.192333290055701,
    0.000000000000000,
    0.692543616440458,
    0.669383104588378,
    -0.000000000000000,
    1.000000000000000,
    0.729141260482755,
    0.161236677500071,
    0.561446507760961,
    0.487479618244102,
    0.728572274406474,
    0.240635584133437,
    -0.000000000000000,
    1.000000000000000,
    0.383816619144691,
    1.000000000000000,
    1.000000000000000,
    0.000000000000000,
    1.000000000000000,
    0.152841300847909,
    0.158379914361422,
    0.210185493857930,
    0.889704350522390,
    0.791266802460270,
    -0.000000000000000,
    0.931445345989215,
    -0.000000000000000,
    -0.000000000000000,
];

fn set_input(x: &ArrayView1<f64>, opt_indices: Option<&[usize]>) {
    let mut x_input = File::create("input.txt").unwrap();
    (0..DIM_X).for_each(|j| {
        if let Some(indices) = opt_indices {
            if indices.contains(&j) {
                let idx = indices.iter().position(|&v| v == j).unwrap();
                writeln!(x_input, "{:.16}", x[[idx]]).expect("x input written");
            } else {
                writeln!(x_input, "{:.16}", REFSOL[j]).expect("input written");
            }
        } else {
            writeln!(x_input, "{:.16}", x[j]).expect("x 124 input written");
        }
    });
}

fn get_output() -> anyhow::Result<Array1<f64>> {
    let file = File::open("output.txt")?;
    let reader = BufReader::new(file);
    let mut output = Array1::zeros(DIM_Y);
    for (i, line) in reader.lines().enumerate() {
        let str = line?;
        let y = str.trim().parse::<f64>().unwrap();
        output[i] = y;
    }
    Ok(output)
}

/// Mopta08 test case: min f(x)=~222.74 at x=REFSOL
fn mopta(x: &ArrayView2<f64>, indices: Option<&[usize]>) -> Array2<f64> {
    let n = x.nrows();
    let mut y = Array2::zeros((n, DIM_Y));

    for i in 0..n {
        set_input(&x.row(i), indices);

        let mut path_exe = std::env::current_dir().unwrap();
        let mopta_exe = if cfg!(windows) {
            "mopta08.exe"
        } else {
            "mopta08_elf64.bin"
        };
        path_exe.push(r"ego/examples");
        path_exe.push(mopta_exe);

        Command::new(path_exe)
            .spawn()
            .expect("ls command failed to start");

        std::thread::sleep(std::time::Duration::from_secs(1));
        let y_i = get_output().unwrap();
        y.row_mut(i).assign(&y_i);
    }
    y
}

fn mopta12d(x: &ArrayView2<f64>) -> Array2<f64> {
    let indices_12d = vec![1, 41, 44, 45, 48, 50, 59, 65, 71, 77, 81, 94];
    mopta(x, Some(&indices_12d))
}

fn mopta30d(x: &ArrayView2<f64>) -> Array2<f64> {
    let indices_30d = vec![
        1, 2, 5, 12, 13, 14, 21, 23, 28, 32, 33, 35, 41, 44, 45, 48, 50, 53, 59, 65, 68, 71, 77,
        81, 84, 85, 94, 95, 100, 102,
    ];
    mopta(x, Some(&indices_30d))
}

fn mopta50d(x: &ArrayView2<f64>) -> Array2<f64> {
    let indices_30d = vec![
        1, 2, 5, 10, 12, 13, 14, 18, 20, 21, 23, 28, 29, 30, 32, 33, 35, 39, 41, 44, 45, 48, 50,
        52, 53, 55, 56, 59, 61, 62, 64, 65, 68, 70, 71, 75, 76, 77, 80, 81, 84, 85, 91, 92, 94, 95,
        100, 102, 120, 123,
    ];
    mopta(x, Some(&indices_30d))
}

fn mopta124d(x: &ArrayView2<f64>) -> Array2<f64> {
    mopta(x, None)
}

fn mopta_func(dim: usize) -> impl Fn(&ArrayView2<f64>) -> Array2<f64> + GroupFunc {
    match dim {
        30 => mopta30d,
        50 => mopta50d,
        124 => mopta124d,
        _ => mopta12d,
    }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 12)]
    dim: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let dim = args.dim;
    let cstr_tol = 1e-4;
    let mut xlimits = Array2::zeros((dim, 2));
    xlimits.column_mut(1).assign(&Array1::ones(dim));
    let res = EgorBuilder::optimize(mopta_func(dim))
        .min_within(&xlimits)
        .n_cstr(68)
        .cstr_tol(cstr_tol)
        .n_clusters(Some(0))
        .n_doe(250)
        .n_eval(100)
        .regression_spec(RegressionSpec::CONSTANT)
        .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
        .infill_optimizer(InfillOptimizer::Cobyla)
        .kpls_dim(Some(3))
        .outdir(Some("./history".to_string()))
        .hot_start(true)
        .run()
        .expect("Minimize failure");
    println!(
        "Mopta08 dim={} minimum y = {} at x = {}",
        dim, res.y_opt, res.x_opt
    );
    println!(
        "Violated constraints = {}/68",
        res.y_opt
            .to_vec()
            .into_iter()
            .filter(|v| *v > cstr_tol)
            .count()
    );
    Ok(())
}
