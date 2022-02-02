use criterion::{black_box, criterion_group, criterion_main, Criterion};
use doe::{SamplingMethod, LHS};
use gp::correlation_models::SquaredExponentialCorr;
use gp::mean_models::ConstantMean;
use gp::GaussianProcess;
use linfa::prelude::{Dataset, Fit};
use ndarray::{array, Array1, Zip};
use ndarray_npy::{read_npy, write_npy};
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;

fn criterion_gp(c: &mut Criterion) {
    let dims = vec![5, 10, 20, 60];
    let nts = vec![100, 300, 400, 800];

    let mut group = c.benchmark_group("gp");
    for i in 0..2 {
        let dim = dims[i];
        let nt = nts[i];
        let griewak = |x: &Array1<f64>| -> f64 {
            let d = Array1::linspace(1., dim as f64, dim).mapv(|v| v.sqrt());
            x.mapv(|v| v * v).sum() / 4000. - (x / &d).mapv(|v| v.cos()).fold(1., |acc, x| acc * x)
                + 1.0
        };
        let prefix = "gp";
        let xfilename = format!("{}_xt_{}x{}.npy", prefix, nt, dim);
        let yfilename = format!("{}_yt_{}x{}.npy", prefix, nt, 1);
        let xt = match read_npy(&xfilename) {
            Ok(xt) => xt,
            Err(_) => {
                let lim = array![[-600., 600.]];
                let xlimits = lim.broadcast((dim, 2)).unwrap();
                let rng = Isaac64Rng::seed_from_u64(42);
                let xt = LHS::new(&xlimits).with_rng(rng).sample(nt);
                write_npy(&xfilename, &xt).expect("cannot save xt");
                xt
            }
        };
        let yt = match read_npy(&yfilename) {
            Ok(yt) => yt,
            Err(_) => {
                let mut yv: Array1<f64> = Array1::zeros(xt.nrows());
                Zip::from(&mut yv).and(xt.rows()).par_for_each(|y, x| {
                    *y = griewak(&x.to_owned());
                });
                let yt = yv.into_shape((xt.nrows(), 1)).unwrap();
                write_npy(&yfilename, &yt).expect("cannot save yt");
                yt
            }
        };

        group.bench_function(format!("gp {}", dims[i]), |b| {
            b.iter(|| {
                black_box(
                    GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
                        ConstantMean::default(),
                        SquaredExponentialCorr::default(),
                    )
                    //.with_kpls_dim(1)
                    //.with_initial_theta(1.0)
                    .fit(&Dataset::new(xt.to_owned(), yt.to_owned()))
                    .expect("GP fit error"),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_gp);
criterion_main!(benches);
