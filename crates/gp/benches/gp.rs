use criterion::{Criterion, criterion_group, criterion_main};
use egobox_doe::{Lhs, SamplingMethod};
use egobox_gp::GaussianProcess;
use egobox_gp::correlation_models::SquaredExponentialCorr;
use egobox_gp::mean_models::ConstantMean;
use linfa::prelude::{Dataset, Fit};
use ndarray::{Array1, Zip, array};
use ndarray_npy::{read_npy, write_npy};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

fn criterion_gp(c: &mut Criterion) {
    let dims = [5, 10, 20, 60];
    let nts = [100, 300, 400, 800];

    let mut group = c.benchmark_group("gp");
    group.sample_size(20);
    for i in 0..2 {
        let dim = dims[i];
        let nt = nts[i];
        let griewank = |x: &Array1<f64>| -> f64 {
            let d = Array1::linspace(1., dim as f64, dim).mapv(|v| v.sqrt());
            x.mapv(|v| v * v).sum() / 4000. - (x / &d).mapv(|v| v.cos()).fold(1., |acc, x| acc * x)
                + 1.0
        };
        let prefix = "gp";
        let xfilename = format!("{prefix}_xt_{nt}x{dim}.npy");
        let yfilename = format!("{}_yt_{}x{}.npy", prefix, nt, 1);
        let xt = match read_npy(&xfilename) {
            Ok(xt) => xt,
            Err(_) => {
                let lim = array![[-600., 600.]];
                let xlimits = lim.broadcast((dim, 2)).unwrap();
                let rng = Xoshiro256Plus::seed_from_u64(42);
                let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
                write_npy(&xfilename, &xt).expect("cannot save xt");
                xt
            }
        };
        let yt = match read_npy(&yfilename) {
            Ok(yt) => yt,
            Err(_) => {
                let mut yt: Array1<f64> = Array1::zeros(xt.nrows());
                Zip::from(&mut yt).and(xt.rows()).par_for_each(|y, x| {
                    *y = griewank(&x.to_owned());
                });
                write_npy(&yfilename, &yt).expect("cannot save yt");
                yt
            }
        };

        group.bench_function(format!("gp {}", dims[i]), |b| {
            b.iter(|| {
                std::hint::black_box(
                    GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
                        ConstantMean::default(),
                        SquaredExponentialCorr::default(),
                    )
                    .kpls_dim(Some(1))
                    .theta_init(array![1.0])
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
