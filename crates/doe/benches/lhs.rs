use criterion::{Criterion, criterion_group, criterion_main};
use egobox_doe::{Lhs, SamplingMethod};
use ndarray::aview1;
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoroshiro128Plus;

fn criterion_lhs(c: &mut Criterion) {
    let dims = [100];
    let sizes = [10, 100];

    let mut group = c.benchmark_group("doe");
    group.sample_size(10);
    let arr1 = aview1(&[0., 1.]);
    let rng = Xoroshiro128Plus::seed_from_u64(42);
    for dim in dims {
        for size in sizes {
            group.bench_function(format!("lhs-{dim}-dim-{size}-size"), |b| {
                let xlimits = arr1.broadcast((dim, 2)).unwrap();
                b.iter(|| {
                    std::hint::black_box(Lhs::new(&xlimits).with_rng(rng.clone()).sample(size))
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, criterion_lhs);
criterion_main!(benches);
