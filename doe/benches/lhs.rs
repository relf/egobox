use criterion::{black_box, criterion_group, criterion_main, Criterion};
use egobox_doe::{Lhs, SamplingMethod};
use ndarray::aview1;

fn criterion_lhs(c: &mut Criterion) {
    let dims = [100];
    let sizes = [10, 100];

    let mut group = c.benchmark_group("doe");
    group.sample_size(10);
    let arr1 = aview1(&[0., 1.]);
    for dim in dims {
        for size in sizes {
            group.bench_function(format!("lhs-{dim}-dim-{size}-size"), |b| {
                let xlimits = arr1.broadcast((dim, 2)).unwrap();
                b.iter(|| black_box(Lhs::new(&xlimits).sample(size)));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, criterion_lhs);
criterion_main!(benches);
