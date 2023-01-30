use criterion::{black_box, criterion_group, criterion_main, Criterion};
use egobox_doe::{Lhs, SamplingMethod};
use ndarray::arr2;

fn criterion_lhs(c: &mut Criterion) {
    let sizes = vec![10, 100, 1000];

    let mut group = c.benchmark_group("doe");
    for size in sizes {
        group.bench_function(format!("lhs {size}"), |b| {
            let xlimits = arr2(&[[0., 1.], [0., 1.]]);
            b.iter(|| black_box(Lhs::new(&xlimits).sample(size)));
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_lhs);
criterion_main!(benches);
