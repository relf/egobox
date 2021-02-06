use criterion::{black_box, criterion_group, criterion_main, Criterion, ParameterizedBenchmark};
use egobox::doe::{SamplingMethod, LHS};
use ndarray::arr2;

fn criterion_lhs(c: &mut Criterion) {
    let sizes = vec![10];

    let benchmark = ParameterizedBenchmark::new(
        "lhs",
        move |bencher, &size| {
            let xlimits = arr2(&[[0., 1.], [0., 1.]]);
            bencher.iter(|| black_box(LHS::new(&xlimits).sample(size)));
        },
        sizes,
    );

    c.bench("lhs", benchmark);
}

criterion_group!(benches, criterion_lhs);
criterion_main!(benches);
