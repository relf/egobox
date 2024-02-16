use criterion::{black_box, criterion_group, criterion_main, Criterion};
use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use ndarray::aview1;

fn criterion_lhs_classics(c: &mut Criterion) {
    let dims = [500];
    let sizes = [1000];
    let kinds = [LhsKind::Classic, LhsKind::Maximin, LhsKind::Centered];

    let mut group = c.benchmark_group("doe");
    group.sample_size(10);
    let arr1 = aview1(&[0., 1.]);
    for dim in dims {
        for size in sizes {
            for kind in kinds {
                group.bench_function(format!("lhs-{kind:?}-{dim}-dim-{size}-size"), |b| {
                    let xlimits = arr1.broadcast((dim, 2)).unwrap();
                    b.iter(|| black_box(Lhs::new(&xlimits).kind(kind).sample(size)));
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, criterion_lhs_classics);
criterion_main!(benches);
