use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use egobox_gp::correlation_models::{Matern32Corr, Matern52Corr, CorrelationModel};
use ndarray::{Array2, arr1, array};

fn bench_matern32_value(c: &mut Criterion) {
    let mut group = c.benchmark_group("matern32_value");
    
    for size in [10, 50, 100].iter() {
        // Create distance matrix
        let d: Array2<f64> = Array2::from_elem((*size, 5), 0.5);
        let theta = arr1(&[1.0]);
        let weights = array![[1., 0., 0., 0., 0.]];
        
        let corr = Matern32Corr::default();
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| corr.value(&d, &theta, &weights));
        });
    }
    group.finish();
}

fn bench_matern52_value(c: &mut Criterion) {
    let mut group = c.benchmark_group("matern52_value");
    
    for size in [10, 50, 100].iter() {
        // Create distance matrix
        let d: Array2<f64> = Array2::from_elem((*size, 5), 0.5);
        let theta = arr1(&[1.0]);
        let weights = array![[1., 0., 0., 0., 0.]];
        
        let corr = Matern52Corr::default();
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| corr.value(&d, &theta, &weights));
        });
    }
    group.finish();
}

fn bench_matern32_jacobian(c: &mut Criterion) {
    let mut group = c.benchmark_group("matern32_jacobian");
    
    for size in [10, 50, 100].iter() {
        let x = arr1(&[0.3, 0.5, 0.7, 0.2, 0.9]);
        let xtrain: Array2<f64> = Array2::from_shape_fn((*size, 5), |(i, j)| {
            (i as f64 * 0.1 + j as f64 * 0.05) % 1.0
        });
        let theta = arr1(&[1.0]);
        let weights = array![[1., 0., 0., 0., 0.]];
        
        let corr = Matern32Corr::default();
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| corr.jacobian(&x, &xtrain, &theta, &weights));
        });
    }
    group.finish();
}

fn bench_matern52_jacobian(c: &mut Criterion) {
    let mut group = c.benchmark_group("matern52_jacobian");
    
    for size in [10, 50, 100].iter() {
        let x = arr1(&[0.3, 0.5, 0.7, 0.2, 0.9]);
        let xtrain: Array2<f64> = Array2::from_shape_fn((*size, 5), |(i, j)| {
            (i as f64 * 0.1 + j as f64 * 0.05) % 1.0
        });
        let theta = arr1(&[1.0]);
        let weights = array![[1., 0., 0., 0., 0.]];
        
        let corr = Matern52Corr::default();
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| corr.jacobian(&x, &xtrain, &theta, &weights));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matern32_value,
    bench_matern52_value,
    bench_matern32_jacobian,
    bench_matern52_jacobian
);
criterion_main!(benches);
