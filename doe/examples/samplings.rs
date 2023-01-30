use egobox_doe::{FullFactorial, Lhs, LhsKind, Random, SamplingMethod};
use ndarray::arr2;

fn main() {
    let xlimits = arr2(&[[0., 1.], [-10., 10.], [5., 15.]]);
    let n = 10;

    println!("Take {n} samples in");
    println!("{xlimits}\n");

    println!("*** using random sampling");
    let samples = Random::new(&xlimits).sample(n);
    println!("{samples}\n");

    println!("*** using full-factorial sampling");
    let samples = FullFactorial::new(&xlimits).sample(n);
    println!("{samples}\n");

    println!("*** using centered latin hypercube sampling");
    let samples = Lhs::new(&xlimits).kind(LhsKind::Centered).sample(n);
    println!("{samples}\n");

    println!("*** using optimized latin hypercube sampling");
    let samples = Lhs::new(&xlimits).sample(n);
    println!("{samples}\n");
}
