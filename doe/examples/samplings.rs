use egobox_doe::{FullFactorial, Lhs, LhsKind, Random, SamplingMethod};
use ndarray::arr2;

fn main() {
    let xlimits = arr2(&[[0., 1.], [-10., 10.], [5., 15.]]);
    let n = 10;

    println!("Take {} samples in", n);
    println!("{}\n", xlimits);

    println!("*** using random sampling");
    let samples = Random::new(&xlimits).sample(n);
    println!("{}\n", samples);

    println!("*** using full-factorial sampling");
    let samples = FullFactorial::new(&xlimits).sample(n);
    println!("{}\n", samples);

    println!("*** using centered latin hypercube sampling");
    let samples = Lhs::new(&xlimits).kind(LhsKind::Centered).sample(n);
    println!("{}\n", samples);

    println!("*** using optimized latin hypercube sampling");
    let samples = Lhs::new(&xlimits).sample(n);
    println!("{}\n", samples);
}
