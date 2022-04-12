# egobox

[![tests](https://github.com/relf/egobox/workflows/tests/badge.svg)](https://github.com/relf/egobox/actions?query=workflow%3Atests)

Rust toolbox for Efficient Global Optimization algorithms inspired from [SMT](https://github.com/SMTorg/smt). 
This library provides a port of the following algorithms:
* `doe`, sampling methods: LHS, FullFactorial, Random
* `gp`, gaussian process regression: Kriging and KPLS surrogates
* `moe`, mixture of experts using kriging models
* `ego`, efficient global optimization with basic constraints and mixed integer handling 

## Usage

Examples can be run as follows:

```bash
cd doe
cargo run --example samplings --release
```

`gp`, `moe` and `ego` modules relies on `linfa` [BLAS/Lapack backend features](https://github.com/rust-ml/linfa#blaslapack-backend). 

Using the Intel MKL BLAS/Lapack backend, you can run :

``` bash
cd gp
cargo run --example kriging --release --features linfa/intel-mkl-static
```

``` bash
cd moe
cargo run --example clustering --release --features linfa/intel-mkl-static
```

``` bash
cd ego
cargo run --example ackley --release --features linfa/intel-mkl-static
```

Thanks to the [PyO3 project](https://pyo3.rs), which makes Rust well suited for building Python extensions, the EGO algorithm written in Rust (aka `egor`) is binded in Python. You can install the Python package using:

```bash
pip install egobox
```

See the [tutorial notebook](doc/TutorialEgor.ipynb) for usage.

## Why egobox?

I started this library as a way to learn Rust and see if it can be used to implement algorithms like those in the SMT toolbox[^1]. As the first components (doe, gp) emerged, it appeears I could translate Python code almost line by line in Rust (well... after great deal of borrow-checker fight!) and thanks to [Rust ndarray library ecosystem](https://github.com/rust-ndarray). 

This library relies also on the [linfa project](https://github.com/rust-ml/linfa) which aims at being the "scikit-learn-like ML library for Rust". Along the way I could contribute to `linfa` by porting gaussian mixture model (`linfa-clustering/gmm`) and partial least square family methods (`linfa-pls`) confirming the fact that Python algorithms translation in Rust could be pretty straightforward.

While I did not benchmark my Rust code against SMT Python one, from my debugging sessions, I noticed I did not get such a great speed up. Actually, algorithms like `doe` and `gp` relies extensively on linear algebra and Python famous libraries `numpy`/`scipy` which are strongly optimized by calling C or Fortran compiled code.

My guess at this point is that interest could come from other Rust algorithms built upon these initial building blocks hence I started to implement mixture of experts algorithm (`moe`) and on top bayesian optimization EGO algorithm (`ego`) which gives its name to the library[^2]. Aside from performance, such library benefits from Rust others selling points, namely reliability and productivity. 
## Cite

If you happen to find this Rust library useful for your research, you can cite this project as follows: 

```
@Misc{,
  author = {Rémi Lafage},
  title = {Egobox: efficient global optimization toolbox in Rust},
  year = {2020--},
  url = "https://github.com/relf/egobox"
}
```

[^1]: M. A. Bouhlel and J. T. Hwang and N. Bartoli and R. Lafage and J. Morlier and J. R. R. A. Martins. A Python surrogate modeling framework with derivatives. Advances in Engineering Software, 2019.

[^2]: Bartoli, Nathalie, et al. "Adaptive modeling strategy for constrained global optimization with application to aerodynamic wing design." Aerospace Science and technology 90 (2019): 85-102.
