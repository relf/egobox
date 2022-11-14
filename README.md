# egobox

[![tests](https://github.com/relf/egobox/workflows/tests/badge.svg)](https://github.com/relf/egobox/actions?query=workflow%3Atests)
[![pytests](https://github.com/relf/egobox/workflows/pytests/badge.svg)](https://github.com/relf/egobox/actions?query=workflow%3Apytests)
[![linting](https://github.com/relf/egobox/workflows/lint/badge.svg)](https://github.com/relf/egobox/actions?query=workflow%3Alint)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04737/status.svg)](https://doi.org/10.21105/joss.04737)

Rust toolbox for Efficient Global Optimization algorithms inspired from [SMT](https://github.com/SMTorg/smt). 

`egobox` is twofold: 
1. for developers: [a set of Rust libraries](#the-rust-libraries) useful to implement bayesian optimization (EGO-like) algorithms,
2. for end-users: [a Python module](#the-python-optimizer-egor), the Python binding of the implemented EGO-like optimizer, named `Egor` and surrogate model `Gpx`, mixture of Gaussian processes. 

## The Rust libraries

`egobox` Rust libraries consists of the following sub-packages.

| Name                                                  | Version                                                                                         | Documentation                                                               | Description                                                                     |
| :---------------------------------------------------- | :---------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| [doe](https://github.com/relf/egobox/tree/master/doe) | [![crates.io](https://img.shields.io/crates/v/egobox-doe)](https://crates.io/crates/egobox-doe) | [![docs](https://docs.rs/egobox-doe/badge.svg)](https://docs.rs/egobox-doe) | sampling methods; contains LHS, FullFactorial, Random methods                   |
| [gp](https://github.com/relf/egobox/tree/master/gp)   | [![crates.io](https://img.shields.io/crates/v/egobox-gp)](https://crates.io/crates/egobox-gp)   | [![docs](https://docs.rs/egobox-gp/badge.svg)](https://docs.rs/egobox-gp)   | gaussian process regression; contains Kriging and PLS dimension reduction       |
| [moe](https://github.com/relf/egobox/tree/master/moe) | [![crates.io](https://img.shields.io/crates/v/egobox-moe)](https://crates.io/crates/egobox-moe) | [![docs](https://docs.rs/egobox-moe/badge.svg)](https://docs.rs/egobox-moe) | mixture of experts using GP models                                              |
| [ego](https://github.com/relf/egobox/tree/master/ego) | [![crates.io](https://img.shields.io/crates/v/egobox-ego)](https://crates.io/crates/egobox-ego) | [![docs](https://docs.rs/egobox-ego/badge.svg)](https://docs.rs/egobox-ego) | efficient global optimization with basic constraints and mixed integer handling |

### Usage

Depending on the sub-packages you want to use, you have to add following declarations to your `Cargo.toml`

```
[dependencies]
egobox-doe = { version = "0.6.0" }
egobox-gp  = { version = "0.6.0" }
egobox-moe = { version = "0.6.0" }
egobox-ego = { version = "0.6.0" }
```

### Features
#### `serializable-gp` 

The `serializable-gp` feature enables the serialization of GP models using the [serde crate](https://serde.rs/). 

#### `persistent-moe` 

The `persistent-moe` feature enables `save()` and `load()` methods for MoE model to/from a json file using the [serde crate](https://serde.rs/). 

### Examples

Examples (in `examples/` sub-packages folder) are run as follows:

```bash
$ cd doe && cargo run --example samplings --release
```

``` bash
$ cd gp && cargo run --example kriging --release
```

``` bash
$ cd moe && cargo run --example clustering --release
```

``` bash
$ cd ego && cargo run --example ackley --release
```

### BLAS/LAPACK backend (optional)

`egobox` relies on [linfa](https://github.com/rust-ml/linfa) project for methods like clustering and dimension reduction, but also try to adopt as far as possible the same [coding structures](https://github.com/rust-ml/linfa/blob/master/CONTRIBUTE.md).

As for `linfa`, the linear algebra routines used in `gp`, `moe` ad `ego` are provided by the pure-Rust [linfa-linalg](https://github.com/rust-ml/linfa-linalg) crate, the default linear algebra provider.

Otherwise, you can choose an external BLAS/LAPACK backend available through the [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg) crate. In this case, you have to specify the `blas` feature and a `linfa` [BLAS/LAPACK backend feature](https://github.com/rust-ml/linfa#blaslapack-backend) (more information in [linfa features](https://github.com/rust-ml/linfa#blaslapack-backend)).

Thus, for instance, to use `gp` with the Intel MKL BLAS/LAPACK backend, you could specify in your `Cargo.toml` the following features:
```
[dependencies]
egobox-gp = { version = "0.6.0", features = ["blas", "linfa/intel-mkl-static"] }
```
or you could run the `gp` example as follows:
``` bash
$ cd gp && cargo run --example kriging --release --features blas,linfa/intel-mkl-static
```

## The `egobox` Python binding

Thanks to the [PyO3 project](https://pyo3.rs), which makes Rust well suited for building Python extensions. 
You can install the Python package using:

```bash
$ pip install egobox
```

See the [tutorial notebooks](https://github.com/relf/egobox/tree/master/doc) for usage of the optimizer 
and mixture of Gaussian processes surrogate model.

## Citation

[![DOI](https://joss.theoj.org/papers/10.21105/joss.04737/status.svg)](https://doi.org/10.21105/joss.04737)

If you find this project useful for your research, you may cite it as follows: 

```
@article{
  Lafage2022, 
  author = {Rémi Lafage}, 
  title = {egobox, a Rust toolbox for efficient global optimization}, 
  journal = {Journal of Open Source Software} 
  year = {2022}, 
  doi = {10.21105/joss.04737}, 
  url = {https://doi.org/10.21105/joss.04737}, 
  publisher = {The Open Journal}, 
  volume = {7}, 
  number = {78}, 
  pages = {4737}, 
} 
```

Additionally, you may consider adding a star to the repository. This positive feedback improves the visibility of the project.

## History

I started this library as a way to learn Rust and see if it can be used to implement algorithms like those in the SMT toolbox[^1]. As the first components (doe, gp) emerged, it appears I could translate Python code almost line by line in Rust (well... after a great deal of borrow-checker fight!) and thanks to [Rust ndarray library ecosystem](https://github.com/rust-ndarray). 

This library relies also on the [linfa project](https://github.com/rust-ml/linfa) which aims at being the "scikit-learn-like ML library for Rust". Along the way I could contribute to `linfa` by porting gaussian mixture model (`linfa-clustering/gmm`) and partial least square family methods (`linfa-pls`) confirming the fact that Python algorithms translation in Rust could be pretty straightforward.

While I did not benchmark exactly my Rust code against SMT Python one, from my debugging sessions, I noticed I did not get such a great speed up. Actually, algorithms like `doe` and `gp` relies extensively on linear algebra and Python famous libraries `numpy`/`scipy` which are strongly optimized by calling C or Fortran compiled code.

My guess at this point was that interest could come from some Rust algorithms built upon these initial building blocks hence I started to implement mixture of experts algorithm (`moe`) and on top surrogate-based optimization EGO algorithm (`ego`) which gives its name to the library[^2][^3]. Aside from performance, such library can also take advantage from the others [Rust selling points](https://www.rust-lang.org/). 


[^1]: M. A. Bouhlel and J. T. Hwang and N. Bartoli and R. Lafage and J. Morlier and J. R. R. A. Martins. A Python surrogate modeling framework with derivatives. Advances in Engineering Software, 2019.

[^2]: Bartoli, Nathalie, et al. "Adaptive modeling strategy for constrained global optimization with application to aerodynamic wing design." Aerospace Science and technology 90 (2019): 85-102.

[^3]: Dubreuil, Sylvain, et al. "Towards an efficient global multidisciplinary design optimization algorithm." Structural and Multidisciplinary Optimization 62.4 (2020): 1739-1765.
