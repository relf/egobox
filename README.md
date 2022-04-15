# egobox

[![tests](https://github.com/relf/egobox/workflows/tests/badge.svg)](https://github.com/relf/egobox/actions?query=workflow%3Atests)
[![pytests](https://github.com/relf/egobox/workflows/pytests/badge.svg)](https://github.com/relf/egobox/actions?query=workflow%3Apytests)
[![linting](https://github.com/relf/egobox/workflows/lint/badge.svg)](https://github.com/relf/egobox/actions?query=workflow%3Alint)


Rust toolbox for Efficient Global Optimization algorithms inspired from [SMT](https://github.com/SMTorg/smt). 

`egobox` consists of the following sub-packages.

| Name         | Version                                                                                         | Documentation                                                               | Description                                                                     |
| :----------- | :---------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| [doe](./doe) | [![crates.io](https://img.shields.io/crates/v/egobox-doe)](https://crates.io/crates/egobox-doe) | [![docs](https://docs.rs/egobox-doe/badge.svg)](https://docs.rs/egobox-doe) | sampling methods; contains LHS, FullFactorial, Random methods          |
| [gp](./gp)   | [![crates.io](https://img.shields.io/crates/v/egobox-gp)](https://crates.io/crates/egobox-gp)   | [![docs](https://docs.rs/egobox-gp/badge.svg)](https://docs.rs/egobox-gp)   | gaussian process regression; contains Kriging and PLS dimension reduction      |
| [moe](./gp)  | [![crates.io](https://img.shields.io/crates/v/egobox-moe)](https://crates.io/crates/egobox-moe) | [![docs](https://docs.rs/egobox-moe/badge.svg)](https://docs.rs/egobox-moe) | mixture of experts using GP models                                              |
| [ego](./ego) | [![crates.io](https://img.shields.io/crates/v/egobox-ego)](https://crates.io/crates/egobox-ego) | [![docs](https://docs.rs/egobox-ego/badge.svg)](https://docs.rs/egobox-ego) | efficient global optimization with basic constraints and mixed integer handling |

## Usage

Depending on the sub-packages you want to use, you have to add following declarations to your `Cargo.toml`

```
[dependencies]
egobox-doe = { version = "0.2.1" }
egobox-gp  = { version = "0.2.1" }
egobox-moe = { version = "0.2.1" }
egobox-ego = { version = "0.2.1" }
```

## Features

`gp`, `moe` and `ego` relies on `linfa` [BLAS/Lapack backend features](https://github.com/rust-ml/linfa#blaslapack-backend).

For instance, using `gp` with the Intel MKL BLAS/Lapack backend, you have to specify the linfa backend feature :

```
[dependencies]
egobox-gp = { version = "0.2.1", features = ["linfa/intel-mkl-static"] }
```

Note: only end-user projects should specify a provider in `Cargo.toml` (not librairies). In case of library development, the backend is specified on the command line as for examples below.

## Examples

Examples (in `examples/` sub-packages folder) are run as follows:

```bash
$ cd doe && cargo run --example samplings --release
```

Using the Intel MKL BLAS/Lapack backend, you can run :

``` bash
$ cd gp && cargo run --example kriging --release --features linfa/intel-mkl-static
```

``` bash
$ cd moe && cargo run --example clustering --release --features linfa/intel-mkl-static
```

``` bash
$ cd ego && cargo run --example ackley --release --features linfa/intel-mkl-static
```

Thanks to the [PyO3 project](https://pyo3.rs), which makes Rust well suited for building Python extensions, the EGO algorithm written in Rust (aka `Egor`) is binded in Python. You can install the Python package using:

```bash
$ pip install egobox
```

See the [tutorial notebook](doc/TutorialEgor.ipynb) for usage of the optimizer.

## Why egobox?

I started this library as a way to learn Rust and see if it can be used to implement algorithms like those in the SMT toolbox[^1]. As the first components (doe, gp) emerged, it appears I could translate Python code almost line by line in Rust (well... after a great deal of borrow-checker fight!) and thanks to [Rust ndarray library ecosystem](https://github.com/rust-ndarray). 

This library relies also on the [linfa project](https://github.com/rust-ml/linfa) which aims at being the "scikit-learn-like ML library for Rust". Along the way I could contribute to `linfa` by porting gaussian mixture model (`linfa-clustering/gmm`) and partial least square family methods (`linfa-pls`) confirming the fact that Python algorithms translation in Rust could be pretty straightforward.

While I did not benchmark exactly my Rust code against SMT Python one, from my debugging sessions, I noticed I did not get such a great speed up. Actually, algorithms like `doe` and `gp` relies extensively on linear algebra and Python famous libraries `numpy`/`scipy` which are strongly optimized by calling C or Fortran compiled code.

My guess at this point is that interest could come from some Rust algorithms built upon these initial building blocks hence I started to implement mixture of experts algorithm (`moe`) and on top surrogate-based optimization EGO algorithm (`ego`) which gives its name to the library[^2][^3]. Aside from performance, such library can also take advantage from the others [Rust selling points](https://www.rust-lang.org/), namely reliability and productivity. 

## Cite

If you happen to find this Rust library useful for your research, you can cite this project as follows: 

```
@Misc{egobox,
  author = {Rémi Lafage},
  title = {Egobox: efficient global optimization toolbox in Rust},
  year = {2020--},
  url = "https://github.com/relf/egobox"
}
```

[^1]: M. A. Bouhlel and J. T. Hwang and N. Bartoli and R. Lafage and J. Morlier and J. R. R. A. Martins. A Python surrogate modeling framework with derivatives. Advances in Engineering Software, 2019.

[^2]: Bartoli, Nathalie, et al. "Adaptive modeling strategy for constrained global optimization with application to aerodynamic wing design." Aerospace Science and technology 90 (2019): 85-102.

[^3]: Dubreuil, Sylvain, et al. "Towards an efficient global multidisciplinary design optimization algorithm." Structural and Multidisciplinary Optimization 62.4 (2020): 1739-1765.
