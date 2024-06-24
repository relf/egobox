# egobox

[![tests](https://github.com/relf/egobox/workflows/tests/badge.svg)](https://github.com/relf/egobox/actions?query=workflow%3Atests)
[![pytests](https://github.com/relf/egobox/workflows/pytests/badge.svg)](https://github.com/relf/egobox/actions?query=workflow%3Apytests)
[![linting](https://github.com/relf/egobox/workflows/lint/badge.svg)](https://github.com/relf/egobox/actions?query=workflow%3Alint)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04737/status.svg)](https://doi.org/10.21105/joss.04737)

Rust toolbox for Efficient Global Optimization algorithms inspired from [SMT](https://github.com/SMTorg/smt).

`egobox` is twofold:

1. for end-users: [a Python module](#the-python-module), the Python binding of the optimizer named `Egor` and the surrogate model `Gpx`, mixture of Gaussian processes, written in Rust.
2. for developers: [a set of Rust libraries](#the-rust-libraries) useful to implement bayesian optimization (EGO-like) algorithms,

## The Python module

Thanks to the [PyO3 project](https://pyo3.rs), which makes Rust well suited for building Python extensions.

### Installation

```bash
pip install egobox
```

### Egor optimizer

```python
import numpy as np
import egobox as egx

# Objective function
def f_obj(x: np.ndarray) -> np.ndarray:
    return (x - 3.5) * np.sin((x - 3.5) / (np.pi))

# Minimize f_opt in [0, 25]
res = egx.Egor(egx.to_specs([[0.0, 25.0]]), seed=42).minimize(f_obj, max_iters=20)
print(f"Optimization f={res.y_opt} at {res.x_opt}")  # Optimization f=[-15.12510323] at [18.93525454]
```

### Gpx surrogate model

```python
import numpy as np
import egobox as egx

# Training
xtrain = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]).T
ytrain = np.array([[0.0, 1.0, 1.5, 0.9, 1.0]]).T
gpx = egx.Gpx.builder().fit(xtrain, ytrain)

# Prediction
xtest = np.linspace(0, 4, 20).reshape((-1, 1))
ytest = gpx.predict(xtest)
```

See the [tutorial notebooks](https://github.com/relf/egobox/tree/master/doc/README.md) and [examples folder](https://github.com/relf/egobox/tree/d9db0248199558f23d966796737d7ffa8f5de589/python/egobox/examples) for more information on the usage of the optimizer and mixture of Gaussian processes surrogate model.

## The Rust libraries

`egobox` Rust libraries consists of the following sub-packages.

| Name                                                  | Version                                                                                         | Documentation                                                               | Description                                                                               |
| :---------------------------------------------------- | :---------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- |
| [doe](https://github.com/relf/egobox/tree/master/doe) | [![crates.io](https://img.shields.io/crates/v/egobox-doe)](https://crates.io/crates/egobox-doe) | [![docs](https://docs.rs/egobox-doe/badge.svg)](https://docs.rs/egobox-doe) | sampling methods; contains LHS, FullFactorial, Random methods                             |
| [gp](https://github.com/relf/egobox/tree/master/gp)   | [![crates.io](https://img.shields.io/crates/v/egobox-gp)](https://crates.io/crates/egobox-gp)   | [![docs](https://docs.rs/egobox-gp/badge.svg)](https://docs.rs/egobox-gp)   | gaussian process regression; contains Kriging, PLS dimension reduction and sparse methods |
| [moe](https://github.com/relf/egobox/tree/master/moe) | [![crates.io](https://img.shields.io/crates/v/egobox-moe)](https://crates.io/crates/egobox-moe) | [![docs](https://docs.rs/egobox-moe/badge.svg)](https://docs.rs/egobox-moe) | mixture of experts using GP models                                                        |
| [ego](https://github.com/relf/egobox/tree/master/ego) | [![crates.io](https://img.shields.io/crates/v/egobox-ego)](https://crates.io/crates/egobox-ego) | [![docs](https://docs.rs/egobox-ego/badge.svg)](https://docs.rs/egobox-ego) | efficient global optimization with constraints and mixed integer handling                 |

### Usage

Depending on the sub-packages you want to use, you have to add following declarations to your `Cargo.toml`

```text
[dependencies]
egobox-doe = { version = "0.20" }
egobox-gp  = { version = "0.20" }
egobox-moe = { version = "0.20" }
egobox-ego = { version = "0.20" }
```

### Features

The table below presents the various features available depending on the subcrate

| Name         | doe  | gp   | moe  | ego  |
| :----------- | :--- | :--- | :--- | :--- |
| serializable | ✔️    | ✔️    | ✔️    |      |
| persistent   |      |      | ✔️    | ✔️(*) |
| blas         |      | ✔️    | ✔️    | ✔️    |
| nlopt        |      | ✔️    |      | ✔️    |

(*) required for mixed-variable gaussian process

#### serializable

When selected, the serialization with [serde crate](https://serde.rs/) is enabled.

#### persistent

When selected, the save and load as a json file with [serde_json crate](https://serde.rs/) is enabled.

#### blas

When selected, the usage of BLAS/LAPACK backend is possible, see [below](#blaslapack-backend-optional) for more information.

#### nlopt

When selected, the [nlopt crate](https://github.com/adwhit/rust-nlopt) is used to provide optimizer implementations (ie Cobyla, Slsqp)

### Examples

Examples (in `examples/` sub-packages folder) are run as follows:

```bash
cd doe && cargo run --example samplings --release
```

``` bash
cd gp && cargo run --example kriging --release
```

``` bash
cd moe && cargo run --example clustering --release
```

``` bash
cd ego && cargo run --example ackley --release
```

### BLAS/LAPACK backend (optional)

`egobox` relies on [linfa](https://github.com/rust-ml/linfa) project for methods like clustering and dimension reduction, but also try to adopt as far as possible the same [coding structures](https://github.com/rust-ml/linfa/blob/master/CONTRIBUTE.md).

As for `linfa`, the linear algebra routines used in `gp`, `moe` ad `ego` are provided by the pure-Rust [linfa-linalg](https://github.com/rust-ml/linfa-linalg) crate, the default linear algebra provider.

Otherwise, you can choose an external BLAS/LAPACK backend available through the [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg) crate. In this case, you have to specify the `blas` feature and a `linfa` [BLAS/LAPACK backend feature](https://github.com/rust-ml/linfa#blaslapack-backend) (more information in [linfa features](https://github.com/rust-ml/linfa#blaslapack-backend)).

Thus, for instance, to use `gp` with the Intel MKL BLAS/LAPACK backend, you could specify in your `Cargo.toml` the following features:

```text
[dependencies]
egobox-gp = { version = "0.20", features = ["blas", "linfa/intel-mkl-static"] }
```

or you could run the `gp` example as follows:

``` bash
cd gp && cargo run --example kriging --release --features blas,linfa/intel-mkl-static
```

## Citation

[![DOI](https://joss.theoj.org/papers/10.21105/joss.04737/status.svg)](https://doi.org/10.21105/joss.04737)

If you find this project useful for your research, you may cite it as follows:

```text
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
