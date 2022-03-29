# Egobox

[![tests](https://github.com/relf/egobox/workflows/tests/badge.svg)](https://github.com/relf/egobox/actions?query=workflow%3Atests)

Toolbox for Efficient Global Optimization algorithms written in Rust inspired from [SMT](https://github.com/SMTorg/smt). This library provides the port of the following algorithms:
* DOE sampling methods: LHS, FullFactorial, Randomized
* GP regression: Kriging and KPLS surrogates
* MOE: Mixture of experts using kriging models
* EGO: Efficient Global Optimization 

Thanks to the [PyO3 project](https://pyo3.rs), which makes Rust well suited for building Python extensions, the EGO algorithm written in Rust (aka Egor) is binded in Python. You can install the Python package using:

```bash
pip install egobox
```

See the [tutorial notebook](doc/TutorialEgor.ipynb).

## Why Egobox?

I started this library as a way to learn Rust and see if it can be used to implement algorithms like those in the SMT toolbox[^1]. As the first components (doe, gp) emerged, it appeears I could translate Python code almost line by line in Rust (well... after great deal of borrow-checker fight!) and thanks to [Rust ndarray library ecosystem](https://github.com/rust-ndarray). 

This library relies also on the [linfa project](https://github.com/rust-ml/linfa) which aims at being the "scikit-learn project for Rust". Along the way I could contribute to linfa by porting gaussian mixture model (`linfa-clustering/gmm`) and partial least square family methods (`linfa-pls`) confirming the fact that Python algorithms translation in Rust could be pretty straightforward.

While I did not benchmark my Rust code against SMT Python one, from my debugging sessions I notice I do not get a great speed up. The point is that actually I do not compare Rust vs Python but vs Python/C/Fortran. Algorithms in `doe` and `gp` relies widely on linear algebra and killer libraries numpy/scipy which are strongly optimized.

My guess here was that interest for this code could come from Rust algorithms built upon these initial building blocks hence I started to implement mixture of experts algorithm (moe) and on top bayesian optimization EGO algorithm which gives its name to the library[^2].

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
