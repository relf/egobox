Changes
-------

Version 0.14.0 - unreleased
===========================

Version 0.13.0 - 30/11/2023
===========================

* `ego`: API refactoring to enable `ask-and-tell` interface
  * Configuration of Egor is factorize out in `EgorConfig`
  * `EgorBuilder` gets a `configure` method to tune the configuration
  * `EgorService` structure represent `Egor` when used as service 
  * Python `Egor` API changes:
    * function under optimization is now given via `minimize(fun, max_iters=...)` method
    * new method `suggest(xdoe, ydoe)` allows to ask for x suggestion and tell current function evaluations 
    * new method `get_result(xdoe, ydoe)` to get the best evaluation (ie minimum) from given ones

Version 0.12.0 - 10/11/2023
===========================

* `gp` uses pure Rust COBYLA by @relf in https://github.com/relf/egobox/pull/110,https://github.com/relf/egobox/pull/113
* `ego` as pure Rust implementation (`nlopt` is now optional) by @relf in https://github.com/relf/egobox/pull/112
* `egobox` Python module: Simplify mixed-integer type declaration by @relf in https://github.com/relf/egobox/pull/115
* Upgrade dependencies by @relf in https://github.com/relf/egobox/pull/114
* Upgrade edition 2021 by @relf in https://github.com/relf/egobox/pull/109
* CI maintainance by @relf in https://github.com/relf/egobox/pull/111
* Bump actions/checkout from 2 to 4 by @dependabot in https://github.com/relf/egobox/pull/107
* Bump actions/setup-python from 2 to 4 by @dependabot in https://github.com/relf/egobox/pull/108

Version 0.11.0 - 20/09/2023
===========================

* Automate Python package build and upload on Pypi from Github CI by @relf in https://github.com/relf/egobox/pull/104
* Fix FullFactorial when asked nb iof samples is small wrt x dimension by @relf in https://github.com/relf/egobox/pull/105
* Make mixed-integer sampling methods available in Python by @relf in https://github.com/relf/egobox/pull/106

Version 0.10.0 - 22/06/2023
===========================

* `gp`, `moe` and `egobox` Python module:
  * Added Gaussian process sampling (#97) 
  * Added string representation (#98)

* `egobox` Python module:
  * Change recombination enum to respect Python uppercase convention (#98)

* Notebooks and documentation updates (#97, #98, #99)

Version 0.9.0 - 02/06/2023
==========================

* `ego`:
    * Infill criterion is now a trait object in `EgorSolver` structure (#92)
    * `Egor` and `EgorSolver` API: methods taking argument of type Option<T> now take argument of type T (#94)
    * `EgorBuilder::min_within_mixed_space()` is now `EgorBuilder::min_within_mixint_space()` (#96)
    * `egobox-ego` library doc updated (#95)

* `egobox` Python module: Upgrade to PyO3 0.18 (#91)

Version 0.8.2 - 31/03/2023
==========================

* `ego`:
   * Fix Egor solver best iter computation (#89)

Version 0.8.1 - 28/03/2023
==========================

* `ego`:
  * Make objective and constraints training in parallel (#86)
  * Lock mopta execution to allow concurrent computations (#84)
  * Fix and adjust infill criterion optimmization retries strategy (#87)
* `moe`:
  * Fix k-fold cross-validation (#85)  

Version 0.8.0 - 10/03/2023
==========================

* `ego`:
  * Renaming `XType`, `XSpec` for consistency (#82)
  * Export history in optimization result (#81)
  * Use nb iter instead of nb eval, rename q_parallel as q_points (#79)
  * Warn when inf or nan detected during obj scaling computation (#78)
  * Parallelize constraint scales computations (#73)
  * Parallelize multistart optimizations (#76)
  * Handle GMM errors during MOE training (#75)
  * Handle possible errors from GMM clustering (#74)
  * Upgrade argmin 0.8.0 (#72)
  * Add mopta08 test case as example (#71)
  * Fix scaling check for infinity (#70)
  * Use kriging surrogate by default (#69)

Version 0.7.0 - 11/01/2023
==========================

* `gp`: 
  * Add analytic derivatives computations (#54, #55, #56, #58, #60). All derivatives available for all mean/correlation models are implemented.
  * Refactor `MeanModel` and `CorrelationModel` methods:
    * `apply()` renamed to `value()`
    * `jac()` renamed to `jacobian()`
  * Fix prediction computation when using linear regression (#52)
* `ego`:
  * Refactor `Egor` using [`argmin 0.7.0` solver framework](http://argmin-rs.org) `EgorSolver` can be used with `argmin::Executor` and benefit from observers and checkpointing features (#67) 
  * `Egor` use kriging setting by default (i.e. one cluster with constant mean and squared exponential correlation model) 
* Add [notebook on Manuau Loa CO2 example](https://github.com/relf/egobox/blob/master/doc/Gpx_MaunaLoaCO2.ipynb) to show `GpMix`/`Gpx` surrogate model usage (#62)
* Use xoshiro instead of isaac random generator (#63)
* Upgrade `ndarray 0.15`, `linfa 0.6.1`, `PyO3 0.17` (#57, #64)

Version 0.6.0 - 2022-11-14
==========================

* `gp`: Kriging derivatives predictions are implemented (#44, #45), derivatives for Gp with linear regression are implemented (#47)
  * `predict_derivatives`: prediction of the output derivatives y wtr the input x
  * `predict_variance_derivatives`: prediction of the derivatives of the output variance wrt the input x
* `moe`: as for `gp`, derivatives methods for smooth and hard predictions are implemented  (#46)
* `ego`: when available derivatives are used to optimize the infill criterion with slsqp (#44)  
* `egobox` Python binding: add `GpMix`/`Gpx` in Python `egobox` module, the Python binding of `egobox-moe::Moe` (#31)

Version 0.5.0 - 2022-10-07
==========================

* Add Egor `minimize` interruption capability (Ctrl+C) from Python (#30)
* Minor performance improvement in moe clustering (#29)
* Improvements following JOSS submission review (#34, #36, #38, #39, #40, #42)

Version 0.4.0 - 2022-07-09
==========================

* Generate Python `egobox` module for Linux (#20)
* Improve `Egor` robustness by adding LHS optimization (#21)
* Improve `moe` with automatic number of clusters determination (#22) 
* Use `linfa 0.6.0` making BLAS dependency optional (#23)
* Improve `Egor` by implementing automatic reclustering every 10-points addition (#25)
* Fix `Egor` parallel infill strategy (qEI): bad objectives and constraints gp models updste (#26)

Version 0.3.0 - 2022-05-05
==========================

Improve mixture of experts (#15)

* Implement moe save/load (feature persistent)
* Rename GpSurrogate to Surrogate
* Remove `fit_for_predict`
* Implement `ParamGuard` for `MoeParams`
* Implement `Fit` for `MoeParams`
* Rename `MoeParams` setters
    
Refactor `moe`/`ego` relation (#16)

* Move `MoeFit` as `SurrogateBuilder` from `moe` to `ego`
* Implement `SurrogateBuilder` for `Moe` 
* `Moe` uses `linfa::Fit` trait
* Rename `Evaluator` as `PreProcessor`
 
Refactor `MixintEgor` (#17)

* Rename `PreProcessor::eval` to `run`
* Implement `linfa::Fit` for `MixintMoeParams`, use `linfa::Dataset`
* Rename `SurrogateParams` to `MoeBuilder`
* Rename `n_parallel` to `q_parallel` (qEI stategy)

Version 0.2.1 - 2022-04-13
==========================

* Improve documentation
* `egobox` Python module: rename egobox `Optimizer` class to `Egor`

Version 0.2.0 - 2022-03-24
==========================

* Add hot start
* Add constraint handling
* Add mixed-integer optimization capability
* Add Python binding with PyO3

Version 0.1.0 - 2021-11-18
==========================

Initial release

* `doe`: `LHS`, `FullFactorial`, `Random sampling`
* `gp`: Gaussian Process models with 3 regression models (constant, linear quadratic) and 4 correlation models (squared exponential, absolute exponential, matern32, matern52) 
* `moe`: Mixture of Experts: find the bests mix of gps given a number of clusters regarding smooth or hard recombination
* `ego`: Contains egor optimizer which is a super EGO algorithm implemented on top of the previous elements. It implements several infill strategy: EI, WB2, WB2S and use either COBYLA or SLSQP for internal optimization.
