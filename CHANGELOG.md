Changes
-------

Version 0.9.0 - unreleased
==========================

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
