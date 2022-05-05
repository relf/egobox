Unreleased
==========

Changes
-------

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
