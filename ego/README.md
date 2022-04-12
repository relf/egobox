# Efficient global optimization 

[![crates.io](https://img.shields.io/crates/v/egobox-ego)](https://crates.io/crates/egobox-ego)
[![docs](https://docs.rs/egobox-ego/badge.svg)](https://docs.rs/egobox-ego)

`egobox-ego` provides a Rust implementation of efficient global optimization algorithm.
It is a Rust port of EGO of the [SMT](https://smt.readthedocs.io) Python library.

## The big picture

`egobox-ego` is a library crate in the top-level package [egobox](https://github.com/relf/egobox).

## Current state

`egobox-ego` currently implements EGO using `egobox-moe` with the following features:

* Mixture of gausian processes
* Infill criteria: EI, WB2, WB2S
* Basic handling of negative constraints
* Mixed integer optimization available through continuous relaxation

## Examples

There is some usage examples in the examples/ directory. To run, use:

```
$ cargo run --release --example ackley --features linfa/intel-mkl-static
```

## License

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0

