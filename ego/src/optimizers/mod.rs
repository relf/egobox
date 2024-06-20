//! Optimizers used internally to optimize the infill criterion

mod lhs_optimizer;
mod optimizer;

pub(crate) use lhs_optimizer::*;
pub(crate) use optimizer::*;
