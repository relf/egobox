mod bounds;
mod cstr_pof;
mod find_result;
mod hot_start;
mod logei_helper;
mod misc;
mod sort_axis;
mod start_points;

pub use bounds::*;
pub use cstr_pof::*;
pub use find_result::*;
pub use hot_start::*;
pub use logei_helper::*;
pub use misc::*;
pub use start_points::*;

/// Env variable to enable logging feature
pub const EGOBOX_LOG: &str = "EGOBOX_LOG";

/// Env variable to enable the use of PoF as criterion while no feasible point is found
pub const EGOBOX_USE_MAX_PROBA_OF_FEASIBILITY: &str = "EGOBOX_USE_MAX_PROBA_OF_FEASIBILITY";
