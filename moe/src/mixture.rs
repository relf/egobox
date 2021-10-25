use crate::expert::*;

struct MixtureParams {}

struct Mixture {
    experts: Vec<Box<dyn Expert>>,
}
