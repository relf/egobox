use crate::errors::Result;
#[cfg(feature = "persistent")]
use std::fs;
#[cfg(feature = "persistent")]
use std::io::Write;
use std::path::Path;

/// Save models in a bincode file
pub(crate) fn save_gp_models<P: AsRef<Path>>(
    path: P,
    models: &[Box<dyn egobox_moe::MixtureGpSurrogate>],
) -> Result<()> {
    let mut file = fs::File::create(path).unwrap();

    let bytes = bincode::serde::encode_to_vec(models, bincode::config::standard())?;
    file.write_all(&bytes)?;

    Ok(())
}
