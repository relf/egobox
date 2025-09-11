use anyhow::Result;
use egobox_moe::MixtureGpSurrogate;
use std::env;
use std::fs;

fn main() -> Result<()> {
    let current_dir = env::current_dir()?;
    println!("Current working directory: {}", current_dir.display());

    for f in [
        egobox_ego::EGOR_INITIAL_GP_FILENAME,
        egobox_ego::EGOR_GP_FILENAME,
    ] {
        let data: Vec<u8> = fs::read(f)?;
        let gp_models: Vec<Box<dyn MixtureGpSurrogate>> = bincode::deserialize(&data[..])?;

        for gp in gp_models {
            println!(
                "training data = {} points",
                gp.as_ref().training_data().0.nrows()
            );
            println!("loocv = {}", gp.as_ref().loocv())
        }
    }

    for f in [
        egobox_ego::EGOR_INITIAL_GP_FILENAME,
        egobox_ego::EGOR_GP_FILENAME,
    ] {
        let data: Vec<u8> = fs::read(f)?;
        let gp_models: Vec<Box<dyn MixtureGpSurrogate>> = bincode::deserialize(&data[..])?;

        for gp in gp_models {
            println!("cv = {}", gp.as_ref().cv(2))
        }
    }

    Ok(())
}
