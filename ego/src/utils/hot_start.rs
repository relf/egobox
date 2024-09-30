pub use argmin::core::checkpointing::{Checkpoint, CheckpointingFrequency};
use argmin::core::Error;
use serde::{de::DeserializeOwned, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use crate::EgorState;

/// Handles saving a checkpoint to disk as a binary file.
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub struct HotStartCheckpoint {
    /// Extended iteration number
    pub extension_iters: u64,
    /// Indicates how often a checkpoint is created
    pub frequency: CheckpointingFrequency,
    /// Directory where the checkpoints are saved to
    pub directory: PathBuf,
    /// Name of the checkpoint files
    pub filename: PathBuf,
}

impl Default for HotStartCheckpoint {
    /// Create a default `HotStartCheckpoint` instance.
    fn default() -> HotStartCheckpoint {
        HotStartCheckpoint {
            extension_iters: 0,
            frequency: CheckpointingFrequency::default(),
            directory: PathBuf::from(".checkpoints"),
            filename: PathBuf::from("egor.arg"),
        }
    }
}

impl HotStartCheckpoint {
    /// Create a new `HotStartCheckpoint` instance
    pub fn new<N: AsRef<str>>(
        directory: N,
        name: N,
        frequency: CheckpointingFrequency,
        ext_iters: u64,
    ) -> Self {
        HotStartCheckpoint {
            extension_iters: ext_iters,
            frequency,
            directory: PathBuf::from(directory.as_ref()),
            filename: PathBuf::from(format!("{}.arg", name.as_ref())),
        }
    }
}

impl<S> Checkpoint<S, EgorState<f64>> for HotStartCheckpoint
where
    S: Serialize + DeserializeOwned,
{
    /// Writes checkpoint to disk.
    ///
    /// If the directory does not exist already, it will be created. It uses `bincode` to serialize
    /// the data.
    /// It will return an error if creating the directory or file or serialization failed.
    fn save(&self, solver: &S, state: &EgorState<f64>) -> Result<(), Error> {
        if !self.directory.exists() {
            std::fs::create_dir_all(&self.directory)?
        }
        let fname = self.directory.join(&self.filename);
        let f = BufWriter::new(File::create(fname)?);
        bincode::serialize_into(f, &(solver, state))?;
        Ok(())
    }

    /// Load a checkpoint from disk.
    ///
    ///
    /// If there is no checkpoint on disk, it will return `Ok(None)`.
    /// Returns an error if opening the file or deserialization failed.
    fn load(&self) -> Result<Option<(S, EgorState<f64>)>, Error> {
        let path = &self.directory.join(&self.filename);
        if !path.exists() {
            return Ok(None);
        }
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let (solver, mut state): (_, EgorState<_>) = bincode::deserialize_from(reader)?;
        state.extend_max_iters(self.extension_iters);
        Ok(Some((solver, state)))
    }

    /// Returns the how often a checkpoint is to be saved.
    ///
    /// Used internally by [`save_cond`](`argmin::core::checkpointing::Checkpoint::save_cond`).
    fn frequency(&self) -> CheckpointingFrequency {
        self.frequency
    }
}
