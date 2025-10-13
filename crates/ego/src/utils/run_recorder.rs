use std::path::Path;

use crate::Result;
use ndarray::{Array2, Zip, s};
use serde::{Deserialize, Serialize};

use crate::{EgorConfig, EgorState};
use egobox_moe::CorrelationSpec;

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ProblemMetadata {
    pub test_function: String,
    pub dimensionality: usize,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub replication_number: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct AlgorithmParameters {
    pub acquisition_function: String,
    pub kernel: String,
    pub initial_samples: usize,
    pub BO_iterations: usize,
    pub total_samples: usize,
    pub batch_size: usize,
    pub other_params: serde_json::Value,
    pub seed: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ExtraInfo {
    pub team_notes: String,
    pub code_reference: String,
    pub other_files: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Sample {
    pub iterations: usize,
    pub locations: Vec<f64>,
    pub evaluations: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct InitialSamples {
    pub batch_size: usize,
    pub sampled_locations: Vec<Sample>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct SearchIteration {
    pub iterations: u64,
    pub batch_size: usize,
    pub sampled_locations: Vec<SampleNoIter>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct SampleNoIter {
    pub locations: Vec<f64>,
    pub evaluations: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct EgorRunData {
    pub problem_metadata: ProblemMetadata,
    pub algorithm_parameters: AlgorithmParameters,
    pub extra_info: ExtraInfo,
    pub initial_samples: InitialSamples,
    pub search_iterations: Vec<SearchIteration>,
}

pub(crate) fn get_run_info(
    xlimits: Array2<f64>,
    config: EgorConfig,
    state: &EgorState<f64>,
) -> EgorRunData {
    let data = state.data.clone().unwrap();
    let version = env!("CARGO_PKG_VERSION");
    let name = env!("CARGO_PKG_NAME");
    let doe_size = state.doe_size;

    let run_data = EgorRunData {
        problem_metadata: ProblemMetadata {
            dimensionality: xlimits.nrows(),
            lower_bounds: xlimits.column(0).to_vec(),
            upper_bounds: xlimits.column(1).to_vec(),
            ..Default::default()
        },
        algorithm_parameters: AlgorithmParameters {
            acquisition_function: config.infill_criterion.name().to_string(),
            kernel: match config.gp.correlation_spec {
                CorrelationSpec::ABSOLUTEEXPONENTIAL => "Absolute Exponential".to_string(),
                CorrelationSpec::SQUAREDEXPONENTIAL => "Squared Exponential".to_string(),
                CorrelationSpec::MATERN32 => "Matern 3/2".to_string(),
                CorrelationSpec::MATERN52 => "Matern 5/2".to_string(),
                _ => "Mixed".to_string(),
            },
            initial_samples: state.doe_size,
            BO_iterations: config.max_iters,
            total_samples: data.0.nrows(),
            batch_size: config.q_points,
            seed: config.seed.map_or(-1, |v| v as i32),
            ..Default::default()
        },
        extra_info: ExtraInfo {
            team_notes: format!("Native configuration info: {:?}", config),
            code_reference: format!("{name} {version}"),
            ..ExtraInfo::default()
        },
        ..EgorRunData::default()
    };

    let (xdata, ydata, _) = data;
    let xdata = xdata.slice(s![doe_size.., ..]);
    let ydata = ydata.slice(s![doe_size.., ..]);

    let mut search_iters = run_data.search_iterations.clone();

    let mut sampled_locations: Vec<SampleNoIter> = vec![];
    Zip::from(xdata.rows()).and(ydata.rows()).for_each(|x, y| {
        sampled_locations.push(SampleNoIter {
            locations: x.to_vec(),
            evaluations: y[0],
        })
    });

    let iter = search_iters.push(SearchIteration {
        iterations: state.iter,
        batch_size: xdata.nrows(),
        sampled_locations,
    });

    EgorRunData {
        search_iterations: search_iters,
        ..run_data
    }
}

/// Save models in a bincode file
pub(crate) fn save_run<P: AsRef<Path>>(path: P, run_data: &EgorRunData) -> Result<()> {
    let out_json = serde_json::to_string_pretty(run_data)?;
    std::fs::write(path, out_json)?;

    Ok(())
}
