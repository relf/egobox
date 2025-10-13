use std::{fs::File, io::BufReader, path::Path};

use crate::Result;
use ndarray::{Array2, Zip};
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
    #[serde(rename = "BO_iterations")]
    pub bo_iterations: u64,
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

pub(crate) fn init_run_info(
    xlimits: Array2<f64>,
    config: EgorConfig,
    state: &EgorState<f64>,
) -> EgorRunData {
    let data = state.data.clone().unwrap();
    let version = env!("CARGO_PKG_VERSION");
    let name = env!("CARGO_PKG_NAME");

    let (xdata, ydata, _) = data;
    let mut sampled_locations: Vec<Sample> = vec![];
    Zip::indexed(xdata.rows())
        .and(ydata.rows())
        .for_each(|i, x, y| {
            sampled_locations.push(Sample {
                iterations: i,
                locations: x.to_vec(),
                evaluations: y[0],
            })
        });

    EgorRunData {
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
            bo_iterations: config.max_iters as u64,
            total_samples: xdata.nrows(),
            batch_size: config.q_points,
            seed: config.seed.map_or(-1, |v| v as i32),
            ..Default::default()
        },
        initial_samples: InitialSamples {
            batch_size: 1,
            sampled_locations,
        },
        extra_info: ExtraInfo {
            team_notes: format!("Native configuration info: {:?}", config),
            code_reference: format!("{name} {version}"),
            ..ExtraInfo::default()
        },
        ..EgorRunData::default()
    }
}

pub(crate) fn update_run_info(
    run_data: &mut EgorRunData,
    n_iter: u64,
    xdata: &Array2<f64>,
    ydata: &Array2<f64>,
) {
    let mut sampled_locations: Vec<SampleNoIter> = vec![];
    Zip::from(xdata.rows()).and(ydata.rows()).for_each(|x, y| {
        sampled_locations.push(SampleNoIter {
            locations: x.to_vec(),
            evaluations: y[0],
        })
    });

    run_data.search_iterations.push(SearchIteration {
        iterations: run_data.search_iterations.len() as u64 + 1,
        batch_size: xdata.nrows(),
        sampled_locations,
    });

    run_data.algorithm_parameters.bo_iterations = n_iter;
}

pub(crate) fn save_run<P: AsRef<Path>>(path: P, run_data: &EgorRunData) -> Result<()> {
    let out_json = serde_json::to_string_pretty(run_data)?;
    std::fs::write(path, out_json)?;

    Ok(())
}

pub(crate) fn load_run<P: AsRef<Path>>(path: P) -> Result<EgorRunData> {
    // Open the file in read-only mode with buffer.
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `User`.
    let run_data = serde_json::from_reader(reader)?;

    // Return the `User`.
    Ok(run_data)
}
