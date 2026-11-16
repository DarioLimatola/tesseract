//! The runner: one command on one config file, one run folder.
//!
//! A run folder holds:
//!
//! - `config.toml`: the exact config of the run, overrides applied;
//! - `run.json`: seed, engine version, source revision, status, result;
//! - `metrics.jsonl`: one line per evaluation;
//! - `snapshots/step-N.tsnp`: state snapshots for the neuron view.
//!
//! The same code runs from the command line (`tesseract run`) and inside
//! the desktop monitor.

pub mod config;
pub mod run;
pub mod snapshot;

pub use config::Config;
pub use run::{Metrics, RunInfo, RunOutcome, list_snapshots, read_metrics, read_run_info, run};
pub use snapshot::Snapshot;

/// Errors of the runner.
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    /// The config is wrong.
    #[error("config: {0}")]
    Config(String),
    /// A file could not be read or written.
    #[error("file: {0}")]
    Io(String),
    /// A file has the wrong format.
    #[error("format: {0}")]
    Format(String),
}
