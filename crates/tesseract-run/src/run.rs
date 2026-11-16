//! The training loop and the run folder.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tesseract_core::{ENGINE_VERSION, Model};
use tesseract_tasks::{Example, Task, is_correct};

use crate::snapshot::Snapshot;
use crate::{Config, RunError};

/// One line of `metrics.jsonl`: the state of a run at one evaluation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Learning steps done.
    pub step: u64,
    /// Score on the training stream since the last evaluation, each example
    /// answered before it was learned.
    pub train: f64,
    /// Score on the held-out set.
    pub held_out: f64,
    /// Trivial scores on the held-out set, by name.
    pub baselines: BTreeMap<String, f64>,
    /// Scores on the task's further evaluation sets, by name.
    pub extra: BTreeMap<String, f64>,
    /// Held-out score needed to pass.
    pub threshold: f64,
    /// Parameter bits changed per learning step since the last evaluation.
    pub flips_per_step: f64,
    /// Connections made dormant since the last evaluation.
    pub pruned: u64,
    /// Live connections of active units.
    pub connections_active: u64,
    /// Dormant connections.
    pub connections_dormant: u64,
    /// Removed connections (always zero in this draft).
    pub connections_removed: u64,
    /// Protected connections.
    pub connections_protected: u64,
    /// Active units.
    pub units_active: u64,
    /// Dormant units.
    pub units_dormant: u64,
    /// Free units.
    pub units_free: u64,
    /// Units recruited since the start.
    pub recruited: u64,
    /// Examples that could not be stored since the start.
    pub dropped: u64,
    /// Learning steps per second since the last evaluation.
    pub steps_per_second: f64,
    /// Forward time per token on one core, in microseconds.
    pub micros_per_token: f64,
    /// Bytes of parameters read by the forward pass.
    pub parameter_bytes: u64,
}

/// `run.json`: what is needed to reproduce and identify a run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunInfo {
    /// Name of the run.
    pub name: String,
    /// Seed of the run.
    pub seed: u64,
    /// Version of the engine.
    pub engine_version: String,
    /// Source revision (`git describe`), or `unknown`.
    pub source_revision: String,
    /// Start time, UTC.
    pub started_utc: String,
    /// Task name.
    pub task: String,
    /// Axes of the engine the task uses.
    pub axes: String,
    /// `running`, `finished` or `stopped`.
    pub status: String,
    /// Learning steps done.
    pub steps_done: u64,
    /// Last held-out score.
    pub held_out: Option<f64>,
    /// Held-out score needed to pass.
    pub threshold: f64,
    /// Whether the last held-out score reached the threshold.
    pub passed: Option<bool>,
}

/// The result of a run.
#[derive(Clone, Debug)]
pub struct RunOutcome {
    /// The run folder.
    pub dir: PathBuf,
    /// The last evaluation.
    pub last: Metrics,
    /// Whether the held-out score reached the pass threshold.
    pub passed: bool,
    /// True when the run was stopped before its budget ended.
    pub stopped: bool,
}

fn io_err(path: &Path, e: std::io::Error) -> RunError {
    RunError::Io(format!("{}: {e}", path.display()))
}

/// UTC time as `YYYYMMDD-HHMMSS`.
fn utc_stamp(now: SystemTime) -> String {
    let secs = now
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}{m:02}{d:02}-{:02}{:02}{:02}",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn source_revision() -> String {
    std::process::Command::new("git")
        .args(["describe", "--always", "--dirty"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn evaluate(model: &dyn Model, task: &dyn Task, examples: &[Example]) -> f64 {
    if examples.is_empty() {
        return 0.0;
    }
    // Counting is integer, so the result is the same for any thread count.
    let right = examples
        .par_iter()
        .filter(|e| is_correct(model, task, e))
        .count();
    right as f64 / examples.len() as f64
}

/// Forward time per token on one core, on up to 200 held-out examples.
fn time_per_token(model: &dyn Model, examples: &[Example]) -> f64 {
    let sample = &examples[..examples.len().min(200)];
    let tokens: usize = sample.iter().map(|e| e.context.len()).sum();
    if tokens == 0 {
        return 0.0;
    }
    let start = Instant::now();
    for e in sample {
        std::hint::black_box(model.forward(&e.context));
    }
    start.elapsed().as_secs_f64() * 1e6 / tokens as f64
}

fn write_info(dir: &Path, info: &RunInfo) -> Result<(), RunError> {
    let path = dir.join("run.json");
    let text = serde_json::to_string_pretty(info).map_err(|e| RunError::Format(e.to_string()))?;
    fs::write(&path, text).map_err(|e| io_err(&path, e))
}

/// Runs a config to the end of its budget, or until `stop` becomes true.
///
/// Creates the run folder under `out_root` and calls `on_metrics` after
/// every evaluation.
pub fn run(
    config: &Config,
    out_root: &Path,
    stop: &AtomicBool,
    on_metrics: &mut dyn FnMut(&Metrics),
) -> Result<RunOutcome, RunError> {
    let mut task = config.build_task();
    let mut model = config.build_model();
    let policy = config.structure.policy();
    let target_mask = task.target_mask();
    let threshold = task.pass_threshold();
    let baselines: BTreeMap<String, f64> = task
        .baselines()
        .into_iter()
        .map(|b| (b.name, b.score))
        .collect();

    // Run folder.
    let stamp = utc_stamp(SystemTime::now());
    let parent = out_root.join(&config.name);
    let mut dir = parent.join(&stamp);
    let mut n = 2;
    while dir.exists() {
        dir = parent.join(format!("{stamp}-{n}"));
        n += 1;
    }
    let snapshots = dir.join("snapshots");
    fs::create_dir_all(&snapshots).map_err(|e| io_err(&snapshots, e))?;
    let config_path = dir.join("config.toml");
    fs::write(&config_path, config.to_toml()?).map_err(|e| io_err(&config_path, e))?;
    let mut info = RunInfo {
        name: config.name.clone(),
        seed: config.seed,
        engine_version: ENGINE_VERSION.to_string(),
        source_revision: source_revision(),
        started_utc: stamp,
        task: task.name().to_string(),
        axes: task.axes().to_string(),
        status: "running".to_string(),
        steps_done: 0,
        held_out: None,
        threshold,
        passed: None,
    };
    write_info(&dir, &info)?;
    let metrics_path = dir.join("metrics.jsonl");
    let mut log = File::create(&metrics_path).map_err(|e| io_err(&metrics_path, e))?;

    let mut measure = |model: &dyn Model,
                       task: &dyn Task,
                       step: u64,
                       window: &Window,
                       totals: &Totals|
     -> Result<Metrics, RunError> {
        let census = model.census();
        let m = Metrics {
            step,
            train: window.train_score(),
            held_out: evaluate(model, task, task.held_out()),
            baselines: baselines.clone(),
            extra: task
                .extra_sets()
                .iter()
                .map(|(name, set)| (name.clone(), evaluate(model, task, set)))
                .collect(),
            threshold,
            flips_per_step: window.per_step(window.flips),
            pruned: window.pruned,
            connections_active: census.connections_active,
            connections_dormant: census.connections_dormant,
            connections_removed: census.connections_removed,
            connections_protected: census.connections_protected,
            units_active: census.units_active,
            units_dormant: census.units_dormant,
            units_free: census.units_free,
            recruited: totals.recruited,
            dropped: totals.dropped,
            steps_per_second: window.steps_per_second(),
            micros_per_token: time_per_token(model, task.held_out()),
            parameter_bytes: model.parameter_bytes() as u64,
        };
        let line = serde_json::to_string(&m).map_err(|e| RunError::Format(e.to_string()))?;
        writeln!(log, "{line}").map_err(|e| io_err(&metrics_path, e))?;
        log.flush().map_err(|e| io_err(&metrics_path, e))?;
        Ok(m)
    };

    let save = |model: &dyn Model, step: u64| -> Result<(), RunError> {
        Snapshot::from_views(step, &model.views())
            .save(&snapshots.join(format!("step-{step:09}.tsnp")))
    };

    let mut totals = Totals::default();
    let mut window = Window::default();
    let mut last = measure(model.as_ref(), task.as_ref(), 0, &window, &totals)?;
    on_metrics(&last);
    save(model.as_ref(), 0)?;

    // Error between structure steps, to know whether it is rising.
    let (mut period_right, mut period_seen, mut previous_error) = (0u64, 0u64, f64::INFINITY);
    let mut stopped = false;
    let mut step = 0;
    let mut clock = Instant::now();
    while step < config.budget.steps {
        if stop.load(Ordering::Relaxed) {
            stopped = true;
            break;
        }
        step += 1;
        let example = task.next_example();
        let right = is_correct(model.as_ref(), task.as_ref(), &example);
        let report = model.learn(&example.context, &example.target, &target_mask);
        window.steps += 1;
        window.right += u64::from(right);
        window.flips += u64::from(report.changes);
        totals.recruited += u64::from(report.recruited);
        totals.dropped += u64::from(report.dropped);
        period_seen += 1;
        period_right += u64::from(right);

        if config.structure.period > 0 && step % config.structure.period == 0 {
            let error = 1.0 - period_right as f64 / period_seen.max(1) as f64;
            let s = model.structure_step(&policy, error > previous_error);
            previous_error = error;
            (period_right, period_seen) = (0, 0);
            let pruned = u64::from(s.connections_pruned) + u64::from(s.units_pruned);
            window.pruned += pruned;
            window.flips += pruned;
        }

        let last_step = step == config.budget.steps;
        if step % config.budget.eval_every == 0 || last_step {
            window.learning += clock.elapsed();
            last = measure(model.as_ref(), task.as_ref(), step, &window, &totals)?;
            on_metrics(&last);
            window = Window::default();
            clock = Instant::now();
        }
        if config.budget.snapshot_every > 0 && step % config.budget.snapshot_every == 0 && !last_step
        {
            window.learning += clock.elapsed();
            save(model.as_ref(), step)?;
            clock = Instant::now();
        }
    }
    if stopped && last.step != step {
        window.learning += clock.elapsed();
        last = measure(model.as_ref(), task.as_ref(), step, &window, &totals)?;
        on_metrics(&last);
    }
    save(model.as_ref(), step)?;

    let passed = last.held_out >= threshold;
    info.status = if stopped { "stopped" } else { "finished" }.to_string();
    info.steps_done = step;
    info.held_out = Some(last.held_out);
    info.passed = Some(passed);
    write_info(&dir, &info)?;
    Ok(RunOutcome {
        dir,
        last,
        passed,
        stopped,
    })
}

#[derive(Default)]
struct Totals {
    recruited: u64,
    dropped: u64,
}

#[derive(Default)]
struct Window {
    steps: u64,
    right: u64,
    flips: u64,
    pruned: u64,
    learning: Duration,
}

impl Window {
    fn train_score(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.right as f64 / self.steps as f64
        }
    }

    fn per_step(&self, count: u64) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            count as f64 / self.steps as f64
        }
    }

    fn steps_per_second(&self) -> f64 {
        let s = self.learning.as_secs_f64();
        if s > 0.0 { self.steps as f64 / s } else { 0.0 }
    }
}

/// Reads the metrics log of a run folder. Lines that do not parse (for
/// example a line still being written) are skipped.
pub fn read_metrics(dir: &Path) -> Result<Vec<Metrics>, RunError> {
    let path = dir.join("metrics.jsonl");
    let file = File::open(&path).map_err(|e| io_err(&path, e))?;
    Ok(BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect())
}

/// Reads `run.json` of a run folder.
pub fn read_run_info(dir: &Path) -> Result<RunInfo, RunError> {
    let path = dir.join("run.json");
    let text = fs::read_to_string(&path).map_err(|e| io_err(&path, e))?;
    serde_json::from_str(&text).map_err(|e| RunError::Format(format!("{}: {e}", path.display())))
}

/// Snapshot files of a run folder with their steps, in step order.
pub fn list_snapshots(dir: &Path) -> Vec<(u64, PathBuf)> {
    let mut out: Vec<(u64, PathBuf)> = fs::read_dir(dir.join("snapshots"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let step = path
                .file_name()?
                .to_str()?
                .strip_prefix("step-")?
                .strip_suffix(".tsnp")?
                .parse()
                .ok()?;
            Some((step, path))
        })
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_stamp_known_dates() {
        let at = |s: u64| utc_stamp(UNIX_EPOCH + Duration::from_secs(s));
        assert_eq!(at(0), "19700101-000000");
        assert_eq!(at(951_782_400), "20000229-000000");
        assert_eq!(at(1794787199), "20261115-235959");
    }
}
