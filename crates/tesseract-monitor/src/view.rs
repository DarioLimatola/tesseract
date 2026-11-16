//! What the monitor knows about one run: read from its run folder.

use std::path::{Path, PathBuf};

use tesseract_run::{
    Config, Metrics, RunInfo, Snapshot, list_snapshots, read_metrics, read_run_info,
};

/// A run being watched or replayed.
#[derive(Debug, Default)]
pub struct RunView {
    /// The run folder.
    pub dir: PathBuf,
    /// `run.json`, when readable.
    pub info: Option<RunInfo>,
    /// The run's config, when readable.
    pub config: Option<Config>,
    /// The metrics lines so far.
    pub metrics: Vec<Metrics>,
    /// Snapshot files with their steps.
    pub snapshots: Vec<(u64, PathBuf)>,
    /// Index of the snapshot shown.
    pub index: usize,
    /// When true the newest snapshot is shown as soon as it appears.
    pub follow: bool,
    /// The snapshot shown.
    pub snapshot: Option<Snapshot>,
    loaded: Option<PathBuf>,
    /// The last problem reading the folder, if any.
    pub problem: Option<String>,
}

impl RunView {
    /// Opens a run folder and reads everything in it.
    pub fn open(dir: &Path) -> Self {
        let mut view = Self {
            dir: dir.to_path_buf(),
            follow: true,
            ..Self::default()
        };
        view.config = Config::load(&dir.join("config.toml"), &[]).ok();
        match read_metrics(dir) {
            Ok(m) => view.metrics = m,
            Err(e) => view.problem = Some(e.to_string()),
        }
        view.refresh();
        view
    }

    /// A run that is about to start in `dir`: nothing to read yet.
    pub fn starting(dir: &Path, config: Option<Config>) -> Self {
        Self {
            dir: dir.to_path_buf(),
            follow: true,
            config,
            ..Self::default()
        }
    }

    /// Reads `run.json` and the list of snapshots again, and loads the
    /// snapshot to show.
    pub fn refresh(&mut self) {
        if let Ok(info) = read_run_info(&self.dir) {
            self.info = Some(info);
        }
        self.snapshots = list_snapshots(&self.dir);
        if self.follow || self.index >= self.snapshots.len() {
            self.index = self.snapshots.len().saturating_sub(1);
        }
        self.load_selected();
    }

    /// Shows the snapshot at `index`.
    pub fn select(&mut self, index: usize) {
        self.index = index.min(self.snapshots.len().saturating_sub(1));
        self.load_selected();
    }

    fn load_selected(&mut self) {
        let Some((_, path)) = self.snapshots.get(self.index) else {
            return;
        };
        if self.loaded.as_ref() == Some(path) {
            return;
        }
        match Snapshot::load(path) {
            Ok(s) => {
                self.snapshot = Some(s);
                self.loaded = Some(path.clone());
                self.problem = None;
            }
            // A file can be missing for a moment; keep what is shown.
            Err(e) => self.problem = Some(e.to_string()),
        }
    }

    /// Steps of the budget, when the config is known.
    pub fn budget_steps(&self) -> Option<u64> {
        self.config.as_ref().map(|c| c.budget.steps)
    }

    /// A short label: `name / time`.
    pub fn label(dir: &Path) -> String {
        let part = |p: Option<&Path>| {
            p.and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        format!("{} / {}", part(dir.parent()), part(Some(dir)))
    }
}
