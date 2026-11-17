//! The monitor window: Setup, Run and Neurons.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};

use eframe::egui::{
    self, Align2, Color32, FontId, Rect, RichText, Sense, Stroke, StrokeKind, Ui, pos2, vec2,
};
use tesseract_run::snapshot::{BlockSnapshot, UnitDetail, UnitSnapshot};
use tesseract_run::{Config, Metrics, create_run_dir, list_runs, run_in};

use crate::chart::{Chart, RefLine, Series, Unit};
use crate::theme::{Theme, compact};
use crate::view::RunView;

/// What the cells of the unit grid show.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColourBy {
    /// Free, active or dormant.
    State,
    /// How often the unit fired.
    Usage,
    /// How many live inputs the unit has.
    LiveInputs,
}

enum Event {
    Metrics(Box<Metrics>),
    Done(Result<bool, String>),
}

struct LiveRun {
    stop: Arc<AtomicBool>,
    events: Receiver<Event>,
}

/// State of the monitor window.
pub struct MonitorApp {
    configs_dir: PathBuf,
    runs_dir: PathBuf,
    config_files: Vec<PathBuf>,
    selected_config: Option<usize>,
    config_text: String,
    config_check: Result<(), String>,
    past_runs: Vec<PathBuf>,
    live: Option<LiveRun>,
    view: Option<RunView>,
    selected_block: usize,
    selected_unit: Option<usize>,
    colour_by: ColourBy,
    message: Option<String>,
}

impl MonitorApp {
    /// A monitor reading configs from `configs_dir` and keeping runs in
    /// `runs_dir`.
    pub fn new(configs_dir: PathBuf, runs_dir: PathBuf) -> Self {
        let mut app = Self {
            configs_dir,
            runs_dir,
            config_files: Vec::new(),
            selected_config: None,
            config_text: String::new(),
            config_check: Err("no config chosen".into()),
            past_runs: Vec::new(),
            live: None,
            view: None,
            selected_block: 0,
            selected_unit: None,
            colour_by: ColourBy::State,
            message: None,
        };
        app.rescan();
        if !app.config_files.is_empty() {
            app.choose_config(0);
        }
        app
    }

    fn rescan(&mut self) {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&self.configs_dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "toml"))
            .collect();
        files.sort();
        self.config_files = files;
        self.past_runs = list_runs(&self.runs_dir);
    }

    fn choose_config(&mut self, index: usize) {
        let Some(path) = self.config_files.get(index) else {
            return;
        };
        match std::fs::read_to_string(path) {
            Ok(text) => {
                self.selected_config = Some(index);
                self.config_text = text;
                self.check_config();
            }
            Err(e) => self.message = Some(format!("{}: {e}", path.display())),
        }
    }

    fn check_config(&mut self) {
        self.config_check = Config::from_toml(&self.config_text, &[])
            .map(|_| ())
            .map_err(|e| e.to_string());
    }

    /// True while a run started from this window is going on.
    pub fn is_running(&self) -> bool {
        self.live.is_some()
    }

    /// The run on screen, if any.
    pub fn view(&self) -> Option<&RunView> {
        self.view.as_ref()
    }

    /// Opens a finished (or running) run folder for replay.
    pub fn open_run(&mut self, dir: &Path) {
        let view = RunView::open(dir);
        // Show the run's own settings, ready to be changed and run again.
        if let Ok(text) = std::fs::read_to_string(dir.join("config.toml")) {
            self.config_text = text;
            self.selected_config = None;
            self.check_config();
        }
        self.selected_block = 0;
        self.selected_unit = None;
        self.view = Some(view);
    }

    /// Starts a run from the config text. `wake` is called from the run's
    /// thread whenever there is something new to show.
    pub fn start(&mut self, wake: impl Fn() + Send + 'static) {
        if self.live.is_some() {
            return;
        }
        let config = match Config::from_toml(&self.config_text, &[]) {
            Ok(c) => c,
            Err(e) => {
                self.message = Some(e.to_string());
                return;
            }
        };
        let dir = match create_run_dir(&config, &self.runs_dir) {
            Ok(d) => d,
            Err(e) => {
                self.message = Some(e.to_string());
                return;
            }
        };
        let stop = Arc::new(AtomicBool::new(false));
        let (sender, events) = channel();
        let (thread_stop, thread_dir) = (stop.clone(), dir.clone());
        let spawned = std::thread::Builder::new()
            .name("tesseract-run".into())
            .spawn(move || {
                let result = run_in(&config, &thread_dir, &thread_stop, &mut |m| {
                    let _ = sender.send(Event::Metrics(Box::new(m.clone())));
                    wake();
                });
                let _ = sender.send(Event::Done(
                    result.map(|o| o.passed).map_err(|e| e.to_string()),
                ));
                wake();
            });
        if let Err(e) = spawned {
            self.message = Some(format!("could not start the run: {e}"));
            return;
        }
        self.live = Some(LiveRun { stop, events });
        self.message = None;
        self.selected_block = 0;
        self.selected_unit = None;
        self.view = Some(RunView::starting(
            &dir,
            Config::from_toml(&self.config_text, &[]).ok(),
        ));
    }

    /// Asks the running run to stop after its current step.
    pub fn stop(&mut self) {
        if let Some(live) = &self.live {
            live.stop.store(true, Ordering::Relaxed);
        }
    }

    /// Takes in what the run's thread reported since the last frame.
    pub fn poll(&mut self) {
        let Some(live) = &self.live else {
            return;
        };
        let mut finished = None;
        let mut fresh = false;
        while let Ok(event) = live.events.try_recv() {
            match event {
                Event::Metrics(m) => {
                    if let Some(view) = &mut self.view {
                        view.metrics.push(*m);
                    }
                    fresh = true;
                }
                Event::Done(result) => finished = Some(result),
            }
        }
        if (fresh || finished.is_some())
            && let Some(view) = &mut self.view
        {
            view.refresh();
        }
        if let Some(result) = finished {
            self.live = None;
            if let Err(e) = result {
                self.message = Some(e);
            }
            self.past_runs = list_runs(&self.runs_dir);
        }
    }

    /// Draws the whole window.
    pub fn ui(&mut self, ui: &mut Ui) {
        self.poll();
        let theme = Theme::of(ui.visuals().dark_mode);
        egui::Panel::left("setup")
            .default_size(330.0)
            .min_size(260.0)
            .show(ui, |ui| self.setup_area(ui, &theme));
        egui::CentralPanel::default_margins().show(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    self.run_area(ui, &theme);
                    ui.add_space(10.0);
                    ui.separator();
                    self.neurons_area(ui, &theme);
                });
        });
    }

    // ------------------------------------------------------------------
    // Setup

    fn setup_area(&mut self, ui: &mut Ui, theme: &Theme) {
        ui.add_space(6.0);
        ui.heading("Setup");
        ui.add_space(4.0);

        let current = self
            .selected_config
            .and_then(|i| self.config_files.get(i))
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| {
                if self.config_text.is_empty() {
                    "choose a test".to_string()
                } else {
                    "settings of the open run".to_string()
                }
            });
        let mut chosen = None;
        ui.horizontal(|ui| {
            ui.label("Test");
            egui::ComboBox::from_id_salt("config")
                .selected_text(current)
                .width(190.0)
                .show_ui(ui, |ui| {
                    for (i, path) in self.config_files.iter().enumerate() {
                        let name = path
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        if ui
                            .selectable_label(self.selected_config == Some(i), name)
                            .clicked()
                        {
                            chosen = Some(i);
                        }
                    }
                });
            if ui
                .button("Rescan")
                .on_hover_text("Look again for config files and past runs")
                .clicked()
            {
                self.rescan();
            }
        });
        if let Some(i) = chosen {
            self.choose_config(i);
        }
        if self.config_files.is_empty() {
            ui.label(
                RichText::new(format!("No .toml file in {}", self.configs_dir.display()))
                    .color(theme.text_secondary),
            );
        }

        ui.add_space(4.0);
        ui.label(RichText::new("Settings (edited here, the file is not changed)").small());
        let editor_height = (ui.available_height() - 190.0).max(120.0);
        let mut changed = false;
        egui::ScrollArea::vertical()
            .id_salt("config-editor")
            .max_height(editor_height)
            .show(ui, |ui| {
                changed = ui
                    .add(
                        egui::TextEdit::multiline(&mut self.config_text)
                            .code_editor()
                            .desired_width(f32::INFINITY)
                            .desired_rows(18),
                    )
                    .changed();
            });
        if changed {
            self.check_config();
        }
        match &self.config_check {
            Ok(()) => {
                ui.label(RichText::new("Settings are valid.").color(theme.text_secondary));
            }
            Err(e) => {
                ui.label(RichText::new(format!("Not valid: {e}")).color(theme.critical));
            }
        }

        ui.add_space(4.0);
        let mut start = false;
        ui.horizontal(|ui| {
            let can_start = self.live.is_none() && self.config_check.is_ok();
            if ui
                .add_enabled(can_start, egui::Button::new("Start"))
                .clicked()
            {
                start = true;
            }
            if ui
                .add_enabled(self.live.is_some(), egui::Button::new("Stop"))
                .clicked()
            {
                self.stop();
            }
            if self.live.is_some() {
                ui.spinner();
                ui.label("running");
            }
        });
        if start {
            let ctx = ui.ctx().clone();
            self.start(move || ctx.request_repaint());
        }
        if let Some(m) = &self.message {
            ui.label(RichText::new(m).color(theme.critical));
        }

        ui.add_space(8.0);
        ui.separator();
        ui.label(RichText::new("Past runs").strong());
        let mut open = None;
        egui::ScrollArea::vertical()
            .id_salt("past-runs")
            .auto_shrink([false, true])
            .show(ui, |ui| {
                if self.past_runs.is_empty() {
                    ui.label(RichText::new("None yet.").color(theme.text_secondary));
                }
                for dir in &self.past_runs {
                    let shown = self.view.as_ref().is_some_and(|v| &v.dir == dir);
                    if ui.selectable_label(shown, RunView::label(dir)).clicked() && !shown {
                        open = Some(dir.clone());
                    }
                }
            });
        if let Some(dir) = open {
            if self.live.is_none() {
                self.open_run(&dir);
            } else {
                self.message = Some("Stop the current run before opening another.".into());
            }
        }
    }

    // ------------------------------------------------------------------
    // Run

    fn run_area(&mut self, ui: &mut Ui, theme: &Theme) {
        ui.heading("Run");
        let Some(view) = &self.view else {
            ui.label(
                RichText::new("Choose a test and press Start, or open a past run.")
                    .color(theme.text_secondary),
            );
            return;
        };
        let Some(last) = view.metrics.last() else {
            ui.label(
                RichText::new("Waiting for the first measurement…").color(theme.text_secondary),
            );
            return;
        };

        // Headline.
        let status = if self.live.is_some() {
            "running".to_string()
        } else {
            view.info
                .as_ref()
                .map(|i| i.status.clone())
                .unwrap_or_else(|| "unknown".into())
        };
        let steps = view
            .budget_steps()
            .map(|b| format!("step {} of {}", last.step, b))
            .unwrap_or_else(|| format!("step {}", last.step));
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(RunView::label(&view.dir)).strong());
            ui.label(format!("· {status} · {steps}"));
            if status != "running" {
                let passed = last.held_out >= last.threshold;
                let (word, color) = if passed {
                    ("PASS", theme.good)
                } else {
                    ("FAIL", theme.critical)
                };
                // A status dot beside the word: colour never stands alone.
                let (dot, _) = ui.allocate_exact_size(vec2(10.0, 10.0), Sense::hover());
                ui.painter().circle_filled(dot.center(), 5.0, color);
                ui.label(RichText::new(word).strong());
                ui.label(format!(
                    "(held-out {:.2}%, needs {:.0}%)",
                    last.held_out * 100.0,
                    last.threshold * 100.0
                ));
            }
        });
        if let Some(p) = &view.problem {
            ui.label(RichText::new(p).color(theme.text_secondary).small());
        }

        // Figures.
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            let tiles: [(&str, String); 10] = [
                ("held-out", format!("{:.2}%", last.held_out * 100.0)),
                ("train", format!("{:.2}%", last.train * 100.0)),
                ("flips / step", compact(last.flips_per_step)),
                (
                    "connections active",
                    compact(last.connections_active as f64),
                ),
                ("dormant", compact(last.connections_dormant as f64)),
                ("removed", compact(last.connections_removed as f64)),
                ("protected", compact(last.connections_protected as f64)),
                (
                    "units active / dormant / free",
                    format!(
                        "{} / {} / {}",
                        last.units_active, last.units_dormant, last.units_free
                    ),
                ),
                (
                    "speed",
                    format!(
                        "{} steps/s · {:.1} µs/token",
                        compact(last.steps_per_second),
                        last.micros_per_token
                    ),
                ),
                (
                    "parameters",
                    format!("{}B", compact(last.parameter_bytes as f64)),
                ),
            ];
            for (label, value) in tiles {
                ui.vertical(|ui| {
                    ui.label(RichText::new(label).small().color(theme.text_secondary));
                    ui.label(RichText::new(value).strong());
                });
                ui.add_space(14.0);
            }
        });
        ui.add_space(6.0);

        // Scores: train and held-out, with baselines and threshold.
        let x_max = view.budget_steps().map(|b| b as f64);
        let points = |f: &dyn Fn(&Metrics) -> f64, skip_first: bool| -> Vec<[f64; 2]> {
            view.metrics
                .iter()
                .skip(usize::from(skip_first))
                .map(|m| [m.step as f64, f(m)])
                .collect()
        };
        let mut ref_lines: Vec<RefLine> = last
            .baselines
            .iter()
            .map(|(name, score)| RefLine {
                label: name.replace('_', " "),
                y: *score,
            })
            .collect();
        ref_lines.push(RefLine {
            label: "pass at".into(),
            y: last.threshold,
        });
        Chart {
            title: "Score".into(),
            x_label: "steps".into(),
            series: vec![
                Series {
                    name: "held-out".into(),
                    color: theme.series[0],
                    points: points(&|m| m.held_out, false),
                },
                Series {
                    name: "train".into(),
                    color: theme.series[1],
                    points: points(&|m| m.train, true),
                },
            ],
            ref_lines,
            unit: Unit::Percent,
            x_max,
            height: 230.0,
        }
        .show(ui, theme);
        ui.add_space(6.0);

        let extra_curve = noise_curve(last);
        let n = if extra_curve.is_some() { 3 } else { 2 };
        ui.columns(n, |cols| {
            Chart {
                title: "Flips per step".into(),
                x_label: "steps".into(),
                series: vec![Series {
                    name: "flips".into(),
                    color: theme.series[0],
                    points: points(&|m| m.flips_per_step, true),
                }],
                ref_lines: Vec::new(),
                unit: Unit::Count,
                x_max,
                height: 170.0,
            }
            .show(&mut cols[0], theme);
            Chart {
                title: "Connections".into(),
                x_label: "steps".into(),
                series: vec![
                    Series {
                        name: "active".into(),
                        color: theme.series[0],
                        points: points(&|m| m.connections_active as f64, false),
                    },
                    Series {
                        name: "dormant".into(),
                        color: theme.series[1],
                        points: points(&|m| m.connections_dormant as f64, false),
                    },
                    Series {
                        name: "protected".into(),
                        color: theme.series[2],
                        points: points(&|m| m.connections_protected as f64, false),
                    },
                ],
                ref_lines: Vec::new(),
                unit: Unit::Count,
                x_max,
                height: 170.0,
            }
            .show(&mut cols[1], theme);
            if let Some(curve) = extra_curve {
                Chart {
                    title: "Recall against key noise".into(),
                    x_label: "% of key bits flipped".into(),
                    series: vec![Series {
                        name: "recall".into(),
                        color: theme.series[0],
                        points: curve,
                    }],
                    ref_lines: Vec::new(),
                    unit: Unit::Percent,
                    x_max: None,
                    height: 170.0,
                }
                .show(&mut cols[2], theme);
            }
        });
    }

    // ------------------------------------------------------------------
    // Neurons

    fn neurons_area(&mut self, ui: &mut Ui, theme: &Theme) {
        ui.heading("Neurons");
        let Some(view) = &mut self.view else {
            return;
        };
        if view.snapshots.is_empty() || view.snapshot.is_none() {
            ui.label(RichText::new("No snapshot yet.").color(theme.text_secondary));
            return;
        }

        // Which snapshot, and what the cells show.
        ui.horizontal_wrapped(|ui| {
            let last = view.snapshots.len() - 1;
            let mut index = view.index;
            ui.label("Snapshot");
            if last > 0 {
                let steps: Vec<u64> = view.snapshots.iter().map(|(s, _)| *s).collect();
                ui.add(
                    egui::Slider::new(&mut index, 0..=last)
                        .show_value(false)
                        .custom_formatter(move |i, _| format!("step {}", steps[i as usize])),
                );
            }
            ui.label(format!("step {}", view.snapshots[view.index].0));
            let mut follow = view.follow;
            ui.checkbox(&mut follow, "follow the newest");
            if index != view.index {
                view.follow = false;
                view.select(index);
            } else if follow != view.follow {
                view.follow = follow;
                if follow {
                    view.refresh();
                }
            }
            ui.add_space(16.0);
            ui.label("Colour by");
            ui.selectable_value(&mut self.colour_by, ColourBy::State, "state");
            ui.selectable_value(&mut self.colour_by, ColourBy::Usage, "usage");
            ui.selectable_value(&mut self.colour_by, ColourBy::LiveInputs, "live inputs");
        });
        let Some(snapshot) = &view.snapshot else {
            return;
        };
        if snapshot.blocks.len() > 1 {
            ui.horizontal(|ui| {
                ui.label("Block");
                for (i, b) in snapshot.blocks.iter().enumerate() {
                    ui.selectable_value(&mut self.selected_block, i, format!("{i}: {}", b.kind));
                }
            });
        }
        let Some(block) = snapshot
            .blocks
            .get(self.selected_block)
            .or(snapshot.blocks.first())
        else {
            return;
        };
        if self.selected_unit.is_none_or(|u| u >= block.units.len()) {
            self.selected_unit = block.units.iter().position(|u| u.detail.is_some());
        }

        ui.add_space(4.0);
        let colour_by = self.colour_by;
        let mut selected = self.selected_unit;
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new(format!(
                        "{} block · {} unit{}",
                        block.kind,
                        block.units.len(),
                        if block.units.len() == 1 { "" } else { "s" }
                    ))
                    .color(theme.text_secondary),
                );
                if let Some(clicked) = unit_grid(ui, theme, block, colour_by, selected) {
                    selected = Some(clicked);
                }
                grid_legend(ui, theme, colour_by);
            });
            ui.add_space(12.0);
            ui.vertical(
                |ui| match selected.and_then(|u| block.units.get(u).map(|x| (u, x))) {
                    Some((index, unit)) => unit_detail(ui, theme, &block.kind, index, unit),
                    None => {
                        ui.label(
                            RichText::new("Click a unit to see its connections.")
                                .color(theme.text_secondary),
                        );
                    }
                },
            );
        });
        self.selected_unit = selected;
    }
}

/// Points of the "recall against noise" curve, when the extra scores are
/// named `noise_<per mille>`.
fn noise_curve(m: &Metrics) -> Option<Vec<[f64; 2]>> {
    if m.extra.len() < 2 {
        return None;
    }
    let mut points = Vec::new();
    for (name, score) in &m.extra {
        let per_mille: f64 = name.strip_prefix("noise_")?.parse().ok()?;
        points.push([per_mille / 10.0, *score]);
    }
    points.sort_by(|a, b| a[0].total_cmp(&b[0]));
    Some(points)
}

fn state_name(state: u8) -> &'static str {
    match state {
        0 => "free",
        1 => "active",
        _ => "dormant",
    }
}

/// The grid of units: one cell per unit. Returns the unit clicked.
fn unit_grid(
    ui: &mut Ui,
    theme: &Theme,
    block: &BlockSnapshot,
    colour_by: ColourBy,
    selected: Option<usize>,
) -> Option<usize> {
    let n = block.units.len().max(1);
    let cols = (n as f64).sqrt().ceil() as usize;
    let rows = n.div_ceil(cols);
    let cell = (340.0 / cols as f32).clamp(4.0, 28.0);
    let gap = if cell >= 8.0 { 2.0 } else { 1.0 };
    let size = vec2(cols as f32 * cell + 8.0, rows as f32 * cell + 8.0);
    let (response, painter) = ui.allocate_painter(size, Sense::click());
    let origin = response.rect.min + vec2(4.0, 4.0);
    painter.rect_filled(response.rect, 6.0, theme.surface);

    let max_usage = block
        .units
        .iter()
        .map(|u| u.usage)
        .max()
        .unwrap_or(0)
        .max(1) as f64;
    let max_live = block.units.iter().map(|u| u.live).max().unwrap_or(0).max(1) as f64;
    let cell_rect = |i: usize| {
        let (r, c) = (i / cols, i % cols);
        Rect::from_min_size(
            origin + vec2(c as f32 * cell, r as f32 * cell),
            vec2(cell - gap, cell - gap),
        )
    };
    for (i, unit) in block.units.iter().enumerate() {
        let color = match (unit.state, colour_by) {
            (0, _) => theme.neutral,
            (2, ColourBy::State) => theme.dormant,
            (_, ColourBy::State) => theme.positive,
            (_, ColourBy::Usage) => theme.magnitude(unit.usage as f64 / max_usage),
            (_, ColourBy::LiveInputs) => theme.magnitude(f64::from(unit.live) / max_live),
        };
        painter.rect_filled(cell_rect(i), if cell >= 8.0 { 2.0 } else { 0.0 }, color);
    }
    if let Some(s) = selected.filter(|s| *s < block.units.len()) {
        painter.rect_stroke(
            cell_rect(s).expand(1.5),
            2.0,
            Stroke::new(2.0, theme.text),
            StrokeKind::Outside,
        );
    }

    let hovered = response.hover_pos().and_then(|p| {
        let rel = p - origin;
        if rel.x < 0.0 || rel.y < 0.0 {
            return None;
        }
        let (c, r) = ((rel.x / cell) as usize, (rel.y / cell) as usize);
        (c < cols && r * cols + c < block.units.len()).then_some(r * cols + c)
    });
    let clicked = response.clicked().then_some(hovered).flatten();
    if let Some(i) = hovered {
        let u = &block.units[i];
        response.on_hover_text_at_pointer(format!(
            "unit {i} · {} · fired {} · {} live inputs · bias {}",
            state_name(u.state),
            u.usage,
            u.live,
            u.bias
        ));
    }
    clicked
}

fn swatch(ui: &mut Ui, color: Color32, label: &str, theme: &Theme) {
    let (rect, _) = ui.allocate_exact_size(vec2(10.0, 10.0), Sense::hover());
    ui.painter().rect_filled(rect, 2.0, color);
    ui.label(RichText::new(label).small().color(theme.text_secondary));
    ui.add_space(6.0);
}

fn grid_legend(ui: &mut Ui, theme: &Theme, colour_by: ColourBy) {
    ui.horizontal(|ui| match colour_by {
        ColourBy::State => {
            swatch(ui, theme.positive, "active", theme);
            swatch(ui, theme.dormant, "dormant", theme);
            swatch(ui, theme.neutral, "free", theme);
        }
        ColourBy::Usage | ColourBy::LiveInputs => {
            ui.label(RichText::new("few").small().color(theme.text_secondary));
            for c in theme.ramp {
                let (rect, _) = ui.allocate_exact_size(vec2(14.0, 10.0), Sense::hover());
                ui.painter().rect_filled(rect, 0.0, c);
            }
            ui.label(RichText::new("many").small().color(theme.text_secondary));
            ui.add_space(6.0);
            swatch(ui, theme.neutral, "free", theme);
        }
    });
}

/// What one cell of a connection map shows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cell {
    Empty,
    Positive,
    Negative,
    /// Masked, but with evidence kept in its counter.
    Kept,
}

/// The detail of one unit: figures and two maps of 64 by 64 connections.
fn unit_detail(ui: &mut Ui, theme: &Theme, kind: &str, index: usize, unit: &UnitSnapshot) {
    ui.label(RichText::new(format!("Unit {index} · {}", state_name(unit.state))).strong());
    let Some(d) = &unit.detail else {
        ui.label(RichText::new("A free unit has no connections yet.").color(theme.text_secondary));
        return;
    };
    let ones = |words: &[u64]| words.iter().map(|w| w.count_ones()).sum::<u32>();
    let need = i64::from(unit.live) + i64::from(unit.bias);
    let threshold = if need <= 0 { 0 } else { (need + 1) / 2 };
    let facts = if kind == "attention" {
        format!(
            "selected {} times · matches on {} positions ({} protected) · copies {} positions",
            unit.usage,
            unit.live,
            ones(&d.input_protected),
            ones(&d.output_plus)
        )
    } else {
        format!(
            "fired {} times · {} live inputs ({} protected) · fires at {} matches (bias {}) · votes +{} / −{}",
            unit.usage,
            unit.live,
            ones(&d.input_protected),
            threshold,
            unit.bias,
            ones(&d.output_plus),
            ones(&d.output_minus)
        )
    };
    ui.label(RichText::new(facts).color(theme.text_secondary));
    ui.add_space(4.0);

    let (input_title, output_title) = if kind == "attention" {
        ("Match mask", "Value mask (positions copied)")
    } else {
        ("Inputs", "Output votes")
    };
    ui.horizontal_top(|ui| {
        ui.vertical(|ui| {
            ui.label(input_title);
            connection_map(ui, theme, "in", |p| {
                let live = UnitDetail::bit(&d.input_mask, p);
                let sign = UnitDetail::bit(&d.input_sign, p);
                let counter = d.counters.as_ref().and_then(|c| c.get(p)).copied();
                let cell = match (live, sign, counter) {
                    (true, true, _) => Cell::Positive,
                    (true, false, _) => Cell::Negative,
                    (false, _, Some(c)) if c != 0 => Cell::Kept,
                    _ => Cell::Empty,
                };
                let mut note = match cell {
                    Cell::Positive => "live, sign +".to_string(),
                    Cell::Negative => "live, sign −".to_string(),
                    Cell::Kept => "dormant, evidence kept".to_string(),
                    Cell::Empty => "no connection".to_string(),
                };
                if let Some(c) = counter {
                    note.push_str(&format!(", counter {c}"));
                }
                if UnitDetail::bit(&d.input_protected, p) {
                    note.push_str(", protected");
                }
                (cell, note)
            });
        });
        ui.add_space(10.0);
        ui.vertical(|ui| {
            ui.label(output_title);
            connection_map(ui, theme, "out", |p| {
                if UnitDetail::bit(&d.output_plus, p) {
                    (Cell::Positive, "votes +1".to_string())
                } else if UnitDetail::bit(&d.output_minus, p) {
                    (Cell::Negative, "votes −1".to_string())
                } else {
                    (Cell::Empty, "no vote".to_string())
                }
            });
        });
    });
    ui.horizontal(|ui| {
        swatch(ui, theme.positive, "positive", theme);
        swatch(ui, theme.negative, "negative", theme);
        swatch(
            ui,
            theme.faded(theme.dormant, 110),
            "dormant, evidence kept",
            theme,
        );
        swatch(ui, theme.neutral, "none", theme);
    });
    ui.label(
        RichText::new("Each map is one entity: 8 domains (rows d0–d7) of 8 values (columns v0–v7) of 8 × 8 bits.")
            .small()
            .color(theme.text_secondary),
    );
}

/// A map of the 4096 positions of an entity. Position `p` is drawn at
/// column `value * 8 + bit`, row `domain * 8 + byte`, so each domain is a
/// band of 8 rows and each value a square of 8 by 8.
fn connection_map(ui: &mut Ui, theme: &Theme, id: &str, cell_at: impl Fn(usize) -> (Cell, String)) {
    const PX: f32 = 4.0;
    const SIDE: f32 = 64.0 * PX;
    let margin = vec2(22.0, 14.0);
    let (response, painter) =
        ui.allocate_painter(vec2(SIDE, SIDE) + margin + vec2(4.0, 4.0), Sense::hover());
    let origin = response.rect.min + margin;
    let area = Rect::from_min_size(origin, vec2(SIDE, SIDE));
    painter.rect_filled(area, 0.0, theme.neutral);
    let small = FontId::proportional(10.0);
    for i in 0..8 {
        let at = i as f32 * 8.0 * PX + 4.0 * PX;
        painter.text(
            pos2(origin.x - 4.0, origin.y + at),
            Align2::RIGHT_CENTER,
            format!("d{i}"),
            small.clone(),
            theme.text_secondary,
        );
        painter.text(
            pos2(origin.x + at, origin.y - 2.0),
            Align2::CENTER_BOTTOM,
            format!("v{i}"),
            small.clone(),
            theme.text_secondary,
        );
    }
    for p in 0..4096usize {
        let (cell, _) = cell_at(p);
        let color = match cell {
            Cell::Empty => continue,
            Cell::Positive => theme.positive,
            Cell::Negative => theme.negative,
            Cell::Kept => theme.faded(theme.dormant, 110),
        };
        let (x, y) = position_to_cell(p);
        painter.rect_filled(
            Rect::from_min_size(origin + vec2(x as f32 * PX, y as f32 * PX), vec2(PX, PX)),
            0.0,
            color,
        );
    }
    // Separators between domains and between values.
    for i in 1..8 {
        let at = i as f32 * 8.0 * PX;
        painter.hline(
            area.x_range(),
            origin.y + at,
            Stroke::new(1.0, theme.surface),
        );
        painter.vline(
            origin.x + at,
            area.y_range(),
            Stroke::new(1.0, theme.surface),
        );
    }
    let _ = id;
    if let Some(pointer) = response.hover_pos().filter(|p| area.contains(*p)) {
        let rel = pointer - origin;
        let (x, y) = ((rel.x / PX) as usize, (rel.y / PX) as usize);
        if let Some(p) = cell_to_position(x.min(63), y.min(63)) {
            let (_, note) = cell_at(p);
            response.on_hover_text_at_pointer(format!(
                "domain {} · value {} · byte {} · bit {} (position {p})\n{note}",
                p >> 9,
                (p >> 6) & 7,
                (p >> 3) & 7,
                p & 7
            ));
        }
    }
}

/// Column and row of a bit position in a connection map.
fn position_to_cell(p: usize) -> (usize, usize) {
    let (domain, value, byte, bit) = (p >> 9, (p >> 6) & 7, (p >> 3) & 7, p & 7);
    (value * 8 + bit, domain * 8 + byte)
}

/// The bit position drawn at a column and row, if inside the map.
fn cell_to_position(x: usize, y: usize) -> Option<usize> {
    (x < 64 && y < 64).then_some(((y / 8) << 9) | ((x / 8) << 6) | ((y % 8) << 3) | (x % 8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_layout_is_a_bijection() {
        let mut seen = vec![false; 4096];
        for p in 0..4096 {
            let (x, y) = position_to_cell(p);
            assert!(x < 64 && y < 64);
            assert_eq!(cell_to_position(x, y), Some(p));
            assert!(!seen[y * 64 + x]);
            seen[y * 64 + x] = true;
        }
        assert_eq!(cell_to_position(64, 0), None);
        // A domain is a band of 8 rows; a value is a square of 8 by 8.
        assert_eq!(position_to_cell(512), (0, 8));
        assert_eq!(position_to_cell(64), (8, 0));
    }
}
