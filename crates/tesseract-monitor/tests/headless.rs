//! The monitor's window logic, run without a window: every area is drawn
//! against real run folders, and a run is started and stopped from it.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use eframe::egui;
use tesseract_monitor::MonitorApp;
use tesseract_run::{Config, read_run_info, run};

fn configs_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs")
}

/// Draws one frame and returns the number of shapes painted.
fn frame(ctx: &egui::Context, app: &mut MonitorApp) -> usize {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1360.0, 900.0),
        )),
        ..Default::default()
    };
    let mut output = ctx.run_ui(input, |ui| app.ui(ui));
    // No renderer here: the font texture is not uploaded anywhere.
    output.textures_delta.clear();
    output.shapes.len()
}

#[test]
fn draws_empty_and_finished_runs_of_every_kind() {
    let tmp = tempfile::tempdir().unwrap();
    let ctx = egui::Context::default();
    let mut app = MonitorApp::new(configs_dir(), tmp.path().to_path_buf());
    let empty = frame(&ctx, &mut app);
    assert!(empty > 10, "setup area is drawn");

    let cases = [
        (
            "test1_majority.toml",
            vec!["budget.steps=4000", "task.held_out=200"],
        ),
        (
            "test2_recall.toml",
            vec![
                "task.pairs=100",
                "model.units=144",
                "budget.steps=100",
                "budget.eval_every=50",
            ],
        ),
        (
            "test3_lookup.toml",
            vec!["budget.steps=600", "task.held_out=100"],
        ),
    ];
    for (file, overrides) in cases {
        let overrides: Vec<String> = overrides.iter().map(|s| s.to_string()).collect();
        let config = Config::load(&configs_dir().join(file), &overrides).unwrap();
        let outcome = run(&config, tmp.path(), &AtomicBool::new(false), &mut |_| {}).unwrap();
        app.open_run(&outcome.dir);
        let view = app.view().expect("run is open");
        assert!(
            !view.metrics.is_empty() && view.snapshot.is_some(),
            "{file}"
        );
        // Charts and neuron maps add many shapes to the empty window.
        let mut shapes = 0;
        for _ in 0..3 {
            shapes = frame(&ctx, &mut app);
        }
        assert!(shapes > empty + 100, "{file}: {shapes} shapes");
    }
}

#[test]
fn opening_a_folder_that_is_not_a_run_does_not_panic() {
    let tmp = tempfile::tempdir().unwrap();
    let ctx = egui::Context::default();
    let mut app = MonitorApp::new(tmp.path().join("no-configs"), tmp.path().join("no-runs"));
    app.open_run(&tmp.path().join("nothing-here"));
    frame(&ctx, &mut app);
    assert!(app.view().unwrap().metrics.is_empty());
}

#[test]
fn starts_watches_and_stops_a_run() {
    let tmp = tempfile::tempdir().unwrap();
    let ctx = egui::Context::default();
    let mut app = MonitorApp::new(configs_dir(), tmp.path().to_path_buf());
    // The first config (test 1) is loaded at start; run it to the end.
    app.start(|| {});
    assert!(app.is_running());
    let deadline = Instant::now() + Duration::from_secs(120);
    while app.is_running() {
        frame(&ctx, &mut app);
        assert!(Instant::now() < deadline, "the run must finish");
        std::thread::sleep(Duration::from_millis(20));
    }
    frame(&ctx, &mut app);
    let view = app.view().unwrap();
    let info = read_run_info(&view.dir).unwrap();
    assert_eq!(info.status, "finished");
    assert_eq!(info.passed, Some(true));
    assert_eq!(view.metrics.last().unwrap().step, 40_000);
    assert!(view.snapshots.len() >= 2);

    // Start again and stop at once: the run is recorded as stopped.
    app.start(|| {});
    app.stop();
    while app.is_running() {
        frame(&ctx, &mut app);
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(5));
    }
    let info = read_run_info(&app.view().unwrap().dir).unwrap();
    assert_eq!(info.status, "stopped");
}
