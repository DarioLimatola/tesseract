//! `tesseract-monitor`: the local desktop monitor.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::Parser;
use eframe::egui;
use tesseract_monitor::MonitorApp;

#[derive(Parser)]
#[command(
    name = "tesseract-monitor",
    version,
    about = "Tesseract monitor (local desktop window)"
)]
struct Cli {
    /// Folder with the config files of the tests.
    #[arg(long, default_value = "configs")]
    configs: PathBuf,
    /// Folder that holds the run folders.
    #[arg(long, default_value = "runs")]
    runs: PathBuf,
    /// Open this run folder at start.
    #[arg(long)]
    open: Option<PathBuf>,
    /// Save a picture of the window to this PNG file and quit (for checks).
    #[arg(long, hide = true)]
    screenshot: Option<PathBuf>,
}

struct Window {
    app: MonitorApp,
    screenshot: Option<PathBuf>,
    frames: u32,
    asked: bool,
    saved: Arc<AtomicBool>,
}

impl eframe::App for Window {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.app.ui(ui);
        let Some(path) = self.screenshot.clone() else {
            return;
        };
        // Let the window settle for a few frames, then save it and quit.
        self.frames += 1;
        let ctx = ui.ctx().clone();
        if self.saved.load(Ordering::Relaxed) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else if self.frames >= 8 && !self.asked {
            self.asked = true;
            let saved = self.saved.clone();
            let wake = ctx.clone();
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(
                egui::ScreenshotCallback::new(move |image| {
                    if let Err(e) = save_png(&path, &image) {
                        eprintln!("error: {}: {e}", path.display());
                    }
                    saved.store(true, Ordering::Relaxed);
                    wake.request_repaint();
                }),
            ));
        }
        ctx.request_repaint();
    }
}

fn save_png(
    path: &std::path::Path,
    image: &egui::ColorImage,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut encoder = png::Encoder::new(file, image.width() as u32, image.height() as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let bytes: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_array()).collect();
    writer.write_image_data(&bytes)?;
    Ok(())
}

fn main() -> eframe::Result {
    let cli = Cli::parse();
    let mut app = MonitorApp::new(cli.configs, cli.runs);
    if let Some(dir) = &cli.open {
        app.open_run(dir);
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Tesseract monitor")
            .with_inner_size([1360.0, 900.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };
    let window = Window {
        app,
        screenshot: cli.screenshot,
        frames: 0,
        asked: false,
        saved: Arc::new(AtomicBool::new(false)),
    };
    eframe::run_native(
        "Tesseract monitor",
        options,
        Box::new(move |_cc| Ok(Box::new(window))),
    )
}
