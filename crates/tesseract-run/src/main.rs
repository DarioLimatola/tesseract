//! `tesseract`: run one config file.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

use clap::{Parser, Subcommand};
use tesseract_run::{Config, Metrics, run};

#[derive(Parser)]
#[command(name = "tesseract", version, about = "Tesseract runner")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run one config file and write one run folder.
    Run {
        /// The config file (TOML).
        config: PathBuf,
        /// Folder that receives the run folders.
        #[arg(long, default_value = "runs")]
        out: PathBuf,
        /// Override a setting, for example `--set task.pairs=2000`.
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
        /// Number of threads for evaluation (default: all cores).
        #[arg(long)]
        threads: Option<usize>,
    },
}

fn print_line(m: &Metrics) {
    let baseline = m
        .baselines
        .iter()
        .map(|(k, v)| format!("{k} {:.1}%", v * 100.0))
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "step {:>8}  train {:>6.2}%  held-out {:>6.2}%  ({baseline})  flips/step {:>7.2}  active {:>8}  dormant {:>8}  {:>8.0} steps/s  {:>7.1} us/token",
        m.step,
        m.train * 100.0,
        m.held_out * 100.0,
        m.flips_per_step,
        m.connections_active,
        m.connections_dormant,
        m.steps_per_second,
        m.micros_per_token,
    );
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let Command::Run {
        config,
        out,
        set,
        threads,
    } = cli.command;
    if let Some(n) = threads
        && let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
    {
        eprintln!("error: threads: {e}");
        return ExitCode::from(2);
    }
    let config = match Config::load(&config, &set) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let stop = AtomicBool::new(false);
    match run(&config, &out, &stop, &mut print_line) {
        Ok(outcome) => {
            for (name, score) in &outcome.last.extra {
                println!("  {name}: {:.2}%", score * 100.0);
            }
            println!(
                "{}: held-out {:.2}%, threshold {:.0}% -> {}",
                config.name,
                outcome.last.held_out * 100.0,
                outcome.last.threshold * 100.0,
                if outcome.passed { "PASS" } else { "FAIL" }
            );
            println!("run folder: {}", outcome.dir.display());
            if outcome.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}
