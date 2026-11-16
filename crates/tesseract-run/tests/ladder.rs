//! Tests 1 to 3 of the test plan, run through the runner on the config
//! files of the repository, plus checks of the run folder.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use tesseract_core::{Entity, WORDS};
use tesseract_run::snapshot::UnitDetail;
use tesseract_run::{Config, Metrics, RunOutcome, Snapshot, list_snapshots, read_metrics, read_run_info, run};
use tesseract_tasks::{Majority, MajoritySettings};

fn config(file: &str, overrides: &[&str]) -> Config {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../configs")
        .join(file);
    let overrides: Vec<String> = overrides.iter().map(|s| s.to_string()).collect();
    Config::load(&path, &overrides).expect("config loads")
}

fn run_in(config: &Config, dir: &Path) -> (RunOutcome, Vec<Metrics>) {
    let mut seen = Vec::new();
    let outcome = run(config, dir, &AtomicBool::new(false), &mut |m| seen.push(m.clone()))
        .expect("run succeeds");
    (outcome, seen)
}

fn last_snapshot(dir: &Path) -> Snapshot {
    let (_, path) = list_snapshots(dir).pop().expect("a snapshot exists");
    Snapshot::load(&path).expect("snapshot loads")
}

fn entity(words: &[u64]) -> Entity {
    let mut e = Entity::ZERO;
    e.0.copy_from_slice(&words[..WORDS]);
    e
}

fn detail(snapshot: &Snapshot) -> &UnitDetail {
    snapshot.blocks[0].units[0].detail.as_ref().expect("detail")
}

#[test]
fn test1_majority_unit_finds_the_hidden_subset() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = config("test1_majority.toml", &["task.held_out=4000"]);
    let (outcome, _) = run_in(&cfg, tmp.path());
    assert!(outcome.passed, "held-out {}", outcome.last.held_out);
    assert!(outcome.last.held_out >= 0.99);
    assert!(outcome.last.baselines["chance"] < 0.6);

    // The mask is exactly the hidden subset and the signs are the hidden signs.
    let task = Majority::new(MajoritySettings {
        subset: 65,
        hidden_signs: true,
        held_out: 0,
        seed: cfg.seed,
    });
    let snap = last_snapshot(&outcome.dir);
    let d = detail(&snap);
    let mask = entity(&d.input_mask);
    assert_eq!(mask, *task.members());
    assert_eq!(entity(&d.input_sign) & mask, *task.signs() & mask);
    assert_eq!(outcome.last.connections_active, 65);
}

#[test]
fn test2_recall_memory_block_matches_nearest_neighbour() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = config("test2_recall.toml", &[]);
    let (outcome, _) = run_in(&cfg, tmp.path());
    assert!(outcome.passed, "held-out {}", outcome.last.held_out);
    assert_eq!(outcome.last.baselines["nearest_neighbour"], 1.0);
    assert_eq!(outcome.last.recruited, 1000);
    assert_eq!(outcome.last.dropped, 0);
    assert_eq!(outcome.last.units_active, 1000);
    // The curve: exact up to 30% noise, and a cliff past the vigilance
    // threshold (0.625 means at most 37.5% of key bits may differ).
    for level in ["noise_000", "noise_100", "noise_200", "noise_300"] {
        assert_eq!(outcome.last.extra[level], 1.0, "{level}");
    }
    assert!(outcome.last.extra["noise_400"] < 0.1);
}

#[test]
fn test2_capacity_is_the_number_of_units() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = config(
        "test2_recall.toml",
        &[
            "task.pairs=300",
            "model.units=200",
            "budget.steps=300",
            "budget.eval_every=300",
            "task.curve=[]",
        ],
    );
    let (outcome, _) = run_in(&cfg, tmp.path());
    assert!(!outcome.passed);
    assert_eq!(outcome.last.dropped, 100);
    assert!((outcome.last.held_out - 200.0 / 300.0).abs() < 1e-9);
}

#[test]
fn test3_lookup_head_learns_what_to_select() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = config("test3_lookup.toml", &["task.held_out=1000"]);
    let (outcome, seen) = run_in(&cfg, tmp.path());
    assert!(outcome.passed, "held-out {}", outcome.last.held_out);
    assert!(outcome.last.held_out >= 0.95);
    // Untrained, the head writes nothing and scores zero.
    assert_eq!(seen[0].held_out, 0.0);
    // Matching on everything is far from the threshold.
    assert!(outcome.last.baselines["nearest_record"] < 0.6);
    assert!(outcome.last.baselines["chance"] < 0.07);

    // What was learned: the key domain is fully matched, no decoy domain
    // is matched at all, and exactly the value domain is copied.
    let snap = last_snapshot(&outcome.dir);
    let d = detail(&snap);
    let matched = entity(&d.input_mask);
    let copied = entity(&d.output_plus);
    assert_eq!(matched.domain_count(2), 512, "key domain");
    for decoy in [0, 1, 3] {
        assert_eq!(matched.domain_count(decoy), 0, "decoy domain {decoy}");
    }
    assert_eq!(copied, Entity::domain_mask(5), "value domain");
}

fn comparable(metrics: &[Metrics]) -> Vec<Metrics> {
    metrics
        .iter()
        .cloned()
        .map(|mut m| {
            m.steps_per_second = 0.0;
            m.micros_per_token = 0.0;
            m
        })
        .collect()
}

#[test]
fn runs_are_reproducible_for_any_thread_count() {
    let cfg = config(
        "test3_lookup.toml",
        &["task.held_out=300", "budget.steps=1500"],
    );
    let mut results = Vec::new();
    for threads in [1, 4] {
        let tmp = tempfile::tempdir().unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let (outcome, seen) = pool.install(|| run_in(&cfg, tmp.path()));
        let snapshot = last_snapshot(&outcome.dir);
        results.push((comparable(&seen), snapshot));
    }
    assert_eq!(results[0].0, results[1].0);
    assert_eq!(results[0].1, results[1].1);
    // A different seed gives a different run.
    let tmp = tempfile::tempdir().unwrap();
    let other = config(
        "test3_lookup.toml",
        &["task.held_out=300", "budget.steps=1500", "seed=2"],
    );
    let (_, seen) = run_in(&other, tmp.path());
    assert_ne!(comparable(&seen), results[0].0);
}

#[test]
fn run_folder_is_complete_and_readable() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = config(
        "test1_majority.toml",
        &[
            "task.held_out=200",
            "budget.steps=3000",
            "budget.eval_every=1000",
            "budget.snapshot_every=1000",
        ],
    );
    let (outcome, seen) = run_in(&cfg, tmp.path());
    let dir: PathBuf = outcome.dir.clone();
    assert!(dir.starts_with(tmp.path().join("test1-majority")));
    // The stored config reproduces the run's config exactly.
    let stored = Config::load(&dir.join("config.toml"), &[]).unwrap();
    assert_eq!(stored, cfg);
    let info = read_run_info(&dir).unwrap();
    assert_eq!(info.status, "finished");
    assert_eq!(info.seed, cfg.seed);
    assert_eq!(info.steps_done, 3000);
    assert_eq!(info.engine_version, tesseract_core::ENGINE_VERSION);
    assert_eq!(info.passed, Some(outcome.passed));
    // One metrics line per evaluation, identical to what the callback saw.
    let logged = read_metrics(&dir).unwrap();
    assert_eq!(logged, seen);
    assert_eq!(
        logged.iter().map(|m| m.step).collect::<Vec<_>>(),
        vec![0, 1000, 2000, 3000]
    );
    // Snapshots at the start, every 1000 steps and at the end.
    let steps: Vec<u64> = list_snapshots(&dir).iter().map(|(s, _)| *s).collect();
    assert_eq!(steps, vec![0, 1000, 2000, 3000]);
    let snap = last_snapshot(&dir);
    assert_eq!(snap.step, 3000);
    assert_eq!(detail(&snap).counters.as_ref().unwrap().len(), 4096);
}

#[test]
fn a_stopped_run_is_recorded_as_stopped() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = config("test1_majority.toml", &["task.held_out=100"]);
    let stop = AtomicBool::new(true);
    let outcome = run(&cfg, tmp.path(), &stop, &mut |_| {}).unwrap();
    assert!(outcome.stopped);
    let info = read_run_info(&outcome.dir).unwrap();
    assert_eq!((info.status.as_str(), info.steps_done), ("stopped", 0));
}

#[test]
fn damaged_snapshots_are_errors_not_panics() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("x.tsnp");
    std::fs::write(&path, b"nope").unwrap();
    assert!(Snapshot::load(&path).is_err());
    std::fs::write(&path, b"TSNP\x63\x00\x00\x00rest").unwrap();
    let err = Snapshot::load(&path).unwrap_err().to_string();
    assert!(err.contains("version 99"), "{err}");
    std::fs::write(&path, b"TSNP\x01\x00\x00\x00\xff\xff\xff").unwrap();
    assert!(Snapshot::load(&path).is_err());
    assert!(Snapshot::load(&tmp.path().join("missing.tsnp")).is_err());
}
