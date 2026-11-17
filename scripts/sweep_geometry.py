#!/usr/bin/env python3
"""Build and measure the engine at several geometries.

The geometry is set by two constants in `tesseract-core/src/entity.rs`.
This script rewrites them, rebuilds, checks that every test still passes,
runs tests 1 to 3 and a few benchmarks, and prints one table.

Usage:  python3 scripts/sweep_geometry.py [--quick]

It restores the original constants on the way out, including after an
error or an interrupt.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENTITY = ROOT / "crates" / "tesseract-core" / "src" / "entity.rs"

# (SIDE, LEVELS). Sides 2, 4 and 8 rotate entirely through the fast paths;
# 16 and 32 have a middle axis whose digit straddles a word boundary and
# falls back to the slow reference rotation.
GEOMETRIES = [
    (2, 7),
    (4, 4),
    (8, 3),
    (8, 4),
    (16, 3),
    (8, 5),
]

BENCHES = [
    ("entity/match_count (one entity)", "match_count"),
    ("memory/forward, 4096 units, one token", "memory_forward"),
    ("attention/scores, 256 candidates, one head", "attention_scores"),
    ("entity/rotate axis 0", "rotate_inner"),
    ("entity/rotate axis 1", "rotate_second"),
]

# Task settings small enough that every geometry finishes quickly and fits
# in memory. The point is to compare geometries, not to reach the best
# score at each.
TASKS = [
    (
        "test1_majority.toml",
        ["--set", "task.subset=33", "--set", "task.held_out=2000", "--set", "budget.steps=40000"],
    ),
    (
        "test2_recall.toml",
        [
            "--set",
            "task.pairs=400",
            "--set",
            "model.units=512",
            "--set",
            "budget.steps=400",
            "--set",
            "budget.eval_every=400",
        ],
    ),
    (
        "test3_lookup.toml",
        ["--set", "task.held_out=1000", "--set", "budget.steps=5000"],
    ),
]


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def set_geometry(side, levels):
    text = ENTITY.read_text()
    text = re.sub(r"pub const SIDE: usize = \d+;", f"pub const SIDE: usize = {side};", text, count=1)
    text = re.sub(
        r"pub const LEVELS: usize = \d+;", f"pub const LEVELS: usize = {levels};", text, count=1
    )
    ENTITY.write_text(text)


def read_geometry():
    text = ENTITY.read_text()
    side = int(re.search(r"pub const SIDE: usize = (\d+);", text).group(1))
    levels = int(re.search(r"pub const LEVELS: usize = (\d+);", text).group(1))
    return side, levels


def measure(side, levels, quick):
    row = {"side": side, "levels": levels, "bits": side**levels}
    set_geometry(side, levels)

    tests = run(["cargo", "test", "--workspace", "--quiet"])
    row["tests"] = "pass" if tests.returncode == 0 else "FAIL"
    if tests.returncode != 0:
        row["note"] = (tests.stdout + tests.stderr).strip().splitlines()[-1][:120]
        return row

    build = run(["cargo", "build", "--release", "--quiet"])
    if build.returncode != 0:
        row["tests"] = "FAIL (release build)"
        return row

    for config, overrides in TASKS:
        name = config.split("_")[0]
        out = run(
            ["./target/release/tesseract", "run", f"configs/{config}", "--out", "/tmp/sweep"]
            + overrides
        )
        held = re.findall(r"held-out +([\d.]+)%", out.stdout)
        row[name] = float(held[-1]) if held else None
        if name == "test2":
            curve = dict(re.findall(r"noise_(\d+): +([\d.]+)%", out.stdout))
            row["test2_noise30"] = float(curve.get("300", "nan"))

    bench_cmd = ["cargo", "bench", "-p", "tesseract-core", "--"]
    if quick:
        bench_cmd += ["--warm-up-time", "0.3", "--measurement-time", "0.6", "--sample-size", "10"]
    else:
        bench_cmd += ["--warm-up-time", "0.5", "--measurement-time", "2", "--sample-size", "20"]
    bench = run(bench_cmd + ["entity/|memory/forward,|attention/scores"])
    text = bench.stdout + bench.stderr
    for label, key in BENCHES:
        # criterion prints: "<name>   time:   [low estimate high]"
        m = re.search(
            re.escape(label) + r"[\s\S]{0,200}?time:\s+\[\S+ \S+ ([\d.]+) (\w+)",
            text,
        )
        if m:
            value, unit = float(m.group(1)), m.group(2)
            ns = {"ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}[unit]
            row[key] = value * ns
    return row


def fmt(v, digits=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != v:
            return "-"
        return f"{v:.{digits}f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="shorter benchmark runs")
    args = parser.parse_args()

    original = read_geometry()
    rows = []
    try:
        for side, levels in GEOMETRIES:
            started = time.time()
            print(f"… side {side}, {levels} levels", flush=True)
            row = measure(side, levels, args.quick)
            row["seconds"] = round(time.time() - started)
            rows.append(row)
            print("  " + json.dumps(row), flush=True)
    finally:
        set_geometry(*original)
        print(f"geometry restored to side {original[0]}, {original[1]} levels", flush=True)

    head = [
        ("side", "n"),
        ("levels", "levels"),
        ("bits", "bits"),
        ("tests", "tests"),
        ("test1", "test1 %"),
        ("test2", "test2 %"),
        ("test2_noise30", "t2 @30% %"),
        ("test3", "test3 %"),
        ("match_count", "match ns"),
        ("memory_forward", "memory ns"),
        ("attention_scores", "attn ns"),
        ("rotate_inner", "rot0 ns"),
        ("rotate_second", "rot1 ns"),
    ]
    print()
    print(" | ".join(h for _, h in head))
    print(" | ".join("---" for _ in head))
    for row in rows:
        print(" | ".join(fmt(row.get(k)) for k, _ in head))

    out = ROOT / "sweep_geometry.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwritten: {out}")
    return 0 if all(r.get("tests") == "pass" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
