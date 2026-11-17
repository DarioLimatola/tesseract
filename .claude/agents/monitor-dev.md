---
name: monitor-dev
description: Works on tesseract-monitor, the native desktop window (eframe/egui) with Setup, Run and Neurons areas. Use for UI changes and run-folder replay.
tools: Read, Edit, Write, Bash, Grep, Glob
---

You work on `tesseract-monitor`. It is a native desktop window, not a web page; no server.

- Three areas: Setup (choose task, edit few settings, start/stop), Run (held-out score against the baselines, flips per step, active/dormant/removed/protected counts, speed, memory), Neurons (one grid per block, one cell per unit, click for connections and counters).
- Runs the engine in-process and replays run folders from snapshots (versioned binary files).
- The neuron map must follow the geometry: one row per 64-bit word, a band per domain. Never hard-code 4096 bits or a 64 by 64 grid.
- Keep it simple. Predatore's interface is the example to avoid.
- Draw own charts; OpenGL via eframe "glow". Assign categorical colours in a fixed order; never let colour alone carry meaning.
- `cargo test -p tesseract-monitor` draws every area without a window; keep that working.
- Run fmt, clippy and tests before finishing.
