# Tesseract — engineering guidelines (draft, 2026-11-16)

Status: proposed by Claude, awaiting the owner's confirmation. Design is in `tesseract-design-notes.md`, tests in `tesseract-test-plan.md`.

**Decided by the owner:** use every available tool that makes the program smarter, more efficient and more adaptable. "Standard library only" was a limit of one work environment (no package downloads), not a project rule.

## Workspace layout

One Rust workspace, one crate per responsibility. Dependencies point inwards: everything may depend on `core`, `core` depends on none of the others.

| Crate | Responsibility | External libraries (to be pinned when the workspace is created) |
|---|---|---|
| `tesseract-core` | bit tensors, units, blocks, learning, pruning, backends | as few as possible; threads and serialisation as optional features (rayon, serde) |
| `tesseract-codecs` | adapters: bytes, numbers, grids and images, audio | format readers as needed |
| `tesseract-tasks` | tasks, scoring, baselines | none expected |
| `tesseract-run` | command-line runner, config files, run folders | clap, toml, serde |
| `tesseract-monitor` | local server and the one-page monitor | axum, tokio |
| benches and tests | measurements and checks | criterion, proptest |
| later: `tesseract-py`, `tesseract-gpu` | Python bindings; GPU backend | pyo3; wgpu |

The core stays light on purpose: it then builds on any target (x86, ARM, WebAssembly, small boards), builds fast, and breaks less. This is a boundary, not a restriction on the project.

Library choices and versions are checked against their current state when the workspace is created. Portable SIMD in the Rust standard library appeared still unstable at the time of writing; stable alternatives exist (processor intrinsics in `std::arch`, and dispatch crates).

## Performance, in order of expected gain

1. **Data layout.** Contiguous, 64-byte-aligned arrays of 64-bit words; 32-bit integer indices; no pointers or hash maps on hot paths. Parameters and learning counters live in separate arrays, so inference touches only parameters.
2. **Batch on bits.** 64 examples per word.
3. **Cache blocking.** Tiles sized to the L2 cache; finish a tile for the whole batch before moving on.
4. **Vector instructions chosen at start-up.** Each kernel is written once over a "word" abstraction and compiled for 64, 256 and 512 bits.
5. **Threads over tiles and batch chunks.** Sums are integer, so results are bit-identical for any thread count.
6. **Lazy tiles.** Skip a tile whose inputs did not change.
7. **GPU** behind the same backend interface, later.

Rule: measure before optimising. Benchmarks exist from the first day; no optimisation is merged without a before and after number.

## Adaptability

- Small interfaces (traits) for: unit, block, learning rule, pruning policy, codec, task, backend. A new variant is a new implementation, selected in the config file.
- Everything about a run is in its config file; nothing is hard-coded in the engine.
- Checkpoint format carries a version number; older versions stay readable.
- The engine is reachable three ways: as a Rust library, through the local server, and later from Python.

## Practices

- **Reference first.** Every kernel has a slow, obvious implementation. Fast paths are tested against it on random inputs (property tests).
- **Determinism is tested:** same seed gives the same bits for any thread count and any backend.
- **Unsafe code** only inside the vector-instruction module, each use documented; forbidden everywhere else.
- **Errors** are typed values in libraries; library code does not panic on bad input.
- **Every run is reproducible:** the run folder stores config, seed, engine version and source revision.
- **Automatic checks on every change:** formatting, lints, tests, benchmarks.
- **Small steps:** each change comes with its test and, where speed matters, its benchmark.
- **Documentation:** every public item documented; design decisions recorded in the project notes.
- **Version control from the first commit.**

## What makes it smarter

Libraries do not; the learning rule does. Engineering contributes by making experiments cheap: swap a rule with one config line, sweep settings, compare runs in the monitor, and compare against conventional models trained outside the engine (for example with PyTorch) on the same task.

## Order of work for the first draft

1. Workspace skeleton, checks, empty benchmarks.
2. Core with reference implementations and tests (test 0 of the plan).
3. Learning rule, masks, protected bits, usage counters.
4. Tests 1 to 3 with runner and config files.
5. Monitor page.
6. Fast paths (vector instructions, threads), each against its benchmark.
