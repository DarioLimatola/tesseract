# Tesseract

Binary, vectorised, multi-dimensional learning engine. It learns, restructures and prunes itself, and can be assembled into a Transformer-like architecture. Priority: efficiency and scalability, accepting lower accuracy than floating-point networks. New engine inspired by Predatore V6.04, not a port.

Owner: Dario Limatola. Repository: `DarioLimatola/tesseract` (public by the owner's choice).

## Language rules

- All code, identifiers, comments, commit messages and documents: **English**.
- Conversation with the owner: **Italian**. Italian for `tile` is "regione" (not "tessera").
- Style with the owner: brief, essential, objective.

## Source of truth

Design documents live in the claude.ai Project "Tesseract" (`claude/tesseract-*.md`). Reading order:

1. `tesseract-design-notes.md` (entry point; wins on conflicts)
2. `tesseract-data-structure.md`
3. `tesseract-language-direction.md`
4. `tesseract-test-plan.md`
5. `tesseract-engineering-guidelines.md`
6. `tesseract-prior-art-review.md` (reference)

Copies of these documents belong in `docs/` in this repository once the owner confirms. Until then, do not invent decisions: label anything not confirmed as **Proposed**.

Known superseded lines: guideline "64 examples per word" (a word now holds one dimension of one example); monitor is a desktop window, not a web server; test 11 (next byte) is replaced by the language ladder.

## Decided (do not reopen without the owner)

- Engine is separate from tests, tasks and UI. Any input enters through adapters. The engine exposes an API.
- Fundamental structure: n-dimensional, n meta-dimensions, recursive; **n = 8**. Value 64 bits, domain 512 bits, entity 4096 bits (8^4), scene 32 KB, episode 2 MB.
- Count and compare, never multiply. No floating point in the core.
- Not a next-token predictor. Language emerges from a ladder of virtualisations (symbols, associations, sequences, schemas, language). Referents are the engine's own entities, not a grid world.
- Structure policy (prune, merge, unite, connect, split) is one subsystem built into the base structures. Pruning is gentle; some connections are protected and never pruned.
- Monitor is a native desktop window (eframe/egui), no server.
- Must run on an ordinary PC first (mid-range gaming laptop, AVX2) and scale up. No dependence on one machine.
- Use any tool that makes the program smarter, more efficient, more adaptable. Integers are allowed wherever they raise capacity.

## Workspace (Rust)

| Crate | Role |
|---|---|
| `tesseract-core` | Entities, units, blocks, attention, learning, structure policy. No dependencies except optional rayon. No floating point. |
| `tesseract-codecs` | Adapters: bytes, numbers (later grids, images, audio). |
| `tesseract-tasks` | Tasks, scoring, baselines, thresholds. |
| `tesseract-run` | CLI: `tesseract run <config.toml> [--set key=value]`; writes a run folder. |
| `tesseract-monitor` | Desktop window: Setup, Run, Neurons. |

Dependencies point inwards: everything may depend on `core`; `core` depends on nothing else.

## Commands

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo bench -p tesseract-core
cargo run -p tesseract-run -- run configs/<test>.toml
```

Run all of fmt, clippy and test before every commit.

## Engineering rules

- **Reference first:** every kernel has a slow obvious implementation; fast paths are property-tested against it on random inputs.
- **Determinism:** same seed gives the same bits for any thread count and any backend.
- **Measure before optimising:** no optimisation merges without a before and after benchmark number.
- **Unsafe** only in the vector-instruction module, each use documented. Forbidden elsewhere.
- **Errors** are typed values; library code does not panic on bad input.
- **Every run is reproducible:** run folder stores config, seed, engine version, source revision. Checkpoints carry a version number.
- **Tests:** every test has a held-out set, a trivial baseline, and a pass threshold written before the first run.
- Small steps: each change ships with its test and, if speed matters, its benchmark.
- Every public item is documented. Design decisions go in the design notes.

## Status (2026-11-17)

- First draft (5 crates, 78 tests, tests 0 to 3 passing) was written in an earlier session on branch `first-draft` but **never pushed**: the repo on GitHub contains only the initial commit (`LICENSE`). It was handed to the owner as a bundle and zip. Next action: owner supplies the bundle or pushes it; then review of the "Choices made while building".
- Suggested next steps: test 4 (position by rotation); stack of two blocks with backward signal (tests main risks 1 and 2); start-up CPU dispatch instead of `target-cpu=native`; engine-property tests (pruning curve, revival, forgetting, damage, throughput).

## Main risks

1. Hard-selection attention with binary keys has no published precedent; stacked case untested.
2. Integer-only learning is shown for one layer, not deep stacks.
3. Learning roles and variables from episodes (language level 4) is an open research problem.
4. Turning dense signals (images) into entities is hard.
5. Several interacting learning processes are easy to overfit: every step needs a held-out test and a baseline.

## Subagents

Defined in `.claude/agents/`: `core-engineer`, `task-designer`, `design-reviewer`, `prior-art-researcher`, `monitor-dev`. Use `design-reviewer` before merging any change to `tesseract-core`.
