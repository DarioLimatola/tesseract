# Tesseract

Binary, vectorised, multi-dimensional learning engine. It learns, restructures and prunes itself, and can be assembled into a Transformer-like architecture. Priority: efficiency and scalability, accepting lower accuracy than floating-point networks. New engine inspired by Predatore V6.04, not a port.

Owner: Dario Limatola. Repository: `DarioLimatola/tesseract` (public by the owner's choice). Theoretical base: `DarioLimatola/Protocollo_Genetico_ita`.

## Language rules

- All code, identifiers, comments, commit messages and documents: **English**.
- Conversation with the owner: **Italian**. Italian for `tile` is "regione" (not "tessera").
- Style with the owner: brief, essential, objective.

## Source of truth

Design documents are in `docs/`, copies of the claude.ai Project "Tesseract" (copied 2026-11-17). The Project is authoritative; when a decision changes, update both. Reading order:

1. `tesseract-design-notes.md` (entry point; wins on conflicts)
2. `tesseract-integrated-plan.md` (2026 literature, the geometry sweep, the ordered plan)
3. `tesseract-roadmap.md`
4. `tesseract-memory-and-genes.md` (coordinates, references, Genes, the Dilemma)
5. `tesseract-data-structure.md`
6. `tesseract-language-direction.md`
7. `tesseract-test-plan.md`
8. `tesseract-engineering-guidelines.md`
9. `tesseract-prior-art-review.md` (reference)

Do not invent decisions: label anything not confirmed by the owner as **Proposed**.

Known superseded lines: guideline "64 examples per word" (a word holds one dimension of one example); monitor is a desktop window, not a web server; test 11 (next byte) is replaced by the language ladder; main risk 2 is revised by the Random Hierarchy Model result of 2026.

## Decided (do not reopen without the owner)

- Engine is separate from tests, tasks and UI. Any input enters through adapters. The engine exposes an API.
- Fundamental structure: n-dimensional, n meta-dimensions, recursive; **n = 8**. An entity is 8^4 = 4096 bits: byte 8 bits, dimension 64, domain 512. Scene and episode sit above it.
- **The geometry is a parameter.** `SIDE` and `LEVELS` at the top of `crates/tesseract-core/src/entity.rs` set the shape of the whole engine; everything else derives from them. 8 and 4 are the default and the measured choice, not an assumption.
- Count and compare, never multiply. No floating point in the core.
- Not a next-token predictor. Language emerges from a ladder of virtualisations. Referents are the engine's own entities, not a grid world.
- **Depth comes from the data structure**: the levels act as blocks, traversed bottom-up and then top-down; the downward pass tells a lower level what its code is used for. Not a stack trained by a backward error signal.
- **Memory is blocks joined by references, addressed by coordinates**: a complete coordinate reads a block, a partial one narrows the search, content is resolved by binary attention over the survivors.
- **An operation is an object of the memory: the Gene.** It can be found, compared, passed and executed; data and code differ only in how the engine reads the block.
- Structure policy (prune, merge, unite, connect, split, and the Dilemma) is one subsystem on one budget. Pruning is gentle; some parts are protected and never pruned.
- Monitor is a native desktop window (eframe/egui), no server.
- Tests must become practical: random bits show the mechanism, not its use.
- Must run on an ordinary PC first (mid-range gaming laptop, AVX2) and scale up. No dependence on one machine.
- Use any tool that makes the program smarter, more efficient, more adaptable. Integers are allowed wherever they raise capacity.

## Workspace (Rust, edition 2024, 1.95+)

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
cargo bench -p tesseract-core --bench kernels
cargo run -p tesseract-run -- run configs/<test>.toml
cargo run -p tesseract-monitor
cargo run -p tesseract-core --example geometry     # the current shape
python3 scripts/sweep_geometry.py                  # rebuild and measure at several shapes
```

Run fmt, clippy and test before every commit. `.cargo/config.toml` builds for the processor of the build machine (`target-cpu=native`); remove that file for a portable build.

## Engineering rules

- **Reference first:** every kernel has a slow obvious implementation; fast paths are property-tested against it on random inputs.
- **Determinism:** same seed gives the same bits for any thread count and any backend.
- **No hard-coded sizes:** derive everything from `SIDE`, `LEVELS`, `BITS`, `WORDS`, `DOMAINS`, `DOMAIN_BITS`, `WORDS_PER_DOMAIN`. A change must still pass the geometry sweep.
- **Thresholds belong to the default geometry:** the test plan's pass marks are asserted when `SIDE == 8 && LEVELS == 4`; elsewhere tests assert structural claims and the sweep collects the numbers.
- **Lints follow the compiler, so keep them clean but never let them gate.** `-D warnings` turns a new compiler's lints into errors with no change to the code, which is what broke every CI run up to 2026-11-17. The CI gates on formatting and tests, and runs clippy as an advisory job; fix what it reports in the next commit.
- **Measure before optimising:** no optimisation merges without a before and after benchmark number.
- **Unsafe** only in the vector-instruction module, each use documented. Forbidden elsewhere (the workspace sets `unsafe_code = "forbid"`; relaxing it needs the owner).
- **Errors** are typed values; library code does not panic on bad input.
- **Every run is reproducible:** run folder stores config, seed, engine version, source revision. Checkpoints carry a version number.
- **Tests:** held-out set, trivial baseline, pass threshold written before the first run.
- Small steps: each change ships with its test and, if speed matters, its benchmark.
- Every public item is documented. Design decisions go in the design notes.
- **Verify every edit landed.** Scripted string replacement fails silently when the file was reformatted, and a test that assumes the default geometry hides it. Check the file after editing it.

## Status (2026-11-17)

Branch `first-draft`, CI green. `main` holds the first draft plus `docs/` and this file, and is behind.

- 5 crates, 80 tests, fmt and clippy clean, green at six geometries: (2,7), (4,4), (8,3), (8,4), (16,3), (8,5).
- Tests 0 to 3 pass at the default geometry: majority 100% (threshold 99%), recall 100% (99%), lookup 100% (95%). Numbers in `README.md`.
- Geometry sweep measured, in `README.md` and `sweep_geometry.json`. Memory capacity is set by key width: a unit firing at vigilance `v` stays distinct from unrelated keys of `m` bits only while `v >= 0.5 + 2.5/sqrt(m)`; at 0.625 that needs `m >= 256`.
- Next: step A of `docs/tesseract-integrated-plan.md` section 4 — coordinates and references, then the scaling test.

## Main risks

1. Hard-selection attention with binary keys has no published precedent; the stacked case is untested.
2. Deep learning with integer-only state. Revised: local rules with layerwise self-supervised objectives match backpropagation on hierarchical data, and rules that approximate the global error locally fail — so the chosen route is the better-supported family, but our particular rule is unproven.
3. Learning roles and variables from episodes (language level 4) is an open research problem.
4. Turning dense signals (images) into entities is hard.
5. Several interacting learning processes are easy to overfit: every step needs a held-out test and a baseline.
6. Learning which path through memory, or which Gene, answers a question is program induction: discrete, no gradient, combinatorial.

## Subagents

In `.claude/agents/`: `core-engineer`, `task-designer`, `design-reviewer`, `prior-art-researcher`, `monitor-dev`. Use `design-reviewer` before merging any change to `tesseract-core`.
