# Tesseract — design notes and entry point (updated 2026-11-17, first draft written)

**Decided** = confirmed by the project owner. **Proposed** = recommendation, not yet confirmed. **Built** = exists in the first draft and is tested.

## Start here

- **Reading order:** this document; `tesseract-data-structure.md`; `tesseract-language-direction.md`; `tesseract-test-plan.md`; `tesseract-engineering-guidelines.md`; `tesseract-prior-art-review.md` (reference).
- **Where documents disagree, this one wins**, then the data structure and language documents. Known superseded lines:
  - engineering guidelines, performance item 2 ("64 examples per word"): a word now holds one dimension of one example;
  - engineering guidelines, crate table, `tesseract-monitor` ("local server and the one-page monitor", axum, tokio) and "reachable through the local server": the monitor is a desktop window, there is no server;
  - test plan, "Monitor: one page": one window; the three areas are unchanged;
  - test plan, test 11 (next byte on a text): replaced by the language ladder in the language document;
  - data structure document, the label "to be confirmed" on n = 8: confirmed.
- **Repository:** `DarioLimatola/tesseract`, public by the owner's choice. The first draft is on branch `first-draft` (3 commits on top of the initial commit). The session that wrote it could not push: Claude had no GitHub access to the repository. The branch was handed to the owner as a bundle and a zip.
- **Next action:** owner pushes or grants access; then the owner's review of the first draft and the choices listed under "Choices made while building".
- **Language of the work:** all code and documents in English; conversation with the owner in Italian. In Italian the word for `tile` is "regione" (not "tessera").

## Goal

A binary, vectorised, multi-dimensional learning engine that learns, restructures and prunes itself, and that can be assembled into something similar to a Transformer. Priority is efficiency and scalability, accepting lower accuracy than conventional floating-point networks. It is a new engine inspired by Predatore V6.04, not a port of it.

## Decided

- Name: Tesseract.
- All code, identifiers, comments and documentation are written in English.
- The engine is separate from the tests and from any UI. Any kind of input can enter through adapters. The engine exposes an API.
- Tests and interface are simple: set up a test, monitor the test and its performance, look at the neurons. Predatore's interface is too complex and is not the model.
- **The monitor is local software, not a web page (owner, 2026-11-16).** The owner does not intend to make Tesseract a web app for now. Built as a native desktop window.
- It must run on an ordinary PC first (a mid-range gaming laptop is the reference) and scale to stronger hardware. It must not depend on one machine. GPU or many-core hardware is acceptable where it scales better.
- Use every available tool that makes the program smarter, more efficient and more adaptable.
- Integers and other data structures are allowed wherever they raise the capacity of a unit; bits are not a dogma.
- **Fundamental data structure:** n-dimensional structures grouped into n meta-dimensions, recursively, with **n = 8** as the default and the same n at every level. An entity is 8^4 = 4096 bits. Details in the data structure document.
- The target architecture is Transformer-like, but it does **not** learn by predicting the next token. Language comes from a ladder of virtualisations, from symbols and associations up to language. Details in the language document.
- Referents for language are the engine's own entities, not a grid world.
- A mechanism merges identical things, unites similar ones and creates connections between different things that occur together (the word "cat" and the picture of a cat).
- Pruning is a foundation of the engine, built into its base data structures, and must not be aggressive.
- Some connections are fundamental and can never be pruned.
- First experiment: can a single attention head with binary keys learn a copy or lookup task? **Answered for the simplest form: yes** (see "First draft").

## First draft (built 2026-11-16/17)

### What exists

Rust workspace, 5 crates, 78 tests passing, formatter and lints clean.

| Crate | Content |
|---|---|
| `tesseract-core` | Entity (8^4 bits) with rotations along the 4 axes; codes for a dimension (thermometer, category); integer state; ternary counting unit with Boolean sensitivity; memory block; attention head; learning counters; structure policy with the dormant state; deterministic random bits. No dependencies except optional threads (rayon). No floating point. |
| `tesseract-codecs` | Adapters for bytes (512-bit category code in a domain) and numbers (thermometer in a dimension). |
| `tesseract-tasks` | Tests 1 to 3: generators, held-out sets, baselines, thresholds. |
| `tesseract-run` | `tesseract run <config.toml> [--set key=value]`; run folder with `config.toml`, `run.json`, `metrics.jsonl`, `snapshots/`. |
| `tesseract-monitor` | Desktop window (eframe/egui, OpenGL): Setup, Run, Neurons. Runs the engine in-process, replays run folders. No server. |

Also: test 0 as property tests against slow references; criterion benchmarks; a CI workflow (written, never run: the branch was not pushed).

### Measured results (configs in `configs/`, seed 1)

| # | Test | Held-out result | Threshold | Baselines |
|---|---|---|---|---|
| 0 | Kernels against references; 1, 2, 3 and 8 threads | exact | exact | — |
| 1 | Majority of 65 hidden inputs among 4095, hidden signs, one unit | 100.00% after 10,000 examples | 99% | chance 50.5% |
| 2 | Recall of 1000 pairs, 10% key noise, 4096 units | 100.00% | 99% | nearest neighbour 100% |
| 3 | Lookup, 8 records, 16 values, 3 misleading domains, one head | 100.00% after 1,500 sequences | 95% | chance 6.2%; last value 19.7%; closest record over all domains 20.4% |

- Test 1: the final mask is exactly the hidden subset with the hidden signs (checked bit by bit). Passes also with subsets of 9, 33, 129, 257 and seeds 2 to 5.
- Test 2: exact up to 30% noise, 99.3% at 35%, 1.4% at 40% (a unit fires when 62.5% of its inputs agree, so the cliff is set by that setting). Capacity equals the number of units: 4096 pairs exact, 6000 pairs 68.3%, 8192 pairs 50.0%.
- Test 3: the head ends matching the whole key domain, none of the misleading domains, and copying exactly the value domain. Passes also with 0 and 6 misleading domains, 16 records with 256 values, other domain positions, seeds 2 to 4.

Speed, one core of a 2.1 GHz Xeon (cloud machine, not the reference laptop):

| Operation | Portable | AVX2 level | Native (AVX-512) |
|---|---|---|---|
| Match count over one entity | 63 ns | 21 ns | 11 ns |
| Memory block, 4096 units, one token | 270 µs | 121 µs | 114 µs |
| Attention scores, 256 candidates, one head | 19 µs | 7.1 µs | 3.8 µs |

The earlier extrapolation for the memory block (about 180 µs) is replaced by these measurements. The block is limited by memory access (4 MB of signs and masks per token), not by counting.

### What the results do not show

- Test 3 is the simplest honest form of the question. Each record is one entity (no position needed); the head learns two masks (where to match, what to copy), not general projections; the signal that trains the selection is computed inside one block from the error at its own output. Whether it survives a stack of blocks is untested. Main risk 1 is reduced, not removed.
- Test 2's memory stores one pair per unit: a noise-tolerant content-addressed table, not a distributed memory. It does not compress.
- Only one block is trained at a time. The unit's Boolean sensitivity is implemented and tested, but no backward signal is passed between blocks yet. Main risk 2 (depth) is untouched.

### Choices made while building (to be confirmed by the owner)

1. **One counter per connection, not one per parameter bit.** A signed counter decides both bits of a ternary connection with hysteresis: dormant below a threshold, live with the sign of the evidence above it; no reset after a flip. The proposal below said "one counter per parameter bit, flip then reset". Reason: a ternary connection passes through "absent" between the two signs, and one counter expresses that directly (as in Tsetlin automata).
2. **Threshold stored as a bias** relative to half of the live connections (`matches - mismatches >= bias`), so it keeps its meaning when connections are pruned or revived.
3. **Zero in the state means "unknown".** A token is read as bits plus a mask of known positions. Attention counts agreement only where both tokens are known. Adapters mark what they wrote as known.
4. **Growth is immediate, pruning is the policy's.** Learning can make a connection live at once; only the structure policy makes one dormant: on a budget per period, never while the error is rising, never if protected, keeping a minimum of live inputs.
5. **Memory block learns by recruiting.** When no unit helps, a dormant unit that matches is revived, otherwise a free unit takes the input as its address and the wanted change as its value. A unit that fires and votes against the target is blamed: its address learns not to fire there.
6. **Attention head:** match mask starts full and is pruned; value mask starts empty (a new head writes nothing). Selection is the single best candidate. Training of the selection: the error says which candidate would have reduced it most; when that is not the one selected, each match connection on which the two differ gets one unit of evidence for the useful one.
7. **Test 3 as records with misleading domains**, instead of separate key and value tokens (that form needs position and belongs with test 4).
8. **Unit dormancy is off by default.** With store-once data a usage rule forgets stored pairs; the mechanism exists and is tested.
9. **Fast path by build flag** (`target-cpu=native`) as an interim; choice at start-up remains the plan.
10. **State as one small integer per position** (reference layout); bit planes are a later fast path.
11. **Monitor draws its own charts** and uses OpenGL (eframe "glow"); snapshots are binary files with a version number (postcard).

Not built from the first-draft scope: damping of the flip rate; separate masks per dimension, domain and entity (bit masks cover them); removal after long dormancy.

## Proposed

### Principle

Count and compare, never multiply. Integers are produced by counting bits and consumed by comparisons that give bits back. No floating point in the core. **Built.**

### Unit

- **Ternary counting unit.** Each input connection has a sign bit and a mask bit (present or absent). Output = 1 when `popcount((x XNOR sign) AND mask) >= threshold`. Over one entity that is 8192 parameter bits per unit, against 4 for a two-input gate. **Built.**
- The mask bit is the pruning state of the connection. **Built.**
- Small lookup tables remain available as a special unit for exact exceptions; they memorise and do not generalise by similarity. Not built.

### Where integers live

1. Counts of matching bits and their thresholds. **Built.**
2. The state of each token: one small saturating integer per position of an entity (4096 values of about 6 bits), stored as bit planes. Blocks read its sign bits and write signed votes into it. **Built** as one byte per position, limit ±31.
3. Output scores: one count per candidate answer. **Built.**
4. Hidden learning counters. **Built**, one per connection (choice 1).
5. Usage and co-occurrence counters for the structure policy. Usage **built**; co-occurrence not.

### Tesseract block (Transformer-like)

1. **Input:** an adapter turns the input into an entity; position is added by rotation along an axis. Adapters and rotations **built**; no task uses position yet.
2. **State:** the identity path through the stack (a block that writes nothing changes nothing). **Built** for single blocks; no stack type yet.
3. **Attention, as match-and-count over the context:** a head matches on chosen domains or dimensions; the score of a pair is a popcount of agreeing bits; entities above a threshold, or the best few, are selected; their value bits are added to the state as votes. It retrieves and binds; it does not score a next token. **Built** with best-one selection.
4. **Memory, as match-and-count over learned addresses:** a layer of ternary counting units; each unit that fires adds its ternary value vector to the state. **Built.**
5. **Output:** the stored code closest to the sign bits of the final state. **Built.**

Attention and memory are the same operation with different keys: other entities in the context in one case, learned addresses in the other.

Open: how the selection signal behaves when the head is not the last block; selection of several candidates; learned alignment between different domains of query and key (candidate rotations with integer counters).

### Learning

- **Error-driven:** Boolean signal per example; sensitivity of a unit to an input is `out(x) XOR out(NOT x)`; signals from several consumers and examples are summed into signed integer counts. **Built** within a block; not between blocks.
- **Parameter updates:** hidden integer counter; flip when it passes a threshold; flip rate damped over training. **Built** without reset and without damping (choice 1).
- **Initialisation:** blocks start writing nothing, so the stack starts as the identity. **Built.**
- **Without labels:** association by co-occurrence, completion of partly hidden records, local prediction at each level, chunking. See the language document. Not built.

### Structure policy (one subsystem)

Pruning, merging, uniting, connecting and splitting share counters, thresholds and budgets.

- **State in the base structures:** every connection has a mask bit and a protected bit; every unit, head, block and tile has a usage counter and a protected flag. Masks also exist per dimension, per domain and per entity. **Built** at bit and unit level.
- **Signal that costs nothing extra:** the sensitivity computed for learning already says whether a part could have changed the result.
- **Three states for pruning:** active; dormant (masked, counters kept, can be revived); removed only after a long dormancy. Active and dormant **built**; removal not.
- **Gentle by rule:** a small budget per period; no pruning while error is rising; a soft state before any irreversible step. **Built.**
- **Protected (never pruned, merged or split automatically):** the state path, by structure; input and output connections, by role; a minimum of live inputs per unit and live units per block; anything pinned in the configuration; connections whose evidence stayed consistent for a long time. **Built:** protected bits, minimums, pinned units, protection by saturated evidence (off unless set).
- Protected is not frozen: a separate flag stops learning. **Built.**
- **Merge, unite, connect, split, references:** see the data structure document, including the simulation of the word and picture example. Not built.
- **Prune and regrow:** freed slots are reused for new candidates, symbols and connections. Not built (free units are recruited; pruned ones are not reused).
- **Compaction:** active parts are kept contiguous in memory at checkpoints. Not built.

### Engineering

See the engineering guidelines, with the corrections listed under "Start here". In short: Rust workspace with a light core and separate crates for adapters, tasks, runner and monitor; one data layout; plain 64-bit words as the portable baseline with faster paths; deterministic random bits; reference implementations and property tests; measure before optimising.

## Evidence so far

- Predatore V6.04, bundled experiment: EVA reaches 88–100% capture in training but 20% on the held-out test, against 30% for random movement and 75% for a greedy baseline. Only 19 of 60 neurons of the trained brain are ancestors of the outputs. About 1.2–1.7 µs per neuron per tick in Python.
- Voting on backward requests costs 1.2–5x a random pick in cache and 3–8.5x on a 4M-unit network; memory access dominates on large networks.
- First draft: tests 0 to 3 and block costs, above.
- Merge and connect simulation (data structure document): similarity alone fragments families when variation is high; similarity plus a shared connection recovers them.
- Prior art: each proposed mechanism has a published precedent; the combination does not.

## Main risks

1. Training attention with hard selection and binary keys has no published precedent found; the binary BERT in the literature needed a floating-point teacher. Test 3 passes in its simplest form; the stacked case is open.
2. Training with integer-only state is shown for one-layer systems (Tsetlin machines), not for deep stacks. The first draft is also one layer.
3. Learning roles and variables from episodes (level 4 of the language ladder) is an open research problem.
4. Turning dense signals such as images into entities is hard; the simulation used synthetic variation.
5. Several interacting learning and structure processes are hard to debug and easy to overfit. Every step needs a held-out test and a trivial baseline.

## Not yet explicitly confirmed by the owner

Working choices unless the owner says otherwise:

- Rust as the language (the first draft is written in it).
- The form of the desktop monitor: one window with the three areas side by side, native window through eframe/egui.
- Thresholds of tests 1 to 3 as written in the test plan (all three were met).
- The choices listed under "Choices made while building".
- A measurable definition of "more efficient, less accurate" (not blocking).

## Suggested next steps

1. Test 4 (copy the token k positions back): position by rotation, and key and value as separate tokens.
2. A stack of two blocks with the backward signal between them: the first real test of risks 1 and 2.
3. Fast path chosen at start-up, replacing the build flag; bit planes for the state.
4. Engine-property tests of the plan on tests 1 to 3: pruning curve, revival, forgetting, damage, throughput.

Still out: merge, unite, connect and split; references and concept entities; removal and regrowth; tiles of tiles; the language ladder; GPU backend; image and audio adapters; Python bindings.
