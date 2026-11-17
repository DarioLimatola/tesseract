# Tesseract — design notes and entry point (updated 2026-11-17, geometry parametric and measured)

**Decided** = confirmed by the project owner. **Proposed** = recommendation, not yet confirmed. **Built** = exists in the repository and is tested.

## Start here

- **Reading order:** this document; `tesseract-integrated-plan.md` (what the 2026 literature says, what the geometry sweep measured, and the ordered plan); `tesseract-roadmap.md`; `tesseract-memory-and-genes.md`; `tesseract-data-structure.md`; `tesseract-language-direction.md`; `tesseract-test-plan.md`; `tesseract-engineering-guidelines.md`; `tesseract-prior-art-review.md` (reference; superseded in places by the integrated plan, section 2).
- **Theoretical base:** `DarioLimatola/Protocollo_Genetico_ita` (public, CC BY 4.0). Tesseract is the engineering of that architecture. Element-by-element mapping in `tesseract-memory-and-genes.md`.
- **Where documents disagree, this one wins**, then the integrated plan, then the data structure and language documents. Known superseded lines:
  - engineering guidelines, performance item 2 ("64 examples per word"): a word now holds one dimension of one example;
  - engineering guidelines, crate table, `tesseract-monitor` ("local server and the one-page monitor", axum, tokio) and "reachable through the local server": the monitor is a desktop window, there is no server;
  - test plan, "Monitor: one page": one window; the three areas are unchanged;
  - test plan, test 11 (next byte on a text): replaced by the language ladder in the language document;
  - data structure document, the label "to be confirmed" on n = 8: confirmed, and now measured (see below);
  - prior art review, "local learning is an experiment, not the default" and main risk 2: revised by the Random Hierarchy Model result of 2026 (integrated plan, 2.3).
- **Repository:** `DarioLimatola/tesseract`, public by the owner's choice. Branch `first-draft` holds the work; CI green since 2026-11-17.
- **Next action:** owner's review of "Choices made while building" and of the open questions in the integrated plan; then step A (coordinates and references).
- **Language of the work:** all code and documents in English; conversation with the owner in Italian. In Italian the word for `tile` is "regione" (not "tessera").

## Goal

A binary, vectorised, multi-dimensional learning engine that learns, restructures and prunes itself, and that can be assembled into something similar to a Transformer. Priority is efficiency and scalability, accepting lower accuracy than conventional floating-point networks. It is a new engine inspired by Predatore V6.04, not a port of it.

## Decided

- Name: Tesseract.
- All code, identifiers, comments and documentation are written in English.
- The engine is separate from the tests and from any UI. Any kind of input can enter through adapters. The engine exposes an API.
- Tests and interface are simple: set up a test, monitor the test and its performance, look at the neurons. Predatore's interface is too complex and is not the model.
- **The monitor is local software, not a web page (owner, 2026-11-16).** Built as a native desktop window.
- It must run on an ordinary PC first (a mid-range gaming laptop is the reference) and scale to stronger hardware. It must not depend on one machine. GPU or many-core hardware is acceptable where it scales better.
- Use every available tool that makes the program smarter, more efficient and more adaptable.
- Integers and other data structures are allowed wherever they raise the capacity of a unit; bits are not a dogma.
- **Fundamental data structure:** n-dimensional structures grouped into n meta-dimensions, recursively, with **n = 8** as the default and the same n at every level. An entity is 8^4 = 4096 bits. Details in the data structure document.
- **The geometry is a parameter of the code (owner, 2026-11-17).** `SIDE` and `LEVELS` at the top of `crates/tesseract-core/src/entity.rs` set the shape of the whole engine; everything else is derived. Changing them and rebuilding is supported and tested. n = 8 with 4 axes stays the default, and is now the measured choice rather than an assertion.
- **Depth comes from the data structure (owner, 2026-11-17).** The levels of the recursive structure act as the blocks, traversed bottom-up and then top-down; the downward pass tells a lower level what its code is used for.
- **Memory is blocks joined by references, addressed by coordinates (owner, 2026-11-17).** A complete coordinate reads a block; a partial one narrows the search; a content query is resolved by binary attention over the survivors. The network learns which path through memory answers the question.
- **An operation is an object of the memory: the Gene (owner, 2026-11-17).** It can be found, compared, passed to another module and executed. Data and code differ only in how the engine reads the block. A recurring path of operations becomes a parametric Gene, evaluated by the Dilemma.
- **Tests must become practical (owner, 2026-11-17).** Tests on random bits show the mechanism, not usefulness.
- The target architecture is Transformer-like, but it does **not** learn by predicting the next token. Language comes from a ladder of virtualisations.
- Referents for language are the engine's own entities, not a grid world.
- A mechanism merges identical things, unites similar ones and creates connections between different things that occur together.
- Pruning is a foundation of the engine, built into its base data structures, and must not be aggressive.
- Some connections are fundamental and can never be pruned.
- First experiment: can a single attention head with binary keys learn a copy or lookup task? **Answered for the simplest form: yes.**

## What exists

Rust workspace, 5 crates, 80 tests passing, formatter and lints clean, green at six different geometries.

| Crate | Content |
|---|---|
| `tesseract-core` | Entity of `SIDE^LEVELS` bits with general axis rotations; codes for a dimension; integer state; ternary counting unit with Boolean sensitivity; memory block; attention head; learning counters; structure policy with the dormant state; deterministic random bits. No dependencies except optional threads. No floating point. |
| `tesseract-codecs` | Adapters for bytes and numbers. |
| `tesseract-tasks` | Tests 1 to 3 with held-out sets, baselines and thresholds. |
| `tesseract-run` | `tesseract run <config.toml> [--set key=value]`; run folder with config, `run.json`, metrics and snapshots. |
| `tesseract-monitor` | Desktop window: Setup, Run, Neurons. No server. |

Also: test 0 as property tests against slow references; criterion benchmarks; `scripts/sweep_geometry.py`; `examples/geometry.rs`; a CI workflow that gates on formatting and tests and runs clippy as an advisory job.

### Measured results at the default geometry (seed 1)

| # | Test | Held-out | Threshold | Baselines |
|---|---|---|---|---|
| 0 | Kernels against references; 1, 2, 3 and 8 threads | exact | exact | — |
| 1 | Majority of 65 hidden inputs among 4095, hidden signs, one unit | 100.00% after 10,000 examples | 99% | chance 50.5% |
| 2 | Recall of 1000 pairs, 10% key noise, 4096 units | 100.00% | 99% | nearest neighbour 100% |
| 3 | Lookup, 8 records, 16 values, 3 misleading domains, one head | 100.00% after 1,500 sequences | 95% | chance 6.2%; last value 19.7%; closest record 20.4% |

Speed, one core of a 2.1 GHz Xeon (cloud machine, not the reference laptop): match count over one entity 63 ns portable, 21 ns at AVX2 level, 11 ns native; memory block of 4096 units 270 / 121 / 114 µs per token; attention over 256 candidates 19 / 7.1 / 3.8 µs. The block is limited by memory access, not by counting.

Memory per unit of a memory block: about 2 KB in use and about 19 KB while learning. One million units: about 2 GB in use, about 19 GB to train as built.

### Measured across geometries (2026-11-17)

Full table and method in `tesseract-integrated-plan.md`, section 1. The five findings:

1. Cost is linear in the bits; no shape is cheaper per bit.
2. Below about 512 bits an entity loses capacity without saving proportional time.
3. **Memory capacity is set by the width of the key, not by the number of units**: a unit firing at vigilance `v` stays distinct from unrelated keys of `m` bits only while `v >= 0.5 + 2.5/sqrt(m)`. At the default 0.625 that needs `m >= 256`. Measured: 99% at 256-bit keys, 56% at 128, 4% at 64.
4. n = 16 buys nothing at the same bit count and costs a 173× slower rotation on its straddling axis.
5. The attention head needs training steps proportional to the entity size, because it starts matching on every bit. Not a wall.

### What the results do not show

- Test 3 is the simplest honest form of the question: each record is one entity, the head learns two masks rather than general projections, and the signal that trains the selection is computed inside one block. The stacked case is untested.
- Test 2's memory stores one pair per unit: a noise-tolerant content-addressed table, not a distributed memory. It does not compress.
- Only one block is trained at a time. The unit's Boolean sensitivity is implemented and tested, but no backward signal passes between blocks.
- The three tests were designed together with the mechanism and use random bits. Nothing yet shows an advantage over existing methods.

### Choices made while building (to be confirmed by the owner)

1. **One counter per connection, not one per parameter bit.** A signed counter decides both bits of a ternary connection with hysteresis; no reset after a flip. Reason: a ternary connection passes through "absent" between the two signs, and one counter expresses that directly (as in Tsetlin automata).
2. **Threshold stored as a bias** relative to half of the live connections, so it keeps its meaning when connections are pruned or revived.
3. **Zero in the state means "unknown".** A token is bits plus a mask of known positions.
4. **Growth is immediate, pruning is the policy's**: on a budget, never while the error is rising, never if protected.
5. **Memory block learns by recruiting**: revive a matching dormant unit, else commit a free one; a unit that fires and votes against the target is blamed.
6. **Attention head:** match mask starts full and is pruned; value mask starts empty. Selection is the single best candidate, trained by comparing the selected candidate with the one that would have helped most.
7. **Test 3 as records with misleading domains**, instead of separate key and value tokens.
8. **Unit dormancy is off by default.**
9. **Fast path by build flag** (`target-cpu=native`) as an interim; choice at start-up remains the plan.
10. **State as one small integer per position**; bit planes are a later fast path.
11. **Monitor draws its own charts** and uses OpenGL; snapshots are binary with a version number.
12. **The geometry is set by two constants and changed by rebuilding**, not by generics. Sides 2, 4 and 8 rotate entirely through fast paths; 16 and above fall back to the reference rotation on a straddling axis.
13. **The pass thresholds of the test plan are asserted at the default geometry only.** Elsewhere the tests check structural claims, and the sweep script collects the numbers.
14. **Lints do not gate the build.** `-D warnings` turns a new compiler's lints into errors with no change to the code; the CI gates on formatting and tests and runs clippy as an advisory job.

Not built from the first-draft scope: damping of the flip rate; separate masks per dimension, domain and entity; removal after long dormancy.

## Proposed

### Principle

Count and compare, never multiply. No floating point in the core. **Built.**

### Unit

- **Ternary counting unit.** Each input connection has a sign bit and a mask bit. Output = 1 when `popcount((x XNOR sign) AND mask) >= threshold`. **Built.**
- The mask bit is the pruning state of the connection. **Built.**
- Small lookup tables as a special unit for exact exceptions. Not built.

### Where integers live

1. Counts of matching bits and their thresholds. **Built.**
2. The state of a token: one small saturating integer per position. **Built** as one byte per position, limit ±31; bit planes later.
3. Output scores: one count per candidate answer. **Built.**
4. Hidden learning counters. **Built**, one per connection.
5. Usage and co-occurrence counters. Usage **built**; co-occurrence not.

### Levels as communicating blocks (the decided route to depth)

A level is the existing machinery working on one level of the structure. Upwards, the best-matching unit's signature becomes one domain of an entity one level up. Downwards, a level sends back the signatures it expected; the lower level is confirmed, corrected, surprised or completed. A task's answer enters as the expectation of the top level. Description, tests H1 to H4 and risks in the roadmap; the published result that supports this class of rule, and its conditions, in the integrated plan 2.3. Not built.

### Memory, coordinates and Genes

Blocks joined by references; coordinates as paths from point 0; binary attention over the candidates a partial coordinate leaves; Genes readable as data and executable as code; the Dilemma over whole structures and procedures. Full description and the order G0 to G3 in `tesseract-memory-and-genes.md`; the external evidence and the merged order in `tesseract-integrated-plan.md`. Not built.

### Tesseract block (Transformer-like)

1. **Input:** an adapter turns the input into an entity; position by rotation along an axis. Adapters and rotations **built**; no task uses position yet.
2. **State:** the identity path through the stack. **Built** for single blocks; no stack type yet.
3. **Attention as match-and-count over the context.** **Built** with best-one selection.
4. **Memory as match-and-count over learned addresses.** **Built.**
5. **Output:** the stored code closest to the sign bits of the final state. **Built.**

Attention and memory are the same operation with different keys. A stack of blocks over one state trained by a backward error signal is no longer the first route to depth; it stays an open alternative, and the integrated plan 2.3 gives a reason to doubt it.

Open: how the selection signal behaves when the head is not the last block; selection of several candidates; learned alignment between different domains of query and key.

### Learning

- **Error-driven:** Boolean signal per example; sensitivity of a unit is `out(x) XOR out(NOT x)`; signals summed into signed integer counts. **Built** within a block; not between blocks.
- **Parameter updates:** hidden integer counter with a threshold. **Built** without reset and without damping.
- **Initialisation:** blocks start writing nothing, so a stack starts as the identity. **Built.**
- **Without labels:** association by co-occurrence, completion, local prediction at each level, chunking. Not built.
- **The Dilemma** (Protocol): a challenger version of a whole structure or procedure, compared against the incumbent and promoted if it wins. Not built.

### Structure policy (one subsystem)

Pruning, merging, uniting, connecting, splitting — and the Dilemma — share counters, thresholds and budgets.

- **State in the base structures:** mask bit and protected bit per connection; usage counter and protected flag per unit, head, block and tile. **Built** at bit and unit level.
- **Signal that costs nothing extra:** the sensitivity computed for learning.
- **Three states:** active; dormant (masked, counters kept, revivable); removed only after long dormancy. Active and dormant **built**; removal not.
- **Gentle by rule:** a small budget per period; no pruning while error is rising; a soft state before anything irreversible. **Built.**
- **Protected:** point 0 and the coordinate root by construction; the state path by structure; input and output connections by role; minimums per unit and per block; anything pinned; long-consistent connections. **Built** except the first.
- Protected is not frozen: a separate flag stops learning. **Built.**
- **Merge, unite, connect, split, references:** not built. Unite and references come first.
- **Prune and regrow:** not built.
- **Compaction:** not built.

### Engineering

See the engineering guidelines with the corrections above. In short: a light core and separate crates; one data layout; plain 64-bit words as the portable baseline with faster paths; deterministic random bits; reference implementations and property tests; measure before optimising.

## Evidence so far

- Predatore V6.04: EVA reaches 88–100% capture in training but 20% on the held-out test, against 30% for random movement and 75% for a greedy baseline. Only 19 of 60 neurons of the trained brain are ancestors of the outputs.
- Voting on backward requests costs 1.2–5× a random pick in cache and 3–8.5× on a 4M-unit network; memory access dominates on large networks.
- First draft: tests 0 to 3, block costs, and the geometry sweep, above.
- Merge and connect simulation: similarity alone fragments families when variation is high; similarity plus a shared connection recovers them.
- Prior art: each proposed mechanism has a published precedent; the combination does not. The 2026 additions are in the integrated plan, section 2.

## Main risks

1. Training attention with hard selection and binary keys has no published precedent found. Test 3 passes in its simplest form; the stacked case is open.
2. **Deep learning with integer-only state.** Revised 2026-11-17. The published Random Hierarchy Model result (integrated plan 2.3) shows local rules with **layerwise self-supervised objectives** matching backpropagation in data efficiency on hierarchical data, which is the class the decided two-pass design belongs to. The same work shows that rules approximating the global error locally (Direct Feedback Alignment) **fail** — so the alternative route, a backward error path through a stack, now looks worse rather than better. The risk is no longer "local learning probably loses" but "our particular local rule is unproven". Tests H2 and H4 are the gate.
3. Learning roles and variables from episodes (language level 4) is an open research problem.
4. Turning dense signals such as images into entities is hard.
5. Several interacting learning and structure processes are hard to debug and easy to overfit. Coordinates, Genes and the Dilemma add three more; the order in the integrated plan exists so that each is tested alone.
6. **Learning which path through memory, or which Gene, answers a question is program induction**: discrete, no gradient, combinatorial. The closest precedent (Neural Turing Machines and successors) needed gradients and did not scale; the 2026 discrete-addressing work reaches only about 46% top-1 when the address must be produced from an embedding. The Dilemma is the proposed answer and is unproven at this job; library learning by MDL compression is the proposed way to generate candidates for it.

## Not yet explicitly confirmed by the owner

- Rust as the language (the first draft is written in it).
- The form of the desktop monitor.
- Thresholds of tests 1 to 3 as written in the test plan (all three were met).
- The choices listed under "Choices made while building", items 1 to 14.
- The two-pass mechanism, its tests and thresholds.
- The three measurements of the scaling test, the order of work, the coarse-plus-fine code.
- Adopting the Random Hierarchy Model as the phase-1 data generator.
- A measurable definition of "more efficient, less accurate".

## Next steps

`tesseract-integrated-plan.md`, section 4: A coordinates and references; B the scaling test; C two levels up and down on the RHM; D practical tests; E memory tiers; F Genes plus interpreter; G finding and adopting Genes. The roadmap keeps the longer horizon: time, acting, the language ladder.
