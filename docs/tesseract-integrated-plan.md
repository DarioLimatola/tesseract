# Tesseract — research findings and the integrated plan (2026-11-17)

**Decided** = confirmed by the owner. **Proposed** = recommendation. **Measured** = a number from a run in this repository. Entry point remains `tesseract-design-notes.md`.

This document answers two questions asked together: what does current research say about the pieces we are building, and how do all the ideas fit into one feasible order of work.

## 1. What was built and measured today

**The geometry is now a parameter.** Two constants at the top of `crates/tesseract-core/src/entity.rs` set the shape of the whole engine:

```rust
pub const SIDE: usize = 8;    // how many of each level fit in the one above
pub const LEVELS: usize = 4;  // axes inside one entity: 8^4 = 4096 bits
```

Everything else is derived. `SIDE` must be a power of two; an entity and a domain must each hold at least one 64-bit word. `cargo run -p tesseract-core --example geometry` prints the current shape; `python3 scripts/sweep_geometry.py` rebuilds at several shapes and measures them.

Rotations are now general: whole-word moves for outer axes, two masked shifts for an axis whose digit sits inside a word, and the slow reference path for a digit that straddles a word boundary (only `SIDE >= 16`).

**Measured, one core, six geometries.** Tests 1 to 3 with 33 relevant inputs, 400 stored pairs in 512 units, 5,000 lookup sequences.

| n | axes | entity | test 1 | test 2 | test 3 | match | memory block | attention | rotate inner | rotate 2nd axis |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 7 | 128 b | 100% | 4% | 100% | 0.8 ns | 4.5 µs | 0.35 µs | 7.9 ns | 7.8 ns |
| 4 | 4 | 256 b | 100% | 56% | 100% | 1.0 ns | 9.7 µs | 0.6 µs | 5.9 ns | 4.8 ns |
| 8 | 3 | 512 b | 100% | 99% | 100% | 1.3 ns | 11.2 µs | 1.0 µs | 5.6 ns | 5.1 ns |
| **8** | **4** | **4096 b** | **100%** | **100%** | **100%** | **8.9 ns** | **137 µs** | **3.1 µs** | **19.8 ns** | **17.1 ns** |
| 16 | 3 | 4096 b | 100% | 100% | 100% | 9.3 ns | 165 µs | 3.0 µs | 19.2 ns | **2941 ns** |
| 8 | 5 | 32768 b | 100% | 100% | 29% → 100%¹ | 80.7 ns | 1151 µs | 51.3 µs | 102.9 ns | 101.8 ns |

¹ 100% at 20,000 steps instead of 5,000.

Five conclusions, all new:

1. **Cost is linear in the bits.** Eight times the bits cost about nine times the time. No shape is cheaper per bit than another.
2. **Below about 512 bits an entity loses capacity without saving proportional time.** From 128 to 512 bits is four times the capacity for 1.6 times the time: fixed overheads dominate small shapes.
3. **Memory capacity is set by the width of the key, not by the number of units.** Two random keys of `m` bits disagree on about `m/2` bits with a spread of `sqrt(m)/2`, so a unit firing at `v` agreement stays distinct only while `v >= 0.5 + 2.5/sqrt(m)`. At the default vigilance 0.625 that needs `m >= 256`. Measured: 99% at 256-bit keys, 56% at 128, 4% at 64. **This is a design rule, not a tuning detail.**
4. **n = 16 buys nothing and costs the fast rotation**: same speed at the same 4096 bits, but its straddling axis rotates 173 times slower.
5. **The attention head needs training steps proportional to the entity size**, because it starts matching on every bit. Not a wall: 32768 bits reach 100% with four times the steps.

**n = 8 with 4 axes was an assertion in the design notes; it is now the only shape measured to be fast everywhere and wide enough to hold a memory.**

## 2. What current research says

Read 2026-11-17. Numbers are as the authors report them.

### 2.1 Discrete addressing is the part that survives

*The Orthogonality Constraint in Infinite-Context Architectures* (arXiv 2601.15313) argues that reliable memory needs orthogonal keys, while trained embeddings cluster similar items, so continuous addressing interferes more as the data gets semantically dense. Measured: a linear associative memory falls from 100% at 5 facts to 1.3% at 75; on real data (UCI Wine, ρ≈0.96) it reaches 0.02% at 10,000 items, and CIFAR-100 with CLIP 0.05% at 2,000. Hash-based discrete addressing stays at 100% in all of these. Three-hop reasoning with 100 distractors: neural 0%, discrete 100%.

**For us.** This is the strongest external support the coordinate design has. It also sharpens the warning already in `tesseract-memory-and-genes.md`: their 100% is the *given coordinate* case. Where the address has to be produced from an embedding instead, the same paper reports only **45.7% top-1 over 16,309 Wikipedia facts**. The hard half of our problem — producing the right partial coordinate from the question — is exactly the half their best number does not cover. The three measurements of the scaling test stand.

### 2.2 Conditional memory as a sparsity axis, with host-memory prefetch

DeepSeek's *Engram* (arXiv 2601.07372, January 2026) adds an n-gram lookup table to a Transformer: parameters that are never computed over, only looked up, so capacity grows at O(1) cost per token. Reported gains over an equal-FLOP baseline: MMLU +3.4, CMMLU +4.0, BBH +5.0, ARC-Challenge +3.7. They formulate a **U-shaped scaling law** trading neural computation against static memory, and note that **deterministic addressing lets the tables be prefetched from host memory with negligible overhead** — DeepSeek-V4.1-Flash carries about 196 billion n-gram parameters, offloaded to host memory by default.

**For us.** This validates two of the owner's ideas from outside the project:
- *memory as a separate axis from computation* — the Gene/memory split;
- *keep the working set in RAM and the rest in slow memory* (16 Nov). A production system does exactly this, and it works **because addressing is deterministic**. Our coordinates give the same property. It is the concrete argument for building coordinates before trying to scale.

### 2.3 Local learning rules do reach deep hierarchies — but only some of them

*Self-supervised local learning rules learn the hidden hierarchical structure of high-dimensional data* (arXiv 2605.18557, May 2026) studies the **Random Hierarchy Model (RHM)**, a synthetic grammar that builds data by recursively applying random production rules (introduced in arXiv 2307.02129, where the sample complexity is shown to be polynomial rather than exponential). Finding: local rules with **layerwise self-supervised objectives** learn the hidden hierarchy and are **as data-efficient as supervised backpropagation**. Rules of the other kind tested — Direct Feedback Alignment, which approximates error propagation with direct feedback — **fail** to learn the hidden structure. A companion result (*Learn from your own latents and not from tokens*, arXiv 2605.27734) shows that predicting a level's own latent codes recovers the hierarchy more efficiently than predicting tokens, with a strictly local rule.

**For us.** This materially changes main risk 2, which said hierarchies learning only locally have done worse than networks trained on a global error. That is now too pessimistic, with two conditions:

- the local objective must be **self-supervised and layerwise** — each level predicts its own codes. That is precisely the owner's two-pass design and the "local prediction" line of the language document;
- rules that merely approximate the global error locally do **not** work. So a backward-error path through the stack is not just hard for us, it is the kind that failed there too.

**It also gives us our phase-1 data.** The RHM is a published generator of exactly the "parts and wholes" structure the H-tests need, with known sample complexity and results to compare against. **Proposed: adopt the RHM instead of inventing a generator.**

### 2.4 Library learning: how a recurring path becomes a Gene

DreamCoder learns a library of reusable abstractions by alternating solving and compressing (wake / abstraction sleep / dream). Successors make the compression step principled and fast: **Stitch** formulates library induction as greedy minimum-description-length optimisation with optimality guarantees for one abstraction at a time; **babble** uses e-graphs and anti-unification to learn several abstractions at once.

**For us.** "The system finds that the sequence works elsewhere and saves it as a parametric Gene" is library learning, and Stitch's MDL formulation is a concrete, non-evolutionary way to do the *finding* — compress the corpus of solved paths and keep what pays for itself. The Dilemma then decides *whether to adopt* the candidate. Search and selection are separate jobs; the Dilemma is good at the second.

### 2.5 Tsetlin machines went deep, and how

The **Multi-Layer Tsetlin Machine** (2024/2025) stacks machines so one layer's clauses feed the next; the authors report that uncontrolled growth in the number of layers leads to overfitting. The **Graph Tsetlin Machine** (arXiv 2507.14874) goes deep differently: nested clauses over graph-structured input through **message passing**, supporting sequences, grids, relations and multimodality.

**For us.** The closest relative of the engine got depth through *message passing over a structure*, not through a stack over a flat vector. That is the same answer the owner reached. Worth reading properly before building phase 1.

### 2.6 Also noted

*Sparse Delta Memory* (arXiv 2607.07386) scales a recurrent state by three orders of magnitude at identical FLOPs using **product-key addressing** — an address split into parts looked up separately. That is a coordinate in all but name, and a ready-made technique for step A.

## 3. How the ideas fit together

One sentence: **addressing is the spine, and everything else hangs off it.**

| Piece | Role | Status |
|---|---|---|
| Entity, levels (`SIDE`, `LEVELS`) | the shape of a block of bits | **Built, now parametric and measured** |
| Coordinates and references | discrete addressing: reach a block, or narrow the set | Not built — step A |
| Memory block | content addressing: match-and-count over learned addresses | **Built** |
| Attention head | query a set of candidates on chosen domains | **Built** |
| Levels up and down | depth: each level recognises, the level above expects | Not built — step C |
| Gene and interpreter | an operation as an object that can be read or run | Not built — step F |
| Dilemma | adopt or reject a whole structure or procedure | Not built — step G |
| Structure policy | prune, unite, merge, connect, split, and the Dilemma, on one budget | Partly built |

The two addressing modes — by coordinate and by content — are the same two a Neural Turing Machine had. The 2026 result in 2.1 says the discrete one is what survives at scale, and the one in 2.2 says it is also what makes host-memory offload possible. So the coordinate half is not an optimisation to add later; it is the load-bearing part, and it is the cheapest thing left to build.

## 4. The plan, ordered by what each step de-risks

Each step is useful even if the next fails. Gates are places to stop and revise rather than continue.

| Step | What it is | Why now | Gate |
|---|---|---|---|
| **A. Coordinates and references** (G0) | scene, episode and relation as real objects; a reference as a 512-bit signature; read by complete and partial coordinate; point 0 as the protected root; product-key style split addressing | The spine. No learning, so nothing can go subtly wrong | Reading by coordinate is exact and O(1) |
| **B. The scaling test** (G1) | grow the memory with irrelevant episodes; measure accuracy, candidates examined, bytes read, latency, with **given coordinate / learned coordinate / full scan** | Decides whether the design pays, and needs no new learning | Work grows slower than memory, with the learned number close to the given one |
| **C. Two levels, up and down, on the RHM** (phase 1, H1–H4) | signatures, unite, `Level::up` and `Level::down`; the downward pass confirms, corrects or completes | The published RHM result says this class of rule can work; our version is untested | H2 (context resolves an ambiguous part) and H4 (feedback keeps two look-alike parts apart) |
| **D. Practical tests** (phase 2) | binarised digits; typo-tolerant lookup on a real word list | Random bits prove the mechanism, not its use | A number comparable with the literature |
| **E. Memory tiers** | working set in RAM, settled counters and dormant units on disk, driven by the coordinates | Engram shows this works *because* addressing is deterministic — so it comes after A | One million units trainable in 8 GB |
| **F. Genes as data plus interpreter** (G2) | hand-written Genes, no learning; "the heaviest red object" end to end | The first demonstrable capability of the project | It runs, and a head can read a Gene as data |
| **G. Finding and adopting Genes** (G3) | MDL compression over solved paths to *find* candidates; the Dilemma to *adopt* them | The research step; everything before it stands alone | A Gene found on one task helps on another |

Time is phase 4, acting is phase 5, language is phase 6 of the roadmap; they come after D at the earliest.

## 5. What I would not do, and why

- **Do not build Genes before coordinates.** A Gene's arguments are references; without addressing it has nothing to take or return.
- **Do not add the Dilemma while the counters are untested at depth.** Two learning mechanisms interacting is main risk 5; the gate at C is what tells us whether the counters hold up.
- **Do not pursue a language model.** Nothing measured here says that is reachable, and the route chosen deliberately rejects the objective that makes it work elsewhere.
- **Do not change `SIDE` from 8 without a reason from a measurement.** The sweep says 8 with 4 axes is the only shape that is fast everywhere and wide enough. Larger entities are a legitimate experiment; smaller ones are not.
- **Do not chase a backward error path through the stack.** The RHM result suggests that family is the one that fails; layerwise self-supervised objectives are the one that works.

## 6. Open questions for the owner

Carried over and still open:

1. First practical test: binarised digits (recommended) or typo-tolerant lookup?
2. Next-byte prediction on a small text as a benchmark only, never as the training method: allowed or excluded?
3. Exactly one upward and one downward pass, or a bounded repeat where a winner changed?
4. Does a coordinate live inside its block, or only in the parents?
5. Can a Gene modify memory, or only read and return?
6. Does the Dilemma apply to single units, or only to structures and Genes?

New:

7. **Adopt the Random Hierarchy Model as the phase-1 data generator?** It replaces a generator I would otherwise invent, and brings a published baseline.
8. The repository builds for the processor it is built on (`.cargo/config.toml`, `target-cpu=native`). Keep that, or move to choosing the fast path at start-up before phase 1?
