# Tesseract — roadmap: next steps (2026-11-17)

**Decided** = confirmed by the owner. **Proposed** = recommendation, not yet confirmed. Entry point of the project remains `tesseract-design-notes.md`; the ordered, merged plan is in `tesseract-integrated-plan.md`.

## Decided by the owner (2026-11-17)

- **Depth comes from the data structure.** The levels of the recursive structure (value, domain, entity, scene, episode) act as the blocks. This replaces "a stack of blocks over one state with a backward error signal" as the first route to depth.
- **The blocks communicate.** The structure is traversed from the bottom up and then from the top down. The downward pass is what tells a lower level what its code is used for.
- **Tests must become practical.** The first three tests show that the mechanism works on random bits, not that it is useful.

## How the two passes work (Proposed)

A **level** is the machinery that exists today (a memory block, later with an attention head) working on one level of the structure.

**Up (recognition).** A level receives an entity whose 8 domains each hold a 512-bit code from the level below (or from an adapter at the bottom). The unit that matches best is its symbol for that input. Every unit has a fixed 512-bit **signature**; the winner's signature becomes one domain of an entity one level up. Eight neighbours (in space, in time or in kind) fill its eight domains. Each level compresses 4096 bits to 512.

**Down (expectation).** A level that recognised a whole sends back what it expected: the address of its winning unit is the 8 signatures it expects from below. Each lower level compares the expected signature with its own winner:

| Case | What happens |
|---|---|
| Same | Confirmed: evidence for the unit; it is useful to the level above, so it is kept and can become protected |
| Different, and the expected unit was a close second | The level switches to it: context resolves an ambiguous input |
| Different and far | Surprise: the lower level keeps its answer; the upper level learns (new unit or adjusted prototype) |
| The lower level had no input | The expected signature is unfolded into bits: completion of the missing part |

**Tasks.** When a task has a right answer, the answer enters as the expectation of the top level. A target is "an expectation from above", so every level learns with the call that already exists (`learn(context, wanted bits, mask)`), and no error has to be carried through several levels at once.

**Passes.** One up, one down. Proposed addition: one more upward pass only through levels whose winner changed, with a fixed bound.

Related work: adaptive resonance theory (bottom-up and top-down match with a vigilance threshold; our recruiting rule is already of this family), predictive coding, the wake-sleep algorithm, target propagation, feedback in HTM, the forward-forward algorithm. The 2026 results that bear on this are in `tesseract-integrated-plan.md`, section 2.3.

**Risks.**

1. Hierarchies that learn only locally have historically done worse than networks trained on a global error — revised in 2026 for rules with layerwise self-supervised objectives; see the integrated plan.
2. Higher levels may need a unit per combination and grow without bound; uniting and the structure policy must hold this down.
3. 8:1 compression per level loses detail; the downward pass must be able to read it back.

## Phase 0 — Close the first draft

1. Push branch `first-draft`; run the checks on GitHub. **Done 2026-11-17**; CI green after the clippy fix.
2. Build and run on the owner's laptop. **Done.**
3. Owner's decision on the "Choices made while building" (design notes). Open.
4. Copy the project documents into `docs/` in the repository. **Done 2026-11-17**, confirmed by the owner.

## Phase 1 — Levels as communicating blocks

Build:

1. A signature per unit, and references (a domain holding a signature).
2. **Unite:** similar inputs vote into one prototype instead of each recruiting a unit.
3. A `Level` type with `up` and `down`, and a hierarchy of two levels, then three.
4. The usefulness signal: confirmed units gain evidence, unused ones go dormant (the structure policy that exists).
5. Monitor: one grid per level; the path of one input up and down.

Tests on hierarchical "parts and wholes" data. **Proposed: use the Random Hierarchy Model** rather than an invented generator (integrated plan, 2.3). Thresholds are proposals, to be fixed before the first run.

| Test | Question | Pass (proposed) | Compared with |
|---|---|---|---|
| H1 Wholes | Does a two-level hierarchy recognise noisy wholes, including unseen combinations of known parts? | 95% on held-out | one flat memory block on the raw bits; units and bits used by each |
| H2 Context | Does the downward pass resolve an ambiguous part? | 95% where the upward pass alone is at chance | the same hierarchy without the downward pass |
| H3 Completion | Is a hidden part rebuilt from the other seven? | 95% of its bits | the most frequent part |
| H4 Usefulness | Two parts differ in few bits but belong to different wholes: does feedback keep them apart? | 95% | without feedback the lower level unites them and is near chance |

H2 and H4 test the owner's claim directly. **Gate:** if they fail, the two-pass design is revised before anything is built on it.

## Phase 2 — Practical tests

| Test | Why | Reference |
|---|---|---|
| P1 Handwritten digits (binarised MNIST) | Real, structured data and a number that can be compared. An entity is naturally a 64 × 64 binary image: a value is an 8 × 8 patch. Run flat first, then with levels | 98.3–98.5% for logic-gate systems (prior art review). Proposed pass: 95% |
| P2 Typo-tolerant lookup on a real word list | The use closest to what already works. Needs a text adapter (letter groups as bits) | exact search by edit distance |
| P3 Completion of real records (a public table: hide a field, recall it) | Completion without labels on real data | most frequent value; a small decision tree |

The layout of an image in an entity has to make a domain a compact region; to be chosen and measured.

## Phase 3 — Scale and efficiency

Measured: a unit takes about 2 KB in use and about 19 KB while it learns (two counters of 16 bits per connection). 4096 units cost 137 µs per token on one core, limited by memory access.

1. **Counters only where learning happens:** a pool of counters for the units that are plastic now; settled units keep only their bits. Smaller counters (8 bits) where enough.
2. **Do not scan every unit:** check one domain first and stop early when the match cannot reach the threshold; later an index from signatures to units.
3. **Compact units:** a unit that uses two domains stores two domains.
4. Fast path chosen at start-up (replaces the build flag); threads over units; bit planes for the state.
5. **Model checkpoints:** save and load a trained model (today only the monitor's snapshots are saved).
6. **A measure of "more efficient":** accuracy against time, memory and energy, compared with a small conventional network, nearest neighbours and boosted trees on the same task.

Working goal (Proposed): one million units trainable within 8 GB on the laptop. Each change ships with a before and after number.

## Phase 4 — Time

1. Position by rotation in a task (test 4 of the test plan).
2. Scenes and episodes: sequences of entities as levels above the entity.
3. Each level predicts its own next code (local prediction), as a learning signal without labels.

## Phase 5 — Acting

1. Reward as an expectation from above at the end of an episode: units active in a won episode gain evidence, in a lost one lose it.
2. Grid chase with held-out maps (test 12). Reference from Predatore's own test: random 30%, greedy 75%.

Chess and video games come after this and need adapters for boards and screens.

## Phase 6 — Language ladder

Steps 1 and 2 of the language document (naming, relations along a dimension) on generated worlds of entities, using the levels of phase 1.

## Further ideas (Proposed, not scheduled)

- **Consolidation while idle:** replay stored episodes to unite, merge and prune without new input.
- **Wiring chosen among candidates** with integer counters (prior art review, correction 5).
- **Learned alignment** between different domains of query and key in attention (candidate rotations).
- **Several winners per level** instead of one, for inputs that are mixtures.
- **GPU backend** for large regular levels.

## Open questions for the owner

1. First practical test: digits (recommended) or typo-tolerant lookup?
2. Next-byte prediction on a small text as a benchmark only, never as the training method: allowed or excluded?
3. Exactly one upward and one downward pass, or a bounded repeat where a winner changed?
4. The "Choices made while building" in the design notes.
