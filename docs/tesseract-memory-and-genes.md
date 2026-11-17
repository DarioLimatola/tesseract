# Tesseract — memory, coordinates and Genes (2026-11-17)

**Decided** = stated by the owner. **Proposed** = recommendation, not yet confirmed. **Built** = exists in the first draft and is tested. Entry point of the project remains `tesseract-design-notes.md`.

## Theoretical base

The owner's theoretical base is `DarioLimatola/Protocollo_Genetico_ita` (public, CC BY 4.0, 2024/2025): "Protocollo Genetico — una architettura vettoriale binaria frattale per AI". Tesseract is the engineering of that architecture. What the document specifies, and where it stands here:

| Element of the Protocol | What it says | Status in Tesseract |
|---|---|---|
| Discrete vector space in Z, Z dimensions and Z metadimensions (tensors), recursive | The fundamental structure | **Built** as n = 8: value, domain, entity, scene, episode |
| **Point 0** (`v0 = T(0,0,…,0)`) | Immutable computational origin; no mutation may alter it; every vector must be able to reach and anchor to it; the logical and cognitive root | **Missing.** Natural reading: the root of the coordinate system — every coordinate is a path from v0 — and a block protected by construction |
| **Gene** | A FIFO stack holding a boolean vector of size N, a set of N instructions, and a Turing machine that processes data and instructions. Turing-capable, recursive, virtualisable, evolutionary through the Dilemma | **Missing.** Tesseract has no execution at all. This is the half of the Protocol that was never built |
| **Dilemma** | A vector or tensor across Z keeps two versions of itself: the base one in N and a modified one in −N. A non-FIFO change is never applied directly; a modified matrix is generated, compared, and if validated becomes official and is permuted | **Missing as a mechanism.** The hidden counters with hysteresis are a gradual analogue at the level of a single connection; the Dilemma works where counters cannot — a whole structure or a whole procedure |
| Fixed instructions as columns, instances or agents as rows; each instruction has sub-vectors of micro-instructions | A shared instruction alphabet across agents | Missing (follows the Gene) |
| **Bucket vectors** | Recursive granularity: a binary datum is split in two, recursively, so precision grows while staying binary. A byte as 2³ buckets | Partly: the thermometer code gives 65 levels in 64 bits and perfect ordering. The recursive form gives 2^k levels in k bits. See "Codes" below |
| Recursion across levels | In an LLM the first tensor analyses characters, the second words, the third connections of sentences and thoughts; upper levels linked to lower ones by vectors | **Decided** 2026-11-17 as the route to depth (design notes, roadmap phase 1) |
| Data and metadata distinction annulled | Lower levels (tokens) and higher levels of abstraction in one recursive structure | This is the basis of "a Gene is both data and code", below |
| Every parameter is 1 bit | Efficiency through number of parameters, not precision of each | **Built** (two bits per connection: sign and mask) |
| Qubit-like values 1, 0, −1 and superposition across dimensions | A reading of the ternary connection | **Built** in effect: a connection is +1, −1 or absent |
| Graph structure that expands and evolves | Memory as a map of tokens and phrases joined by binary vectors | Missing: references do not exist yet |
| External transformer | Open point in the document | Open here too |

Tesseract contributes what the Protocol does not specify: the learning rule (integer counters), the structure policy, the vigilance-and-recruiting memory, and measured kernels.

## Stated by the owner (2026-11-17): memory, coordinates and binary attention

**Memory as dense binary blocks joined by sparse references.** A complete coordinate — `episode → scene → entity → domain` — reads the corresponding block directly. A partial coordinate narrows the search. When the query describes a content instead, the head compares the remaining candidates on the known and relevant bits only.

**The network learns where to read and what to bring back.** Each head learns which domains to consult, compares keys with XOR and popcount, selects the candidates and combines the values with integer votes. The next level can use the retrieved bits, or follow a reference to an entity, a relation or a Gene. The task of the network is to learn which path through memory answers the question.

**An operation is an object of the memory.** The network can find it, compare it with others, pass it to another module and, when needed, execute it. That is the role of the Gene. The same operation can be executed at one level and become the object of a decision at the level above.

### Objects

| Object | What it holds |
|---|---|
| Entity | Domains and binary values, with a mask that separates known bits from missing ones (**built**) |
| Scene, episode, relation | References to other objects, organised along axes or by roles. An episode orders scenes in time. A relation binds, for example, who acts and on what |
| Neural module | Connections, attention masks and learned update rules. At run time it receives bits or references and produces new bits or references (**built** as memory block and attention head) |
| Gene | A reusable procedure with declared inputs and outputs. It can call elementary operations, neural modules and other Genes |

A reference identifies a block even when it is reachable by several paths: the same entity can appear in a scene, in an episode and in the category "red objects" without duplicating its bits.

### Worked example, as the owner gives it

"Which is the heaviest red object in the scene?" An attention head selects the colour domain to narrow the candidates. A later module reads the weight domain and calls a comparison Gene. The result is a reference to the chosen entity. A further level can follow it to read the object's name, use it as the argument of another operation, or store it as part of an episode.

If the system finds that the sequence works elsewhere, it saves it as a **parametric Gene** that receives a set of entities, a filter criterion and an attribute to compare. Evaluated with the Dilemma, the same Gene serves to find the heaviest among the red ones or the longest among the blue ones. Virtualisation, here, is making a whole path of operations callable as one unit.

Data and code stay apart only in how the engine reads the block. The Rust interpreter executes a procedure when it receives a reference to a Gene with compatible inputs. A head can read that same Gene as data, for instance to compare it with a variant.

## The scaling test

Stated by the owner: grow the memory — more irrelevant episodes and scenes — while the information needed to answer stays fixed. Compare coordinate search followed by binary attention against a full binary scan, measuring accuracy, candidates examined, bytes read and latency. It succeeds if quality holds on unseen combinations and the search work grows more slowly than the memory.

This is the best-posed test proposed in the project so far: it has a trivial baseline, it measures work and not only accuracy, and it tests the property that decides whether the design is worth anything.

**One warning on how to run it (Proposed).** If the right coordinate is handed to the system, sublinear search is true by construction — that is what an index is, and every database has one. The result would be a tautology. The test must therefore report three numbers on the same data:

1. **Given coordinate** — the upper bound: what the index can do at best.
2. **Learned coordinate** — the real measurement: the head must produce the partial coordinate from the question.
3. **Full scan** — the lower bound on work, the upper bound on candidates examined.

The claim of the project is the gap between 2 and 3, with 2 close to 1. The published figures in `tesseract-integrated-plan.md` section 2.1 give the same split from outside: 100% with the address given, about 46% when it must be produced.

## Genes: what is engineering and what is research

**Engineering, buildable now.** A Gene as a block with declared inputs and outputs; an interpreter in Rust that executes it when it receives a reference with compatible arguments; the same block readable as data by a head. Treating code as a first-class value is old and solid ground (Lisp's homoiconicity, closures as values). Nothing here is risky.

**Research, not proven.** Learning which Gene to call and in which order. This is program induction, and it has no cheap solution: the choice is discrete, so there is no gradient, and the search space grows combinatorially with the length of the path.

**The Dilemma is the owner's own answer to this**, and it is the right shape for it: keep a challenger version, compare, promote the winner. For whole procedures, where per-connection counters do not apply, generate-and-test is the mechanism that fits. The prior art review already records that no project found combines evolution with error-driven Boolean learning in one system — so Dilemma over Genes, counters inside modules, is exactly the point where Tesseract would be new.

Conditions for the Dilemma to work (Proposed):

- the task must give a **checkable answer**, so the comparison has a fitness signal;
- Genes must be **small and composable**, which is the regime where program search does work;
- it needs a **budget**, like pruning: a challenger doubles the memory of whatever is under challenge. The Dilemma therefore belongs to the structure policy, alongside prune, merge, unite, connect and split — one subsystem, one set of budgets.

**Prior art.** Checked 2026-11-17; see `tesseract-integrated-plan.md` section 2 for the figures.

- **Neural Turing Machines and Differentiable Neural Computers**: learned read and write heads with content-based *and* location-based addressing — the direct precedent for "learn where to read". Trained with gradients, hard to train, never scaled. The cautionary case.
- **DreamCoder, Stitch, babble**: library learning — the direct precedent for "save that path as a parametric Gene". Stitch formulates it as greedy minimum-description-length optimisation, which is a concrete way to *find* candidates; the Dilemma then decides whether to *adopt* them.
- **Genetic programming**: the Dilemma's family. Works on small programs; scales poorly.
- **Cognitive architectures** (Soar, ACT-R): memory of declarative facts plus procedures, with a selection mechanism.
- **Product-key addressing** (Sparse Delta Memory): an address split into parts looked up separately — a coordinate in all but name, and a ready-made technique for step G0.
- Vector symbolic architectures and sparse distributed memory, already in the review.

## What this changes about what Tesseract is

With Genes, Tesseract stops being a neural network and becomes **a memory, an interpreter, and a learned control over both**. This is a repositioning, not a criticism, but it changes what success means and what to compare against: cognitive architectures and program-induction systems, graph and vector databases, not only binary neural networks.

## Proposed order

Each step is worth having even if the next one fails.

| Step | Content | Why here |
|---|---|---|
| **G0** | References and coordinates: scene, episode and relation as real objects; a reference as a 512-bit signature; read by complete and partial coordinate; point 0 as the protected root. No learning | The substrate everything else needs. Testable at once |
| **G1** | The scaling test, with the three measurements above | Decides whether the design pays. Needs no new learning |
| **G2** | Genes as data and the Rust interpreter, with **hand-written** Genes. No learning | Makes "the heaviest red object" work end to end: the first demonstrable capability of the project |
| **G3** | The Dilemma over Genes, on tasks with checkable answers; parametric Genes discovered from paths that recur | The research step. Everything before it stands on its own |

Relation to the roadmap: G0 and G1 fit inside phase 1 (levels as communicating blocks), because a reference between levels and a reference between memory objects are the same mechanism. G2 and G3 are a new phase, after phase 2 (practical tests). The merged order is in `tesseract-integrated-plan.md` section 4.

## Codes: the recursive bucket (Proposed)

The Protocol's bucket vectors are a better use of bits than the thermometer code built today, but not in their pure form:

| Code | Levels in 64 bits | Ordering |
|---|---|---|
| Thermometer (**built**) | 65 | Perfect: distance = difference of levels |
| Pure recursive buckets | 2⁶⁴ | Breaks at every boundary: 0111 and 1000 are neighbours in value and opposite in bits |
| **Coarse plus fine, Proposed** | 65 × 65 ≈ 4225 over two dimensions | Preserved inside a coarse bucket; one boundary per scale instead of one per bit |

The third is the Protocol's recursion made robust: one dimension carries the coarse thermometer, the next carries the position inside that bucket. It costs a second dimension and buys about 65 times the precision.

## Open questions

1. How many levels does a coordinate have, and is the coordinate stored in the block (a block knows where it is) or only in the parents (a block is reached but does not know its address)?
2. Can a Gene modify memory, or only read and return? A read-only Gene is far easier to test and to roll back.
3. Does the Dilemma apply to single units too, or only to structures and Genes? Counters already cover units.
4. Earlier open questions, still open: first practical test (digits or typo-tolerant lookup); next-byte prediction as a benchmark only, allowed or excluded; exactly one upward and one downward pass or a bounded repeat.
