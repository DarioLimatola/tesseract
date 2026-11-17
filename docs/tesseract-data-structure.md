# Tesseract — the fundamental data structure and its structure operations (2026-11-16)

## Decided by the owner

- The fundamental data structure of the engine is an n-dimensional structure whose dimensions are grouped into n meta-dimensions (tensors), recursively.
- The same structures are the referents for language: no grid world.
- The engine needs a mechanism to merge identical dimensions, to unite similar ones, and to create connections between different things that belong together. The owner's example: the word "cat" and the picture of a cat.
- The repository is `DarioLimatola/tesseract` and it is public by the owner's choice.

This replaces the "Data model" bullets of `tesseract-design-notes.md` and item 2 ("batch on bits") of the performance list in `tesseract-engineering-guidelines.md`, as explained below. The design notes will be aligned once the reading below is confirmed.

## Proposed concrete form (to be confirmed by the owner): n = 8

Each level is "8 by 8 of the level below". With n = 8 every level coincides with a unit of the hardware.

| Level | Made of | Size | Meaning | Coincides with |
|---|---|---|---|---|
| Value | 8 x 8 bits | 64 bits | one dimension: one attribute of a thing | a processor word |
| Domain (meta-dimension) | 8 values | 512 bits | 8 related dimensions | a cache line, an AVX-512 register |
| Entity (the tesseract, 8^4 bits) | 8 domains | 4096 bits | one referent or record: 64 dimensions in 8 domains | 512 bytes |
| Scene | 8 x 8 entities | 32 KB | entities present together: associations | order of the L1 cache |
| Episode | 8 x 8 scenes | 2 MB | scenes in order: sequences and events | order of the L2 cache |

An entity is literally a four-dimensional hypercube of side 8, which gives the project name a concrete meaning. Predatore already used an 8 x 8 shape for its 64-bit vectors.

The side n can stay a parameter of the type; 8 is the default because of the hardware fit.

## Coding a dimension in 64 bits

- **Ordered quantities:** thermometer code, value v is v ones. 65 levels.
- **Unordered categories:** a fixed random or sparse code per category.
- **Free bits:** 64 independent binary features.

## Operations and their cost

- Distance along one dimension: `popcount(a XOR b)` on one word.
- "Greater than" along one dimension (thermometer): `(a AND NOT b) != 0`.
- Similarity within a domain: 8 words; within an entity: 64 words.
- Rotation along an axis: bit rotation in a byte, byte rotation in a word, word rotation in a domain, domain rotation in an entity. All are permutations and are their own transposes for the backward pass.
- Masks can be per bit, per dimension, per domain or per entity, so pruning can remove whole dimensions or domains at once.

## Structure operations: merge, unite, connect (proposed)

Two measurements decide what happens to two things. Both are counts.

- **Similarity:** how many bits agree, per domain.
- **Co-occurrence:** how often the two are present in the same scene or episode.

| | Often together | Rarely together |
|---|---|---|
| **Identical** | merge into one | merge into one |
| **Similar** | unite into one prototype | unite only if they also share a connection |
| **Different** | connect | nothing |

- **Merge (identical):** one copy is kept, references are redirected, evidence counters are added. Applies to entities, to units, and to dimensions that always carry the same value.
- **Unite (similar):** members vote bit by bit into a prototype (the hidden counters already do this). The prototype is the symbol for the family.
- **Connect (different, together):** a co-occurrence counter passes a threshold and a link is created. When a group stays linked, a new entity is created one level up whose domains hold references to the members. This is the virtualisation step: the concept "cat" is an entity with a "name" domain and an "appearance" domain.
- **Reference:** a domain can hold a 512-bit signature of another entity; following it is a nearest-match lookup. Entities that point to entities are the recursive tensors of the design.
- **Split:** the reverse of unite, when members of a prototype start to differ in what they are connected to.

Safeguards, shared with pruning: a small budget per period; a soft state first (alias: both kept and linked) and a hard merge only after long agreement; protected items are never merged or split automatically; a merged item keeps the protection of either parent.

Pruning, merging, uniting, connecting and splitting are one subsystem ("structure policy") with the same counters, thresholds and budgets. Slots freed by pruning and merging are the ones reused for new symbols and connections.

### The owner's example, step by step

1. Many pictures of cats are similar: they are united into one visual prototype.
2. Every occurrence of the word "cat" is identical: merged into one symbol.
3. The prototype and the word occur together: they are connected, then a concept entity is created above them.
4. A new picture of a cat finds the nearest prototype and, through the connection, the word. The word finds the prototype.

### Simulation (Python, synthetic data, 512-bit codes)

100 concepts, each a random visual prototype and a random word code. 2000 training events, each one noisy picture (every bit flipped with the given probability) with its exact word. 1000 unseen noisy pictures for the test. "Symbols" is the number of visual symbols formed (the true number is 100).

| Noise | Method | Symbols | Picture to word | Word to prototype (at least 90% of bits right) |
|---|---|---|---|---|
| 10% | similarity only | 100 | 100.0% | 100/100 |
| 20% | similarity only | 101 | 100.0% | 100/100 |
| 30% | similarity only | 1476 | 99.8% | 18/100 |
| 30% | similarity plus shared word | 100 | 100.0% | 99/100 |
| 35% | similarity only | 1993 | 71.7% | 0/100 |
| 35% | similarity plus shared word | 100 | 99.3% | 72/100 |

Reading: similarity alone breaks down as variation grows (the family fragments into many symbols). Using the connection as a second measurement ("similar and connected to the same word: same kind") restores the right number of symbols. The word teaches the categories.

Limits of this evidence: the variation is independent random bit flips. Real pictures of the same thing differ in structured ways (pose, light, background) and are not close in raw bits, so for real images the features must come from the lower levels or from an adapter.

## Consequences for the block design

- The state of a token is one entity (4096 positions) instead of 1024.
- An attention head matches on chosen domains or dimensions: "find the entity whose domain k is close to this one". Retrieval by attribute is binding by attribute.
- A memory unit has a ternary mask over the entity; cost grows about four times against the 1024-bit measurement (roughly 180 µs per block per token on one core, by extrapolation, not measured).
- **Layout:** a word holds one dimension of one example. Counting units need the bits of one example side by side, so the earlier guideline "64 examples per word" applies only to small-gate units. The batch becomes an outer array axis: vector instructions across the words of an example, threads across examples.

## Referents: a world of entities

A referent is a point in the 64-dimensional space of an entity. Scenes are sets of entities; episodes are scenes in order.

| Word kind | Refers to |
|---|---|
| value name ("red") | a value on one dimension |
| region name ("big") | a range on one dimension, or a region in a domain |
| relation ("bigger than", "left of") | a comparison of two entities along one dimension |
| event ("grew", "moved") | a difference between two scenes along some dimensions |

Generated worlds give exact ground truth, adjustable difficulty (number of dimensions, domains, entities) and the tests that matter for language:

- **Unseen combinations:** train on "red big" and "blue small", test on "red small".
- **Transfer along the dimension axis:** a relation such as "more than" learned on some dimensions must work on a dimension where it was never trained. This is where a rule shared along an axis pays off.

A grid is a special case: two of the dimensions are positions.

## Caveats

- Compared with a grid of cells, this object-centred form is more compact when there are few things with many attributes. Dense signals (images, sound) still arrive as grids or sequences and must be turned into entities by adapters; that step is the symbol level of the language ladder and is hard.
- 65 levels per ordered dimension; finer quantities need two dimensions or a different code.
- A fixed 4096-bit entity wastes space for small things; the uniform size is what makes vector instructions and caches work well.
- Cache sizes vary by processor; the L1 and L2 correspondences are orders of magnitude.
