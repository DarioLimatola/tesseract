# Tesseract — language direction (2026-11-16, reading confirmed by the owner)

## Decided by the owner

- The Transformer-like architecture built on the engine must **not** learn by predicting the next token. Language should emerge from a series of virtualisations, as in the brain: from symbols and associations up to language.
- The reading of "virtualisation" below is confirmed.
- Referents are not a grid world. They are the engine's own n-dimensional structures; see `tesseract-data-structure.md`.

This replaces two earlier lines: "predict the next input" as the general training signal in `tesseract-design-notes.md`, and test 11 (next byte on a small text) as the language test in `tesseract-test-plan.md`. The first draft (tests 0 to 3) is not affected: those tests are about association and retrieval.

## Virtualisation (confirmed)

Each level works on codes produced by the level below and gives the level above a simpler, more abstract world, the way a virtual machine hides the hardware under it. Every level is the same machinery (state, match-and-count memory, attention as retrieval), which fits the recursive design of the engine.

| Level | Works on | Produces |
|---|---|---|
| 0. Signals | bits from the adapters | — |
| 1. Symbols | recurring patterns of signals | a stable code for each family of similar inputs |
| 2. Associations | symbols that occur together, including across input types (a word form and what it refers to) | pairs and records |
| 3. Sequences and events | ordered symbols | episodes: what follows what, at this level |
| 4. Schemas | many episodes | records with roles (who, does what, to what): variables |
| 5. Language | role records and word sequences | the mapping in both directions: understanding (words to structure) and production (structure to words) |

Attention is used to retrieve and bind (query a memory with a partial key), not to score the next token.

## Learning signals without next-token prediction

1. **Association by co-occurrence:** things that occur together are bound; evidence is counted. Local, integer, no backward pass.
2. **Completion:** hide part of a record and recall it. This is the native operation of an associative memory.
3. **Local prediction:** each level predicts its own next code, not the next word.
4. **Chunking:** when a combination recurs and is predictable, it gets a symbol of its own at the level above. This is the mechanism that creates new levels of abstraction.
5. **Task feedback** when a task gives one (a right or wrong answer).

Generating language is then the unrolling of a structure into words, role by role. Some ordering of words is still needed at the last step, but it is the last layer, not the objective that shapes the whole system.

## Why it suits this engine

- Outputs are retrieved by nearest match among stored codes, not chosen from a probability table over a large vocabulary. Count-based outputs over thousands of classes are the weak point found in the prior art review.
- Association, completion and chunking need counting and comparing only.

## Test ladder for language

On generated worlds of entities (see the data structure document):

1. Naming: entity or value seen, recall its name, and the reverse.
2. Relations along a dimension: "bigger than", "left of".
3. Events: what changed between two scenes.
4. Questions and answers about a scene, including unseen combinations of values.
5. Transfer: a relation learned on some dimensions, tested on a dimension where it was never trained.
6. Production: describe a scene.

Each has a held-out set and a trivial baseline, as in the test plan.

## Risks, stated plainly

- Next-token prediction is the objective behind today's language models because it is simple, dense and scales with text. The alternatives in this family (hierarchical memory models, vector symbolic architectures, grounded micro-worlds) have not reached comparable language ability. From memory, not re-checked in this pass.
- Level 4 is the hard step: learning roles and variables from episodes (compositional generalisation) is an open research problem.
- A miniature grounded language is feasible on a PC; open-ended conversation is not a near-term outcome of this route.

## Related lines of work (from memory, to be checked before relying on them)

Vector symbolic architectures and hyperdimensional computing (role and filler binding); conceptual spaces (quality dimensions grouped into domains); hierarchical temporal memory and the Thousand Brains Project; prediction in representation space instead of input space; chunking and grammar induction; large brain models built on semantic pointers.
