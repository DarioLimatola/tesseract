# Tesseract — test plan and monitor (draft, 2026-11-16)

Status: proposed, except where marked **Decided**.

## Rules

- **Decided:** the engine and the tests are separate. The engine knows nothing about any task.
- **Decided:** any kind of input can enter the engine, through adapters.
- **Decided:** tests and interface are simple. One page to set up a test, watch it, and look at the neurons. Predatore's interface is the example of what to avoid.
- Every test has a held-out set, at least one trivial baseline, and a pass threshold written down before the first run.
- Every test declares which axes of the engine it uses.

## Three parts

1. **Engine (library).** Bit tensors, units, blocks, learning, pruning. Input and output are bit tensors with declared axes.
2. **Adapters.** Turn bytes, numbers, grids or images, and audio frames into bit tensors and back. One small interface: `encode`, `decode`, declared shape.
3. **Tasks.** One small interface: `next_batch()` gives inputs and targets, `score()` gives metrics, `baseline()` gives the trivial score. A task is chosen and configured in one short config file: task and its parameters, model shape, learning settings, budget, seed.

A run is one command on one config file. It writes one folder: a copy of the config, a metrics log (one line per step), and state snapshots for the neuron view.

## Test ladder

| # | Test | What it shows | Axes | Baseline or reference |
|---|---|---|---|---|
| 0 | Kernel checks: every bit operation against a slow reference; same results for any thread count | The kernel is correct and deterministic | all | exact match |
| 1 | Majority of a hidden subset of inputs | One counting unit learns, masks find the relevant inputs | batch | chance 50% |
| 2 | Associative recall: store N key-value pairs, query with noisy keys | Capacity of the memory block, as a curve against N and noise | batch | nearest-neighbour search |
| 3 | Attention lookup: several key-value pairs, then a query key; answer with its value | One attention head with binary keys can learn what to select (the riskiest part) | batch, time | chance; "last value seen" |
| 4 | Copy the token k positions back | Position by rotation works | batch, time | chance |
| 5 | Nested records: read a value at depth 1, 2, 3 | Capacity and noise of recursive vectors | batch, plane | exact table |
| 6 | Shape anywhere: detect a small shape on a 2D grid, train on some positions, test on others | A rule shared along two space axes generalises to unseen positions | batch, 2 space | chance; unit without sharing |
| 7 | Game of Life: predict the next board, then roll out many steps | Local rule on 2 space axes plus time; exact solution exists | batch, 2 space, time | published: learned by differentiable logic cellular automata |
| 8 | Moving object: predict the next frame | Prediction without labels on space plus time | batch, 2 space, time | "copy last frame" |
| 9 | Cross-input association: digit image and digit byte, recall either from the other | Two input types in one network | batch, 2 space | chance 10% |
| 10 | Binarised MNIST | Comparison with the literature | batch, 2 space | 98.3–98.5% (see review) |
| 11 | Next byte on a small text | First language test, in bits per character | batch, time | n-gram model; small float Transformer |
| 12 | Grid chase with held-out maps | Control task inherited from Predatore | batch, 2 space, time | random 30%, greedy 75% in Predatore's own test |

Tests 6 to 8 use the multi-dimensional nature of the engine directly. Test 7 on 64 boards at once uses four axes together: batch, two of space, and time.

## Engine-property tests (run on any task above)

- **Pruning curve:** accuracy against the fraction of connections left.
- **Revival:** change the task and check that dormant parts wake up.
- **Forgetting:** task A, then B, then A again, with protected connections on and off.
- **Damage:** remove 30% of units at random and measure the drop.
- **Throughput:** steps per second against model size and thread count.

## Proposed thresholds for the first three tests

To be confirmed before the first run.

- Test 1: at least 99% on held-out examples.
- Test 2: report the curve; pass if recall is at least 99% for 1000 stored pairs with 10% of key bits flipped, using 4096 units.
- Test 3: at least 95% on unseen sequences of 8 pairs with 16 possible values (chance 6.25%).

A failure of test 3 is an acceptable outcome: it would send the attention design back for revision before anything is built on it.

## Monitor: one page, three areas

1. **Setup:** choose a task, edit its few settings, start and stop.
2. **Run:** score on training and held-out data with the baseline as a line; flips per step; counts of active, dormant, removed and protected connections; steps per second, time per token, memory.
3. **Neurons:** one grid per block, one cell per unit, coloured by activity, usage or state; click a unit to see its inputs and counters. Works from the state snapshots, so it can replay a finished run.
