# Tesseract

A binary, vectorised, multi-dimensional learning engine that learns,
restructures and prunes itself. Its principle is **count and compare, never
multiply**: integers are produced by counting bits and consumed by
comparisons that give bits back. There is no floating point in the engine.

This is the **first draft**: the engine's base parts, the first three tests
of the plan, a command-line runner and a local desktop monitor. It is local
software; nothing in it is a web application.

## What is here

| Crate | What it does |
|---|---|
| `tesseract-core` | The engine: the 8⁴-bit entity and its axis rotations, codes for a dimension, the integer state, the ternary counting unit, the memory block, the attention head, the learning counters and the structure policy. No dependencies except optional threads. |
| `tesseract-codecs` | Adapters: bytes and numbers to entities and back. |
| `tesseract-tasks` | Tasks 1 to 3 with held-out sets, baselines and pass thresholds. |
| `tesseract-run` | `tesseract run <config>`: one command, one config file, one run folder. |
| `tesseract-monitor` | A desktop window: set up a test, watch it, look at the neurons. |

The engine knows nothing about tasks, files or windows. Dependencies point
inwards: everything may depend on `tesseract-core`, which depends on none
of the others.

## Build and run

Needs [Rust](https://rustup.rs) 1.95 or newer (`rustup update`). Nothing else
on Windows and macOS; on Linux the monitor needs a desktop with OpenGL.

```sh
cargo test --workspace                      # kernel checks and tests 0 to 3
cargo run --release -p tesseract-run -- run configs/test1_majority.toml
cargo run --release -p tesseract-run -- run configs/test2_recall.toml --set task.pairs=2000 --set budget.steps=2000
cargo run --release -p tesseract-monitor    # the desktop monitor
cargo bench -p tesseract-core               # speed measurements
```

`--set key.path=value` overrides any setting of the config file. The exit
code of `tesseract run` is 0 when the test passes, 1 when it fails.

Each run writes `runs/<name>/<UTC time>/` with `config.toml` (the exact
settings), `run.json` (seed, engine version, source revision, result),
`metrics.jsonl` (one line per evaluation) and `snapshots/` (the state of
every unit, for the neuron view). The same seed gives the same run on any
number of threads; this is tested.

## The monitor

One window, three areas:

1. **Setup**: choose a test, edit its settings, start and stop. Past runs
   are listed and can be opened again.
2. **Run**: score on training and held-out data with the baselines and the
   pass threshold as lines; flips per step; active, dormant, removed and
   protected connections; speed and memory.
3. **Neurons**: one cell per unit, coloured by state, usage or live
   inputs. Click a unit to see its 4096 input connections and its output
   votes laid out as the hypercube: 8 domains of 8 values of 8 × 8 bits.
   A slider replays the snapshots of a finished run.

It runs the engine inside its own process and reads the run folders; there
is no server and no browser.

## Results of the first three tests

Measured with the configs in `configs/`, seed 1. Each test was also run on
other seeds and settings (see below).

| # | Test | Result on held-out data | Threshold | Baselines |
|---|---|---|---|---|
| 0 | Kernel checks: every bit operation against a slow reference on random inputs; same results for 1, 2, 3 and 8 threads | exact match | exact | — |
| 1 | Majority of a hidden subset (65 of 4095 inputs, hidden signs), one counting unit | 100.00% after 10,000 examples | 99% | chance 50.5% |
| 2 | Associative recall, 1000 pairs, 10% of key bits flipped, memory block of 4096 units | 100.00% | 99% | nearest-neighbour search 100% |
| 3 | Attention lookup, 8 records, 16 values, 3 misleading domains, one head | 100.00% after 1,500 sequences | 95% | chance 6.2%, last value 19.7%, closest record over all domains 20.4% |

What the runs show beyond the score:

- **Test 1.** The unit's mask ends as exactly the 65 hidden inputs with the
  hidden signs (checked bit by bit in the tests). It also passes with
  subsets of 9, 33, 129 and 257 inputs and on seeds 2 to 5.
- **Test 2.** Recall stays exact up to 30% noise, is 99.3% at 35% and
  collapses at 40%: a unit fires when at least 62.5% of its inputs agree
  (`vigilance`), so the cliff is where the setting puts it. Capacity is
  the number of units: 4096 pairs are recalled exactly, 6000 pairs give
  68.3% and 8192 give 50.0%, because a full block drops new pairs.
- **Test 3.** The head ends matching the whole key domain, none of the
  misleading domains, and copying exactly the value domain (checked in the
  tests). It also passes with 0 and 6 misleading domains, 16 records with
  256 values, other domain positions, and seeds 2 to 4.

### What these results do not show

- Test 3 is the simplest honest form of the question. Each record is one
  entity, so no position is needed; the head learns two masks (where to
  match, what to copy), not general projections; and the signal that
  trains the selection is computed inside one block from the error at its
  own output. Whether that signal survives a stack of blocks is not
  tested. The risk named in the design notes is reduced, not removed.
- Test 2's memory stores one pair per unit. It is a content-addressed
  table that tolerates noise, not a distributed memory: it does not
  compress, and its capacity is its size.
- Only one block is trained at a time. The error signal is not yet passed
  backwards through a block (the unit's Boolean sensitivity is implemented
  and tested, but nothing consumes it).

## How it learns

Every connection has one small signed counter. Evidence moves it by one:
up when a positive connection would have helped, down when a negative one
would have. The counter decides the connection's two bits with hysteresis:
dormant below a threshold, live with the sign of the evidence above it. A
live connection that loses its evidence is made dormant by the structure
policy, never immediately: on a budget per period, never while the error
is rising, never if protected. A dormant connection keeps its counter and
returns when the evidence does. Removal does not exist yet.

The same counters train unit inputs, memory values and both masks of the
attention head. For the head's hard selection, the error at the output
says which record would have reduced it most; when that is not the record
selected, every match connection on which the two differ gets one unit of
evidence in favour of the useful one.

## Speed

One core of an Intel Xeon at 2.1 GHz (a cloud machine, not the reference
laptop), `cargo bench -p tesseract-core`:

| Operation | Portable build | AVX2 level | Native (AVX-512) |
|---|---|---|---|
| Match count over one entity (4096 connections) | 63 ns | 21 ns | 11 ns |
| Memory block, 4096 units, one token | 270 µs | 121 µs | 114 µs |
| Attention scores, 256 candidates, one head | 19 µs | 7.1 µs | 3.8 µs |
| State update, 4096 votes | 4.2 µs | 1.6 µs | 1.1 µs |
| One learning step of a unit | 12 µs | 6.1 µs | 5.8 µs |

The repository builds for the processor of the machine it is built on
(`.cargo/config.toml`). That is an interim choice; choosing the fast path
at start-up is planned. The memory block is limited by memory access, not
by counting: 4096 units are 4 MB of signs and masks per token.

## Geometry

The shape of the engine is set by two constants at the top of
`crates/tesseract-core/src/entity.rs`:

```rust
pub const SIDE: usize = 8;    // how many of each level fit in the one above
pub const LEVELS: usize = 4;  // axes inside one entity: 8^4 = 4096 bits
```

Change them and rebuild; every type, test and benchmark follows.
`cargo run -p tesseract-core --example geometry` prints the current shape,
and `python3 scripts/sweep_geometry.py` rebuilds at several shapes and
measures them. `SIDE` must be a power of two, and an entity and a domain
must each hold at least one 64-bit word.

Measured with that script, one core (`sweep_geometry.json`). Tests 1 to 3
were run with 33 relevant inputs, 400 stored pairs in 512 units, and
5,000 lookup sequences:

| n | axes | entity | test 1 | test 2 | test 3 | match | memory block | attention | rotate inner | rotate 2nd axis |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 7 | 128 b | 100% | 4% | 100% | 0.8 ns | 4.5 µs | 0.35 µs | 7.9 ns | 7.8 ns |
| 4 | 4 | 256 b | 100% | 56% | 100% | 1.0 ns | 9.7 µs | 0.6 µs | 5.9 ns | 4.8 ns |
| 8 | 3 | 512 b | 100% | 99% | 100% | 1.3 ns | 11.2 µs | 1.0 µs | 5.6 ns | 5.1 ns |
| **8** | **4** | **4096 b** | **100%** | **100%** | **100%** | **8.9 ns** | **137 µs** | **3.1 µs** | **19.8 ns** | **17.1 ns** |
| 16 | 3 | 4096 b | 100% | 100% | 100% | 9.3 ns | 165 µs | 3.0 µs | 19.2 ns | 2941 ns |
| 8 | 5 | 32768 b | 100% | 100% | 29%¹ | 80.7 ns | 1151 µs | 51.3 µs | 102.9 ns | 101.8 ns |

¹ 100% with 20,000 steps instead of 5,000: the head starts matching on
every bit, so the evidence it must gather grows with the entity.

What the numbers say:

- **Counting costs scale with the bits**, as they should: eight times the
  bits cost about nine times the time.
- **Below about 512 bits an entity saves little and loses a lot.** From
  128 to 512 bits is four times the capacity for 1.6 times the time,
  because fixed overheads dominate the small shapes.
- **Memory capacity is set by the width of the key, not by the number of
  units.** Two random keys of `m` bits disagree on about `m / 2` bits
  with a spread of `sqrt(m) / 2`, so a unit that fires at `v` agreement
  only stays distinct while `v >= 0.5 + 2.5 / sqrt(m)`. At the default
  0.625 vigilance that needs `m >= 256`: measured 99% at 256-bit keys,
  56% at 128 and 4% at 64.
- **n = 16 buys nothing and costs the fast rotation.** At the same 4096
  bits it matches as fast, but its middle axis has a digit that straddles
  a 64-bit word, so rotation falls back to the reference path and is 173
  times slower.
- **n = 8 with 4 axes is the only shape here that is fast everywhere and
  large enough to hold a memory.** The choice was an assertion in the
  design notes until this sweep; now it is measured.

## Not in this draft

Stacks of blocks and the backward signal between them; merge, unite,
connect and split; references and concept entities; removal and regrowth;
position by rotation in a task (the rotations exist and are tested); the
language ladder; bit planes for the state; a GPU backend; image and audio
adapters; Python bindings.

## Licence

MIT.
