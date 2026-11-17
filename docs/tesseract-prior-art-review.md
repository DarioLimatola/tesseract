# Tesseract — prior art review (2026-11-16)

Question: have other projects already met the problems Tesseract faces, and how did they solve them?

**How this was read.** Each paper was read through an automated page reader with targeted questions. Long papers were read up to their first ~100,000 characters, so appendices of BOLD and of the convolutional logic-gate paper were not covered. Numbers are as reported by the authors. Items marked *(memory)* come from general knowledge and were not re-checked in this pass.

## 1. Short answer

- Most of our open problems have a published answer. The answers agree with the choices proposed in the design notes, with three corrections (section 5).
- Four independent lines of work arrive at the same mechanism for learning a binary parameter: a hidden scalar per parameter that accumulates evidence, plus a threshold that triggers the flip (BOLD, Bop, Tsetlin machines, HTM).
- Every system that scales keeps something non-binary at the edges: real-valued first and last layers, real embeddings, or real accumulators.
- Structure matters more than size: shared rules across positions and learned wiring each buy more than a 10x increase in gate count.
- No project found combines native Boolean learning, learned wiring, recurrence, online prediction and several input types. That combination is where Tesseract would be new, and where the risk sits.

## 2. The closest projects

| Project | Unit | How it learns | What is not binary | Best reported results | Main limit |
|---|---|---|---|---|---|
| BOLD (Huawei, 2024) | Neuron with many inputs: XNOR or XOR per input, count, threshold | Boolean "variation" with a chain rule; backward signals are Boolean per example and are summed into signed counts over the batch and over consumers | Per-weight accumulator is real-valued; first and last layers are full precision, trained with Adam; optional batch norm | CIFAR-10 90.29% (92.37% with batch norm) vs 93.80% float; ImageNet ResNet18 51.8%, or 66.9% at 4x width, vs 69.7% float; BERT on GLUE 70.9 vs 83.9 float, using distillation from a float teacher | Energy savings are analytical estimates, not measurements; runs on GPUs built for float arithmetic; language models left to future work |
| Bop (2019) | Binary-weight neuron | Moving average of the gradient per weight; flip when it exceeds a threshold and points against the weight | Gradients and the moving average are real | CIFAR-10 91.3% vs 90.9% for latent weights; ImageNet up to 56.6% | Fixed threshold for all layers; high flip rates learn fast then plateau lower |
| Logic gate networks (Petersen, 2022) | Two-input gate, one of 16 | 16 real parameters per gate, softmax over gates, gradient descent; hardened to one gate after training | All of training | MNIST 98.47% with 384K gates; CIFAR-10 62.14% with 5.12M gates after 90 h | Wiring random and fixed; depth limited to about 8–10 layers; training cost |
| Convolutional logic gate networks (2024) | Trees of gates applied at every position, OR pooling | Same, plus "residual initialisation" (each gate starts ~90% as pass-through) | All of training | CIFAR-10 86.29% with 61M gates (84.99% with 28.9M); MNIST 99.35% with 1.27M gates; about 23 logic layers | Wiring still random and fixed; needs custom GPU kernels |
| LILogicNet (2025) | Two-input gate | Each gate input chooses among K random candidates through learned real scores (K = 2 to 128) | All of training | MNIST with 8K gates, one layer: 91.48% fixed wiring vs 98.15% learned; CIFAR-10 60.98% with 256K gates vs 60.78% with 1.28M fixed | Cost grows with K; the output counters take over two thirds of the FPGA logic |
| DWN (2024) | Lookup table with up to 6 inputs | Approximate derivative of a table by its inputs ("extended finite difference"); learned input mapping | Table entries and mapping scores are real during training | MNIST 98.31% with 4.6K tables; CIFAR-10 57.42%; beats XGBoost on tabular data | Bit-packed inference on a microcontroller was 8.3x slower than XGBoost; no convolution |
| Recurrent logic gate networks (ETH, 2025) | Two-input gates with recurrent state | Gradient descent on relaxed gates | 1024-dimension real embeddings; real softmax over the vocabulary | Translation BLEU 5.00 (4.39 once hardened) vs GRU 5.41 and a small Transformer 5.98, on 16-token sequences | Worse at 64 tokens; vanishing gradients on long sequences; subword tokens worked much worse than words |
| Tsetlin machines (2018–) | Clause: AND of chosen input bits and their negations; clauses vote per class | One small integer automaton per (clause, input bit); reward and penalty move its state; no backward pass *(memory for the feedback rules)* | Nothing in the core; inputs need hand-made Boolean encodings | CIFAR-10 about 61% (2020), 75.1% (2023), 82.8% with 22 specialised machines combined | Essentially one layer; accuracy depends on encodings and ensembles |
| HTM (Numenta) | Columns of cells; about 2% of columns active | Synapse "permanence" between 0 and 1 with a connection threshold; local Hebbian rule; learns online by predicting the next input | Permanences are real | Sequence tasks comparable to LSTM; grammar 98.4% vs 100% for LSTM; unaffected by loss of 30% of cells; fastest to recover after a change | Sensitive to timing noise; slower on long dependencies; untested on speech or video in that study |
| Sparse distributed memory and attention (2021) | Binary addresses, integer counters | Write and read within a Hamming radius, majority readout | Counters are integers | Theory: this read operation approximates the softmax attention of Transformers | Approximate; assumes random patterns; classic setting needs ~1000-bit vectors and ~10^6 storage units |
| Evolved circuits (Tiny Classifiers, 2023) | AND, OR, NAND, NOR gates | Evolution (1+4 strategy with neutral drift) | Nothing | 33 tabular datasets: 78% mean accuracy vs 81% for XGBoost | Circuits of at most ~300 gates |
| BitNet b1.58 (Microsoft) | Float-style Transformer with weights in {-1, 0, +1} | Gradient descent with hidden float weights | 8-bit activations; float training | 2-billion-parameter language model trained on 4 trillion tokens | Training cost is that of a normal language model |

Also noted: a study of logic gate networks at scale (ETH) found they match a plain float network up to roughly 67–100 classes but not on 1000-class ImageNet at 32x32; one cause is that an output of a depth-n network of two-input gates can depend on at most 2^n inputs. A fix for the mismatch between soft training and hard inference (noise plus hard selection during training) reports 4.5x faster training and a 98% smaller gap, according to its summary page. A 2025 thesis on predictive coding states that local learning rules remain limited to simple tasks and are orders of magnitude more expensive than backpropagation on GPUs. The Thousand Brains Project (Monty) uses local Hebbian learning on real-valued features, not binary ones, and is at an early stage.

## 3. Our problems against their solutions

| Our problem | Who addressed it | Their solution | Status for Tesseract |
|---|---|---|---|
| What do the 64 lanes of a word hold? | Logic gate networks | Each bit position is one example: 64 examples per word at inference | Adopt. No precedent was found for Predatore's "different rule per lane" |
| Conflicting backward requests | BOLD | Boolean vote per example and per consumer, summed into a signed integer count | Adopt, and keep the count; do not reduce it to one majority bit |
| How fast to flip, without oscillating | Bop, BOLD, Tsetlin, HTM | Hidden accumulator per parameter, threshold, reset after a flip; BOLD damps by the fraction of unchanged weights | Solved in principle. Tsetlin shows small integers suffice; schedules need tuning |
| Reading the answer | Logic gate networks, DWN | One group of output bits per class, count the ones | Solved for tens of classes. Sensitive to a temperature setting; expensive in hardware; unsolved for large vocabularies |
| Encoding inputs | DWN, logic gate networks, HTM, Tsetlin | Thermometer codes (3 to 31 thresholds per colour value); sparse codes; many hand-made encodings | Solved for images and tables. For text the working system used learned real embeddings |
| Depth | Convolutional logic gate networks | Start every gate as pass-through | Adopt |
| Wiring | LILogicNet, DWN, evolved circuits | Learned choice among K candidates (real scores); evolution up to ~300 gates | Partly solved, never with integers only. Worth an experiment: candidate sets with integer counters |
| Recurrence | Recurrent logic gate networks, HTM | Works on short sequences | Partly solved; long sequences remain hard |
| Learning without labels | HTM | Predict the next input with a local rule, online | Works on simple sequences only; no evidence at scale |
| Attention | BOLD (BERT), sparse distributed memory | Binary BERT needs a float teacher; Hamming-radius memory approximates attention in theory | Open. The memory route fits a bit engine and is untested as a trained language model |
| Pruning | Logic gate networks | Many gates end as pass-through and are removed by circuit simplification after training | Solved after training; not found during training |
| Learning with bits only | Tsetlin | Integer automata, no backward pass | Only this line avoids real numbers in training, and it is one layer deep |

## 4. Not found anywhere

1. Training small gates or lookup tables with integer-only state. All such work uses real-valued relaxations on GPUs.
2. Learned wiring without real-valued scores.
3. Online or continual learning in logic gate networks.
4. Evolution combined with error-driven Boolean learning in one system.
5. One logic network handling several input types.
6. A generative text model made of gates, beyond a 16-token translation model.

## 5. Corrections to the Tesseract design

1. **Backward signals are signed counts.** Boolean per example, accumulated as small signed integers in bit planes. The earlier plan thresholded them to one bit by majority, which discards the strength of the evidence.
2. **Boolean core, integer edges.** Input encoders and output counts are integers by design. Every working system needed richer first and last stages.
3. **Lanes hold examples (and positions when a rule is shared).** Distinct rules are distinct units, not distinct lanes of one unit.
4. **Units start as pass-through.**
5. **Structure before size:** shared rules across positions first; candidate-set wiring as the first structural experiment.
6. **Text starts at byte or character level** (at most 256 classes), where count-based outputs are known to work.
7. **Local learning is an experiment, not the default.** The global Boolean backward pass is the baseline.
8. **Efficiency must be measured.** BOLD's energy figures are estimates, and bit-packed inference was slower than a tree model in one microcontroller test.

Reference points for the test ladder: MNIST 98.3–98.5% with 4.6K six-input tables, 8K gates with learned wiring, or 384K gates with random wiring; CIFAR-10 about 60–62% without spatial structure and 86% with it.

## 6. Reference hardware

The owner's machine is an Acer Nitro V 15 laptop; the exact configuration is not yet known. Typical configurations of that series *(memory, not verified)*: a recent Intel Core H-series or AMD Ryzen HS-series processor with 8 or more threads, 8–16 GB of RAM and an Nvidia laptop GPU with 4–8 GB. These processors support AVX2 and, as far as known, not AVX-512.

The engine must not depend on one machine:

- Baseline: any 64-bit processor, plain 64-bit words.
- Faster paths chosen at start-up: AVX2, then AVX-512 or ARM NEON where present.
- GPU as an optional backend for large regular layers.
- Same data layout and same results on every path (deterministic random bits).

## 7. Sources

- BOLD: https://arxiv.org/html/2405.16339v2
- Boolean variation and Boolean logic backpropagation: https://arxiv.org/html/2311.07427v2
- Bop, "Latent Weights Do Not Exist": https://ar5iv.arxiv.org/html/1906.02107
- Deep differentiable logic gate networks: https://ar5iv.labs.arxiv.org/html/2210.08277
- Convolutional differentiable logic gate networks: https://arxiv.org/html/2411.04732v1
- LILogicNet: https://arxiv.org/html/2511.12340
- From MNIST to ImageNet, scalability of logic gate networks: https://arxiv.org/pdf/2509.25933
- Mind the Gap (summary page): https://paperswithcode.co/paper/2506.07500
- Differentiable weightless neural networks: https://arxiv.org/html/2410.11112v5
- Recurrent deep differentiable logic gate networks: https://arxiv.org/pdf/2508.06097
- Tsetlin machine composites toolbox: https://ar5iv.arxiv.org/html/2406.00704
- HTM sequence memory, summary of "Continuous online sequence learning": https://blog.acolyer.org/2017/10/24/continuous-online-sequence-learning-with-an-unsupervised-neural-network-model/
- Attention approximates sparse distributed memory: https://arxiv.org/pdf/2111.05498.pdf
- Tiny classifier circuits: https://ar5iv.labs.arxiv.org/html/2303.00031
- Scaling predictive coding (thesis): https://arxiv.org/pdf/2510.23323
- Thousand-brains systems: https://arxiv.org/html/2507.04494v1
- BitNet: https://jmlr.org/papers/volume26/24-2050/24-2050.pdf
