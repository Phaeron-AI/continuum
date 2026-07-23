# Continuum: A From-Scratch Mamba World Model for Interactive Simulation

*Research foundation — working draft.*

## Abstract

Continuum is a neural world model that learns to simulate an interactive
environment frame-by-frame from action-conditioned video. It is built from
scratch in three stages: a finite-scalar-quantization (FSQ) visual tokenizer, a
selective state-space (Mamba) dynamics model over discrete tokens, and a
real-time recurrent generation loop served to a browser client. On a 2D grid
environment we ablate three architectural choices — the Mamba block, FiLM action
conditioning, and context-noise augmentation — and find that **teacher-forced
next-token accuracy is a misleading proxy for free-running generation quality**:
the Mamba block that dominates one-step prediction (+20 points) confers no
rollout benefit, while context-noise augmentation, despite the *lowest*
teacher-forced accuracy among the model variants, yields the best rollout
stability. We argue that noise augmentation is the decisive ingredient that
turns a sharp autoregressive SSM into a usable world model, and that rollout
metrics — not teacher-forced accuracy — must be the objective of record.

## 1. Motivation

A world model learns the dynamics of an environment: given past frames and an
action, it predicts the next frame. Run autoregressively, it becomes a
simulator — the neural network *is* the engine. Recent systems (GameNGen, Genie,
DIAMOND) have shown that such models can be made interactive and even playable,
but they are large, diffusion- or transformer-based, and compute-intensive.

Continuum asks a narrower question: **how far can a small, linear-time,
token-based world model be pushed toward real-time interactivity, and which
architectural choices actually matter for *generation* rather than for one-step
prediction?** The choice of a selective state-space model (Mamba) is deliberate:
its recurrent form costs O(1) per token at inference, making constant-rate
real-time generation feasible on modest hardware — unlike a Transformer, whose
per-token cost grows with context length.

## 2. Background and related work

Continuum sits at the intersection of five lines of work, each mapping onto one
of its components.

- **Selective state-space models (Mamba).** Gu & Dao (2023) introduce
  input-dependent state-space dynamics with a hardware-aware parallel scan,
  matching Transformers on language while remaining linear-time. Continuum's
  dynamics backbone is a from-scratch selective SSM with a pure-PyTorch
  associative scan for training and a recurrent step for generation.
- **Token-based world models (IRIS).** Micheli et al. (2022) cast dynamics
  learning as autoregressive sequence modeling over discrete image tokens from a
  learned autoencoder. This is Continuum's closest architectural sibling —
  Continuum is, in essence, the IRIS recipe with a Mamba backbone in place of a
  Transformer.
- **Real-time playable simulation (GameNGen).** Valevski et al. (2024) show a
  diffusion model can serve as a real-time game engine, and identify
  *context-frame noise augmentation* as the key to arresting autoregressive
  drift. Continuum adopts the same anti-drift principle in the discrete,
  token-based setting, and our ablation independently confirms its importance.
- **Generative interactive environments (Genie).** Bruce et al. (2024) learn
  action-controllable world models from unlabeled video via a spatiotemporal
  tokenizer, an autoregressive dynamics model, and a latent action model. This
  is the playable-world vision Continuum's Phase 3 targets.
- **Finite Scalar Quantization (FSQ).** Mentzer et al. (2023) replace VQ-VAE's
  vector quantizer with per-dimension scalar quantization, eliminating codebook
  collapse. Continuum's tokenizer uses FSQ and achieves 100% codebook usage in
  practice.

## 3. Method

Continuum is built in three explicit stages, each derived and pinned by tests
before the next begins.

### 3.1 Phase 1 — Visual tokenizer (FSQ)

A convolutional encoder maps each 64×64 RGB frame to a latent grid, which FSQ
quantizes to an 8×8 grid of discrete tokens; a decoder reconstructs pixels. The
tokenizer is trained once and **frozen** — the world model never sees pixels,
only token grids. On the grid environment the frozen tokenizer reaches **37.8 dB
reconstruction PSNR with 100% codebook usage (64/64 codes)**, confirming the FSQ
no-collapse property.

### 3.2 Phase 2 — Dynamics model (selective SSM)

Each frame's 8×8 token grid is flattened row-major into 64 tokens; per-frame
actions are interleaved into the stream. The model predicts the next token
autoregressively, with the loss scored on visual tokens only (actions are
context, never targets). Key components:

- **Selective SSM** with input-dependent Δ, B, C. Computed two equivalent ways
  behind one interface: a **parallel associative scan** (Hillis–Steele,
  log-depth) for training, and a **recurrent step** carrying a fixed-size state
  for generation. Both are validated against a fully-explicit reference.
- **True Mamba block** (optional): input projection → depthwise **causal**
  conv → SiLU → selective SSM → gated output. The short causal convolution
  captures local frame structure (edges, single-step motion) a bare SSM misses.
- **Action conditioning:** *inline* (the action rides the token stream) or
  *FiLM* (the action additionally modulates every position of the frame it
  governs via feature-wise linear modulation).
- **Context-noise augmentation:** during training, a fraction of the *visual
  context* tokens are replaced with random tokens (targets and action tokens are
  never corrupted). This teaches the model to recover from its own errors at
  generation time — the anti-drift mechanism.

### 3.3 Phase 3 — Real-time generation

Generation uses the recurrent step (O(1) per token). A FastAPI server holds a
per-client `GenerationSession`, warms its state on a seed clip, and advances one
frame per action; a browser client streams frames over a WebSocket
(action out, PNG frame in), keeping the latency-critical data path free of any
middleware.

## 4. Experiments

### 4.1 Setup

All arms are trained on an identical grid-2D token cache generated with a
movement-biased policy (~44% unique frames), using a frozen FSQ tokenizer
(vocabulary 64), a 0.84M-parameter model, and 4000 training steps, with data
split, batch order, and seed held fixed. We report two metric families:

- **Teacher-forced accuracy** — next-token accuracy one step from ground truth.
- **Rollout accuracy** — free-running token accuracy over a 24-frame greedy
  rollout (mean over the rollout, and at the final frame), the quantity that
  actually reflects generation quality.

### 4.2 Additive ablation

| arm | ce loss ↓ | teacher-forced acc ↑ | rollout acc mean ↑ | rollout acc @24f ↑ |
| --- | ---: | ---: | ---: | ---: |
| baseline (SSM) | 1.605 | 60.2% | 22.6% | 20.6% |
| + Mamba block | 0.842 | 79.8% | 21.8% | 20.6% |
| + FiLM actions | 0.844 | 79.8% | 24.3% | 27.3% |
| + noise aug (full) | 1.030 | 74.4% | **29.5%** | 27.3% |

### 4.3 Principal finding

**Teacher-forced accuracy does not predict generation quality.** The Mamba block
is the strongest one-step predictor by a wide margin (+20 points teacher-forced,
−48% cross-entropy) yet yields *no* improvement in free-running rollout
(21.8% vs. the baseline's 22.6% mean) — a textbook exposure-bias signature: a
sharper, more confident model compounds its own errors faster once it feeds on
its own outputs. Only **context-noise augmentation** improves rollout (best mean,
29.5%), and it does so *despite* the lowest teacher-forced accuracy of the model
variants. FiLM action conditioning is neutral one-step but improves rollout
stability, indicating its value is likewise invisible to teacher-forced metrics.

The implication for world-model research is concrete: **optimize and report
rollout metrics, not teacher-forced accuracy**, and treat noise augmentation as
a first-class ingredient rather than a regularization afterthought.

## 5. Discussion

The results decompose into three distinct, individually-interpretable effects
that the standard teacher-forced view collapses into one misleading number:
representational sharpness (Mamba), conditioning stability (FiLM), and
error-recovery (noise augmentation). That a change can *help* one-step accuracy
while doing nothing for generation — and another can *hurt* one-step accuracy
while being the single most important change for generation — is the core
lesson.

An earlier attempt to measure drift via a PSNR-threshold "horizon" failed: on a
sparse environment, PSNR is dominated by the static background and is nearly
insensitive to the small dynamic region that actually drifts, so the metric was
either saturated (every frame passes) or degenerate (no frame passes). Per-step
token accuracy over the rollout proved to be the discriminating signal.

## 6. Limitations

This is a foundation, not a finished result. Honestly stated:

- **Toy environment.** A 2D grid with a single agent is the smallest setting
  that exercises the machinery; it does not test high-resolution perception or
  rich dynamics.
- **Small scale.** 0.84M parameters, 4000 training steps, a 64-code vocabulary.
  Absolute rollout accuracies are low by construction; only the *relative*
  ordering across arms is claimed, and it is stable across evaluation batch
  sizes.
- **Out-of-distribution idle action.** The training policy excluded the no-op
  (INTERACT) action, so idle frames — which request that action — drift; this is
  a data-coverage artifact, not a model failure.
- **Evaluation breadth.** Single training seed, small rollout batch, greedy
  decoding. Error bars over seeds and larger rollout batches would harden the
  numbers.

## 7. Future work

- **Scale the environment and model** — higher resolution, richer dynamics,
  larger vocabulary and parameter count — to test whether the ordering holds.
- **Noise-probability sweep** to locate the rollout-stability optimum.
- **FiLM in the recurrent generation path** (currently forward-path only).
- **Better drift metrics** — dynamic-region-weighted PSNR, learned perceptual
  distance (LPIPS/VMAF), and longer horizons.
- **Transfer of the capability** beyond games — action-conditioned rollout is
  directly relevant to robotics simulation and to world-model tooling, where the
  same machinery has a clearer economic use than neural game replay.

## 8. Reproducibility

The full pipeline is scripted and configuration-driven:
`generate_dataset` -> `train_tokenizer` -> `build_token_cache` ->
`train_world_model` (or `run_ablation` for the study). Every design decision is
derived in `docs/` and pinned by a load-bearing test; the ablation table above
is produced by `scripts/run_ablation.py`.

## References

1. Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752 (2023).
2. Micheli, Alonso & Fleuret. *Transformers are Sample-Efficient World Models (IRIS).* arXiv:2209.00588 (ICLR 2023).
3. Valevski et al. *Diffusion Models Are Real-Time Game Engines (GameNGen).* arXiv:2408.14837 (2024).
4. Bruce et al. *Genie: Generative Interactive Environments.* arXiv:2402.15391 (ICML 2024).
5. Mentzer et al. *Finite Scalar Quantization: VQ-VAE Made Simple (FSQ).* arXiv:2309.15505 (ICLR 2024).