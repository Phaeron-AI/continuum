# Phase 2 — The Mathematics of the Mamba World Model

A world model predicts the next token grid from the history of frames and
actions. This document derives the machinery — the selective state-space
recurrence, autoregression, teacher forcing — and settles how actions enter,
by working both options rather than guessing.

- **Input:** token grids from the frozen Phase 1 tokenizer
- **Mixer:** selective SSM (Mamba)
- **Target:** next-token prediction

Pipeline: `frames x → frozen tokenizer → token grids q → + actions a → Mamba → predicted next tokens q̂`

Each stage ends in a **code decision** — the implementation choice the math forces.

---

## Stage 00 — Objects & notation

**Derives:** the symbols, and where Phase 1 hands off to Phase 2.

Phase 1 gave us a frozen map from a frame to a grid of discrete tokens.
Phase 2 never sees pixels — it operates entirely on those token grids,
learning to predict the next grid from the history.

**Given / Definitions**

- An episode is a sequence of frames and actions: `x₀, a₀, x₁, a₁, …` — at
  each step the agent sees frame `xₜ` and takes action `aₜ`, producing `xₜ₊₁`.
- The **frozen tokenizer** `E` maps each frame to a token grid:
  `qₜ = E(xₜ) ∈ {0,…,V−1}^(h×w)`, with `V = |C|` the vocabulary and
  `h×w = 8×8 = 64` tokens (Phase 1's `TokenSpec`).
- Actions `aₜ` come from a small discrete set `A` (the `action_space`,
  Phase 0), embedded into the same space as token embeddings.
- The **world model** `p_θ` predicts the distribution of the next frame's
  tokens given everything so far: `p_θ(qₜ₊₁ | q≤ₜ, a≤ₜ)`.

The whole phase is learning `p_θ`. Once learned, feeding it a start frame and
a stream of actions lets it generate the future — the environment, simulated.

> **Code decision.** Phase 2 imports the `FrozenTokenizer` and calls only
> `encode` / `decode` — never the pixels. A Phase-2 `SequenceDataset` reads
> Phase 0 shards, tokenizes frames through the frozen artifact, and emits
> `(tokens, actions)` sequences. The tokenizer's `tokenizer_version` is
> recorded in the world-model checkpoint — a mismatch is a loud error.

---

## Stage 01 — Frames become one long sequence

**Derives:** how a grid of tokens plus an action becomes a 1-D sequence a
sequence model can consume.

A sequence model reads a 1-D stream of tokens. But each frame is a 2-D 8×8
grid. So the grid must be *flattened* into a fixed order — row-major is the
natural choice — turning one frame into 64 tokens in sequence:

```
qₜ = [ qₜ^(0,0), qₜ^(0,1), …, qₜ^(7,7) ]   →   64 tokens
```

An episode of `T` frames is then a stream of `64T` visual tokens, with the
actions slotted in between frames. The model reads this stream left to right,
and at each visual-token position it predicts the *next* token in the stream.
Predicting a whole next frame means predicting its 64 tokens in order.

> **Code decision.** The flatten order is a fixed contract — encode and
> decode must use the identical row-major order, or a predicted grid gets
> scrambled on the way back to pixels. This is a `reshape`, but it's a
> *contract*: assert round-trip (grid → flat → grid) in a test. The sequence
> builder is where actions get interleaved, which stage 06 settles.

---

## Stage 02 — The state-space recurrence

**Derives:** the core equation that makes Mamba linear-time — a running state
updated once per token.

A state-space model carries a hidden state `hₜ` — a fixed-size summary of
everything seen so far — and updates it one token at a time. The
continuous-time SSM is defined by

```
h'(t) = A·h(t) + B·x(t),    y(t) = C·h(t)
```

where `A` governs how the state evolves on its own, `B` how new input enters,
`C` how the state is read out. To run on discrete tokens, it's *discretized*
with a step size `Δ`, giving the recurrence the model actually computes:

```
hₜ = Ā·hₜ₋₁ + B̄·xₜ,    yₜ = C·hₜ
```

with `Ā = exp(Δ·A)` and `B̄ ≈ Δ·B` (the discretization). The crux: computing
`hₜ` needs only `hₜ₋₁` and the current `xₜ`. **Constant work per token,
constant memory.** A Transformer, by contrast, re-attends over all previous
tokens at each step — quadratic in sequence length. That difference is the
entire reason an SSM suits a long game session and real-time generation.

**Two views of the same operation**

- **Recurrent view** (above): one step at a time, O(1) per token — the
  *inference* path, and why generation is fast.
- **Convolutional / parallel view**: the same recurrence unrolls into one big
  parallel operation over a whole sequence at once — the *training* path, so
  a full episode trains in parallel on the GPU rather than looping.

> **Code decision.** Both views live behind one `SequenceMixer` interface: a
> pure-PyTorch recurrence (correctness-first, decision 0007) and the compiled
> `mamba-ssm` kernel later, validated to match. Training uses the parallel
> path; Phase-3 real-time generation uses the recurrent path with a carried
> state — the same math, two schedules.

---

## Stage 03 — Why "selective" is the whole trick

**Derives:** what Mamba adds over a plain SSM — input-dependent dynamics — and
why it matters for a game.

In a classic SSM, `Ā, B̄, C` are fixed — the state evolves the same way
regardless of what the input says. Mamba's insight is to make them *functions
of the input*:

```
Bₜ = Linear_B(xₜ),   Cₜ = Linear_C(xₜ),   Δₜ = softplus(Linear_Δ(xₜ))
```

Now the model can *choose*, per token, how much to let in and how much to
remember. A large `Δₜ` resets the state toward the new input (attend to
this); a small `Δₜ` preserves the running state (ignore this, keep context).
This is **selectivity**: the model dynamically filters what matters and
forgets what doesn't — exactly the property a game needs, where a "jump"
action should sharply update state while a no-op should barely perturb it.

> **Pitfall.** Selectivity is also what removes the efficient global
> convolution a non-selective SSM enjoys — the parameters now vary per step.
> Mamba recovers speed with a hardware-aware parallel scan. The pure-PyTorch
> reference won't have that kernel, so it's correct but slow; don't mistake
> its slowness for a bug.

> **Code decision.** For the from-scratch reference, the input-dependent
> `Bₜ, Cₜ, Δₜ` are three linear projections of the token embedding, and the
> scan is a plain Python loop over the sequence. Correct and legible; the
> kernel replaces the loop later. This is the stage most worth testing
> against the `mamba-ssm` output once available.

---

## Stage 04 — Autoregression and the loss

**Derives:** the training objective — next-token prediction as classification,
and why cross-entropy.

The model factorizes the probability of a whole sequence into a product of
next-token predictions (the chain rule of probability):

```
p_θ(q₁,…,q_N) = ∏ᵢ p_θ(qᵢ | q<ᵢ)
```

At each position, the SSM's output `yᵢ` is projected to a vector of `V`
logits — one score per vocabulary entry — and softmax turns those into a
probability distribution over the next token. Training maximizes the
probability of the *true* next token, which is minimizing **cross-entropy**:

```
L = −(1/N) ∑ᵢ log p_θ(qᵢ_true | q<ᵢ)
```

This is the same next-token-prediction objective that trains language
models — the tokens just happen to be visual. The head outputs `V` logits,
and `V` here is the tokenizer's vocabulary (12,800 for the default FSQ
levels). That's the direct link back to Phase 1's `TokenSpec.vocab_size`.

> **Code decision.** The prediction head is `Linear(d_model, vocab_size)`;
> the loss is `F.cross_entropy` over flattened positions. The head's output
> dimension *must* equal the frozen tokenizer's `vocab_size` — read it from
> the `TokenSpec`, never hardcode. A mismatch is the world-model analog of
> the encoder/quantizer dimension coupling from Phase 1.

---

## Stage 05 — Teacher forcing: the training/inference gap

**Derives:** why training feeds ground truth but inference feeds
predictions — Phase 2's subtlest concept.

During *training*, at every position the model is given the **true** previous
tokens as context, even when predicting position `i` — it never sees its own
mistakes. This is **teacher forcing**, and it's what makes the parallel
training path possible: every position's input is known ahead of time, so the
whole sequence trains at once.

```
context (always ground truth): q<ᵢ_true  →  p_θ(·)  →  target: qᵢ_true
```

During *inference*, there is no ground truth for the future — so the model
feeds its *own* predictions back in as context for the next step. This
asymmetry is the crux: the model is trained on a distribution of perfect
contexts but deployed on a distribution of its-own-possibly-wrong contexts.
This is Phase 2's equivalent of the straight-through estimator — a subtle
concept where a misunderstanding produces code that trains beautifully and
generates garbage.

> **Pitfall.** The classic bug: accidentally leaking the target into the
> context (an off-by-one in the shift), so the model "predicts" a token it was
> already shown. Training loss plummets to near-zero, and generation is
> nonsense. Guard with a test that shifts inputs/targets explicitly and
> asserts position `i` never sees token `i`.

> **Code decision.** Training: inputs are the sequence, targets are the
> sequence shifted left by one; the causal SSM guarantees position `i` only
> sees `<i`. Inference (Phase 3): a generate loop that appends each sampled
> token and re-feeds it. Both are tested — training with a shift-correctness
> assertion, inference with a short rollout.

---

## Stage 06 — Action conditioning: working both options

**Derives:** the actual decision — how actions enter the model — by deriving
both candidates and comparing.

The model must predict `qₜ₊₁` *conditioned on the action* `aₜ` — the same
frame plus "move left" versus "move right" yields different futures. There are
two principled ways to inject the action.

### Option A — interleave actions as tokens in the sequence

Give each action its own embedding and splice it into the stream between
frames:

```
[ qₜ^(0,0)…qₜ^(7,7) | aₜ | qₜ₊₁^(0,0)… ]
   64 frame tokens    action   next frame
```

The action becomes just another token the SSM's state absorbs. When the model
then predicts the next frame's tokens, its state already carries the action —
conditioning is automatic, through the same recurrence that handles everything
else.

### Option B — add action as a conditioning signal

Keep the sequence purely visual, and inject the action embedding *additively*
(or via a learned modulation) into the model's state at the frame boundary:

```
h̃ = h + Embed(aₜ)   at each frame boundary, before predicting qₜ₊₁
```

Here the action never occupies a sequence position; it conditions the state
directly, more like the cross-attention conditioning diffusion world models
use.

### Comparison

| Axis | A · interleaved token | B · conditioning signal |
|---|---|---|
| fit with SSM | native — just another token in the same recurrence | bolted-on — needs a separate injection path outside the scan |
| sequence length | +1 token per frame (negligible at 64 tokens/frame) | no change |
| implementation | one embedding table shared with the vocab stream; sequence builder splices | separate modulation layer at frame boundaries; more moving parts |
| action-space change | grow the embedding table; `action_space_version` guards it | same, plus re-verify the injection path |
| failure mode | action token mis-positioned in the splice (caught by a sequence test) | injection silently averaged away by the recurrence (harder to detect) |

> **Verdict — Option A (interleaved).** It's *Mamba-native*: the action rides
> the identical selective recurrence as every visual token, so conditioning
> needs no separate pathway, no extra layer, no special-case in the scan.
> Selectivity (stage 03) even lets the model learn how strongly each action
> should perturb the state. Option B's separate injection is the pattern
> diffusion models need because they lack a natural sequence position for the
> action — but an SSM *has* that position for free. Fewer moving parts, one
> embedding space, and the failure mode is a splice bug a test catches rather
> than a silent averaging.

> **Code decision (settles the architecture).** Interleave:
> `[frame tokens][action token][frame tokens]…`, one embedding table spanning
> `vocab_size + num_actions`. The sequence builder owns the splice and is
> tested for exact positioning. This is why Phase 0's `action_space_version`
> mattered — the action set is now baked into the model's embedding table, so
> changing it grows the table, not silently reindexes it.

---

## Stage 07 — Rollout and drift

**Derives:** why generated futures degrade over time, and what to measure.

At inference the model generates a frame, feeds it back, generates the next,
and so on — a **rollout**. Each prediction carries a small error; feeding a
slightly-wrong frame back as context means the next prediction is conditioned
on an imperfect input, so errors *compound*. After enough steps the generated
world drifts away from anything plausible. This is the fundamental hard
problem of world models, and it's a form of the training/inference mismatch
from stage 05: the model never trained on its own imperfect outputs.

```
error(t) ≳ error(t−1)   — accumulation, not cancellation, is the default
```

The point of naming this now is measurement: the world model's success metric
isn't single-step accuracy alone, it's *how many steps of coherent rollout* it
sustains before drift. That's the evaluation the phase is built around, the
analog of Phase 1's reconstruction PSNR.

> **Code decision.** Evaluation measures both single-step next-token accuracy
> (cheap, per-position) and N-step rollout coherence (generate N frames from a
> seed, compare against the true continuation). Rollout length before drift is
> the headline metric. Mitigations (scheduled sampling, feeding the model its
> own predictions during training) are later tuning, not the first build.

---

## Stage 08 — The full objective

**Derives:** everything above, assembled into the single quantity training
minimizes.

Tokenize frames through the frozen encoder, interleave actions, run the
selective recurrence with teacher forcing, predict each next token, and
minimize cross-entropy against the true next token across the interleaved
sequence `s = [q₀, a₀, q₁, a₁, …]`:

```
min_θ  E_episode[ −∑_{i ∈ visual positions} log p_θ(sᵢ_true | s<ᵢ) ],
   where s = interleave(E(x_{0:T}), a_{0:T})
```

Reading it recovers the whole phase: take an episode, tokenize every frame
with the frozen Phase-1 artifact, interleave the actions, feed the sequence
through the selective SSM with the true tokens as context, and penalize the
model wherever its predicted distribution puts low probability on the actual
next visual token. One loss, one recurrence, one embedding space. The action
positions aren't scored — the model predicts frames, not actions.

> **Code decision (assembles the whole phase).** The training step:
> `seq = build_interleaved(frozen_tokenizer.encode(frames), actions)` →
> `logits = world_model(seq[:-1])` →
> `loss = cross_entropy(logits[visual], seq[1:][visual])` → `backward`. Same
> device/AMP/checkpoint-resume scaffolding as Phase 1's loop, reused.

---

*continuum · phase 2 derivation roadmap · math precedes code · selective SSM world model*

*next: 2.0 — sequence dataset over the frozen tokenizer + the `SequenceMixer`
interface (decision 0007)*