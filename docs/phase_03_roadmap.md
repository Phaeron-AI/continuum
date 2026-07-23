# Phase 3 — Real-Time Generation: the recurrent path made live

Phase 2 trained a world model on the *parallel* schedule of the selective
recurrence — teacher-forced, a whole episode at once. Phase 3 deploys the
*recurrent* schedule of the identical math: one token at a time, a carried
state, O(1) work per step, fast enough to drive a live client. Nothing about
the model changes; what changes is how it is run, and everything that has to
be built around a stateful, mutable, latency-bound thing.

- **Input:** a frozen trained `WorldModel` + the frozen Phase-1 tokenizer
- **Schedule:** recurrent — carried per-layer state, one token per step
- **Target:** a playable stream — action in, next frame out, at interactive fps

Pipeline: `seed clip → encode → warm state → (action → step ×64 → sample → decode → frame)* → drift eval`

Each stage ends in a **code decision**, and is tagged **[built]**, **[built, hardening]**,
or **[next]** against the current tree — Phase 3 is partly standing already
(stage 3.1 landed the step; the session and server exist), and this doc names
what remains and why the built parts are shaped as they are.

---

## Stage 00 — Objects & notation

**Derives:** the two schedules of one recurrence, and what Phase 2 hands off.

Phase 2 left a trained `p_θ` whose core is the selective recurrence

```
hₜ = Āₜ·hₜ₋₁ + B̄ₜ·xₜ,    yₜ = C ₜ·hₜ
```

run in the **parallel view**: given a whole sequence `s = [q₀, a₀, q₁, …]`, the
scan unrolls every position at once, and teacher forcing supplies every input
ahead of time. That is the *training* schedule, and it is the only schedule
Phase 2 needed.

Generation cannot use it. At inference there is no future to unroll — each
token depends on the one just sampled. So Phase 3 runs the **recurrent view**
of the *same* equation: carry `hₜ`, and at each step consume one token and emit
one distribution. Same weights, same `_recurrence_step`, different schedule.

- The generation state is **per layer**: `state = [h⁽⁰⁾, h⁽¹⁾, …]`, one `(B, D, N)`
  hidden tensor per block, because each block's mixer carries its own state
  independently (`GenerationState = list[Tensor | None]`).
- A **frame** is `tokens_per_frame = h·w = 64` visual tokens; an **action** is one
  token in the shared embedding space (`vocab_size + a`), stepped in before the
  frame it conditions.

> **Code decision.** The frozen artifacts are loaded once and set to `eval()`:
> `build_world_model_from_checkpoint(..., expect_tokenizer_version=...)` pairs
> the model with the exact tokenizer it was trained against — a version
> mismatch is a loud error here, not silent garbage generations later. Phase 3
> imports the model and calls `step`; it never touches the training loop.

---

## Stage 01 — The recurrent step: why O(1) is the whole point

**Derives:** the cost argument that forces a carried-state `step`, and the
equivalence it must satisfy.

Generating `L` tokens by re-running the parallel forward means, at token `t`,
re-reading the whole context of length `t`: that is `∑ₜ O(t) = O(L²)` work to
produce one sequence. For a live session the context only grows — a minute of
play is thousands of tokens — so the quadratic path degrades exactly as the
session gets interesting.

The recurrent step escapes it. Because `hₜ` is a fixed-size summary of *all*
prior tokens, advancing one token needs only `hₜ₋₁` and the current input:
**constant work, constant memory, per token**, regardless of how many tokens
came before. `L` tokens cost `O(L)` total. This is the property that makes
real-time generation possible and is the entire reason the SSM was chosen over
a Transformer for the sequence mixer.

The catch: the recurrent path is a *second* implementation of a recurrence the
parallel path already computes, and two implementations drift. They must be
proven identical, or a state-threading slip becomes an invisible generation
bug — logits that are subtly wrong, never a crash.

> **Code decision. [built]** `WorldModel.step(token, state)` threads one
> `(B, D, N)` state per block through `SSMBlock.step → SelectiveSSM.step`, all
> routed through the same `_recurrence_step` the parallel `_scan` uses — one
> source of truth so the schedules cannot diverge. The load-bearing guarantee
> is `test_model_step_matches_forward`: stepping a sequence token-by-token
> through the full stack must equal the parallel forward to `atol=1e-4`. Note
> the hot path here is `TokenEmbedding.forward`; its optional id range-check is
> gated behind `CONTINUUM_CHECK_TOKEN_IDS` precisely because reading `ids.max()`
> forces a device sync that would stall this loop.

---

## Stage 02 — Rollout: autoregression under its own predictions

**Derives:** how a stream of actions becomes a stream of frames, and the
ordering the conditioning depends on.

A rollout generates a frame, feeds it back, and generates the next. Within one
frame the model emits 64 tokens autoregressively; between frames the action
that *produces* the next frame is stepped in first — the same
`[…frame][action][next frame…]` interleave settled in Phase 2 stage 06. Get
that ordering wrong and the action conditions the wrong frame, or nothing.

The subtle part is the feedback: after the first token, the model consumes its
**own** sample as context for the next — this is where the training/inference
mismatch (Phase 2 stage 05) becomes real and error begins to compound (stage
07 below). Sampling is a knob: `temperature = 0` is greedy `argmax` (used for
reproducible eval); `> 0` softmaxes and draws from the distribution.

There are two implementations, and they serve different masters. The reference
`rollout()` re-runs the parallel `model(context)` each token — legible, and the
numerical oracle — but it is the O(L²) path from stage 01, acceptable only for
short offline eval. The live path must use `step()`.

> **Code decision. [built, hardening]** `_pick(logits, temperature)` owns the
> sampling and is shared by both paths. `rollout()` is the readable oracle;
> `session.advance()` is the O(1) production path over `step()`. The gap to
> close: a step-based rollout so `evaluate_drift` stops paying O(L²) on long
> horizons, plus a parity test asserting `rollout()` (forward) and the
> step-based rollout produce identical tokens at `temperature=0`. Until that
> lands, drift eval is correct but scales quadratically in horizon.

---

## Stage 03 — The session: state you warm once and carry

**Derives:** why generation becomes a stateful object rather than a pure
function, and how it starts.

The parallel forward is a pure function of its input. The recurrent path is
not: its whole speed advantage comes from *not* re-reading history, which means
the history has to live somewhere — in a carried state, mutated in place across
calls. That makes a live generator an **object with a lifecycle**, not a
function.

The lifecycle has three moments. **Open**: warm the state by stepping every
token of a seed clip in order, so the first generated frame is conditioned on
real context — this is the one time we pay to ingest history, and after it the
state stands in for that history forever. **Advance**: one action in, one frame
out, mutating the state. **Close**: discard the state and reclaim its memory.

A running session also accumulates — latency samples, and any retained history
— so it must be *bounded*, or a long-lived play session leaks.

> **Code decision. [built]** `GenerationSession.open(seed_tokens)` steps every
> seed token to warm `self._state`; `advance(action)` steps the action token,
> then generates `tokens_per_frame` tokens via `step()`, feeding each sample
> back — mirroring `rollout()`'s ordering exactly but on the O(1) path.
> Latency history is a `deque(maxlen=512)` so the session is memory-bounded.
> `close()` drops the state.

---

## Stage 04 — The latency budget: measuring honestly

**Derives:** what "real-time" decomposes into, and why naïve timing lies.

A frame's wall-clock time is three parts: **step** (running the model over the
action + 64 tokens), **sample** (turning logits into token ids), and **decode**
(the frozen tokenizer turning the 64-token grid back into pixels). Naming them
separately is what turns "it feels slow" into "decode dominates" or "we are
step-bound" — the difference between optimizing the kernel and optimizing the
tokenizer.

The measurement trap: GPU kernels are asynchronous, so a `perf_counter()`
around a launch times the *launch*, not the *work* — the number is a fiction
until the device is synchronized. Honest per-segment timing requires a
`cuda.synchronize()` before reading the clock.

The headline is per-frame total and its distribution — mean, p50, p95, and the
implied fps — because interactivity is set by the *slow* frames, not the
average.

> **Code decision. [built]** `FrameLatency` records step/sample/decode/total;
> `advance()` brackets each segment with `_sync()` (a no-op off CUDA) so the
> numbers are real. `metrics()` reports mean/p50/p95/fps over the bounded
> history. p95 is deliberately kept alongside the mean — a 30 fps average with
> a 120 ms p95 is not a playable session.

---

## Stage 05 — Serving many sessions at once

**Derives:** what a stateful, mutable generator forces on the server, and where
concurrency bites.

A stateless model server can fan requests out freely. A stateful one cannot:
the recurrent state is mutated in place and is strictly order-dependent, so two
`advance` calls racing on the *same* session interleave their state updates and
corrupt both. Each session therefore needs a lock; different sessions stay
independent and parallel.

Two more consequences follow from state being *memory*. It must be **reclaimed**
— an abandoned session would pin its state forever, so idle sessions time out
and get swept. And the model work is **synchronous** torch that would block the
async event loop, so it must be offloaded to a threadpool, keeping the server
responsive to other sessions while one steps.

> **Code decision. [built, hardening]** `SessionRegistry` holds one
> `SessionEntry` per session with an `asyncio.Lock`; `advance` runs under the
> lock, inside `run_in_threadpool`, so concurrent sessions progress while a
> single session's calls serialize. A background sweep evicts sessions idle
> past `ttl_seconds` and `stop()` closes all on shutdown. Hardening left:
> checkpoint loading uses `weights_only=False` (fine for local trusted
> artifacts, but pin the intent as torch tightens the default), and the
> server is single-process — horizontal scale-out means sticky routing so a
> session's requests always reach the process that holds its state.

---

## Stage 06 — Drift, measured

**Derives:** why single-step accuracy is the wrong headline, and what to
measure instead.

Each generated frame carries a small error; feeding a slightly-wrong frame back
as context makes the next prediction condition on an imperfect input, so errors
**accumulate rather than cancel**. A model with excellent single-step accuracy
can still drift into nonsense after a handful of frames — so the success metric
is not per-step accuracy, it is *how many steps of coherent rollout* the model
sustains before it goes off the rails.

That requires measuring at each rollout **depth**, not in aggregate: token
accuracy against the true continuation, and pixel PSNR after decoding both the
generated and the true grid. The **drift horizon** is the first depth at which
quality falls below a threshold — the analog of Phase 1's reconstruction PSNR,
and the number the phase is judged by.

> **Code decision. [built]** `evaluate_drift` rolls out `num_frames`, and per
> depth records token accuracy and `_psnr` between decoded frames, extending
> `drift_horizon` only while PSNR holds above threshold (monotone in practice,
> but not assumed). `DriftReport.summary()` prints the per-step curve so a
> regression shows *where* it broke, not just that it did. This shares
> `rollout()`, so the stage-02 step-based rewrite speeds drift eval directly.

---

## Stage 07 — Drift, mitigated

**Derives:** the mitigation the mismatch invites, and the cost that makes it a
tuning step rather than the first build.

The drift of stage 06 is the training/inference gap of Phase 2 stage 05 made
visible: the model trained only on *perfect* teacher-forced contexts, then is
deployed on its *own imperfect* ones — a distribution it never saw. The
mitigation is to close that gap during training: sometimes feed the model its
own sampled token instead of the ground truth, so it learns to recover from its
own mistakes. This is **scheduled sampling** — anneal the probability of using a
prediction over the truth as training proceeds.

It is deliberately *not* the first build, because it has a real cost: feeding
the model its own samples reintroduces a sequential dependency mid-sequence,
eroding the full parallelism that made teacher-forced training cheap. The first
build establishes the honest drift metric; only once it plateaus does buying
rollout length with training-time complexity pay off.

> **Code decision. [next]** Add an optional scheduled-sampling schedule to the
> training loop, gated off by default, measured strictly by its effect on
> `drift_horizon` — never on single-step loss, which it can worsen while
> improving the metric that matters. Keep the parallel teacher-forced path as
> the baseline the mitigation must beat on horizon.

---

## Stage 08 — The compiled kernel

**Derives:** what replaces the Python scan, and the oracle that keeps it honest.

The pure-PyTorch `_scan` is a Python loop over the sequence — correct and
legible, and slow, exactly as Phase 2 stage 03 warned: selectivity removes the
global convolution a fixed SSM enjoys, so the reference has no fast path. The
production speedup is the hardware-aware parallel scan in the compiled
`mamba-ssm` kernel, dropped in behind the existing `SequenceMixer` interface so
nothing above the mixer changes.

A kernel swap is the highest-risk change in the phase: a fast path that is
subtly wrong trains beautifully and generates garbage. The safeguard is already
in hand — the reference implementation is a *numerical oracle*. The kernel is
accepted only when it matches the reference `_scan` (and thence
`naive_reference`) to tolerance across shapes.

> **Code decision. [next]** Register a kernel-backed mixer via `make_mixer`
> alongside `identity`/`ssm`, selected by config, and gate it on an
> `allclose` parity suite against the reference across `(d_model, d_state, L,
> batch)` — the same equivalence discipline as stages 01 and 02, now against a
> real oracle rather than a self-check. Recurrent `step()` stays as-is; the
> kernel accelerates the training/parallel path.

---

## Stage 09 — The full loop assembled

**Derives:** everything above, run end to end.

Load the frozen model and tokenizer, paired by `tokenizer_version`. Open a
session on a seed clip — encode its frames to token grids, flatten row-major
(the stage-01 contract), and warm the state by stepping every token. Then, per
action: step the action token, generate 64 frame tokens on the O(1) recurrent
path feeding each sample back, decode the grid to pixels, and emit the frame —
timing each segment honestly. Periodically evaluate drift against a held-out
true continuation and read the horizon.

```
model, tok = build_world_model_from_checkpoint(ckpt, expect_tokenizer_version=tok.version)
s = GenerationSession(model, tok); s.open(encode(seed_clip))
for a in action_stream:            # live
    frame = s.advance(a, temperature=τ)
report = evaluate_drift(model, tok, seed, actions, truth, num_frames=N)
```

One state, two schedules (parallel trains it, recurrent runs it), one embedding
space (visual tokens and actions share the table), one metric that judges it
(rollout length before drift). That is the whole phase: the Phase-2 recurrence,
made live, bounded, served, measured, and — stages 07–08 — made faster and made
to last.

---

*continuum · phase 3 build roadmap · math precedes code · the recurrent schedule of the selective SSM*

*status: 3.1 step + per-layer state, session, latency, and server are built;
next — step-based rollout + parity (3.2), then drift mitigation (3.7) and the
mamba-ssm kernel with parity (3.8)*
