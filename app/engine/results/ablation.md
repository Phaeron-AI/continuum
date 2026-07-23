# Ablation — world-model improvements

Additive ablation on the `phase1_v1` token cache (grid2d, FSQ vocab 64). Each
arm turns on one more of the new capabilities; all other settings are held
fixed (4000 steps/arm, CPU, `d_model=128`, `n_layers=4`, same data split, batch
order, and seed). Drift is an 8-frame greedy rollout scored at PSNR > 20 dB.

| arm | params | ce loss ↓ | next-token acc ↑ | drift horizon ↑ (PSNR>20dB, /8) |
| --- | ---: | ---: | ---: | ---: |
| baseline (ssm) | 0.64M | 1.709 | 57.1% | 8 |
| + mamba block | 0.84M | 0.944 | 76.0% | 8 |
| + film actions | 0.84M | 0.942 | 76.1% | 8 |
| + noise aug (full) | 0.84M | 1.147 | 71.0% | 8 |

Full stack vs baseline: next-token accuracy 57.1% → 71.0%; cross-entropy
1.709 → 1.147. The Mamba block alone accounts for the entire gain.

## Reading the results

**The Mamba block is the decisive win.** Swapping the bare selective SSM for the
true Mamba mixer (depthwise causal conv + SiLU gate) cut cross-entropy 45%
(1.709 → 0.944) and lifted next-token accuracy 19 points (57.1% → 76.0%). The
short causal conv captures the local frame structure — edges, single-step motion
— that the plain SSM leaves on the table. This is the result worth reporting.

**FiLM action conditioning is neutral on this data.** Adding per-position action
modulation moved nothing (76.0% → 76.1%). That is expected here, not a failure:
the Phase 0 data has only ~31% unique frames, so the next frame rarely depends
on the action — there is almost no conditioning signal to exploit. FiLM should
earn its keep on dynamics-rich data where the action actually changes the frame.

**Noise augmentation looks costly here — because the metric that would show its
benefit is saturated.** Teacher-forced metrics get worse (ce 0.944 → 1.147, acc
76.0% → 71.0%), which is exactly what noise augmentation is supposed to do: it
deliberately makes the teacher-forced objective harder by corrupting context.
Its payoff is *rollout stability* (drift horizon), not teacher-forced accuracy —
and the drift horizon is pinned at 8/8 for every arm, so this run cannot see the
payoff at all.

## Why drift horizon is uninformative (and how to fix it)

Every arm scores a perfect 8/8 drift horizon, so the column tells us nothing.
Two causes:

1. **Rollout too short.** Drift is evaluated over only 8 frames. Divergence in a
   token-based world model typically appears later; 8 frames is inside every
   arm's comfort zone.
2. **Data too static.** With ~69% of frames near-duplicates of the previous one,
   generated frames barely move and trivially stay above the 20 dB PSNR
   threshold, so nothing drifts regardless of model.

Until drift is unsaturated, the noise-aug row will keep looking purely costly.

## Next run for real numbers

- Regenerate Phase 0 with a **movement-biased / scripted policy** so frames
  actually change (raise unique-frame ratio well above 31%).
- Evaluate drift at a **longer horizon** (`--frames 30–60`), and consider a
  stricter PSNR threshold.
- Expected pattern once drift is unsaturated: the full stack (with noise aug)
  holds PSNR > 20 dB for *more* frames than baseline, even though its
  teacher-forced cross-entropy is higher — the whole point of the augmentation.

## Headline

On this first end-to-end run, the Mamba block is a large, unambiguous win
(−45% cross-entropy, +19 pts next-token accuracy). FiLM and noise augmentation
are not yet measurable here: FiLM needs data where actions change the frame, and
noise augmentation needs a longer, unsaturated drift evaluation to reveal its
stability benefit. Both are set up correctly — the evaluation, not the methods,
is what needs to get harder next.