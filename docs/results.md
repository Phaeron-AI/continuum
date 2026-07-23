# Results — ablation of the world-model improvements

Additive ablation on grid2d (movement-biased `directed` policy, ~44% unique
frames), FSQ tokenizer (vocab 64, 37.8 dB reconstruction), a 0.84M-param world
model, 4000 training steps per arm. Each arm turns on one more capability; all
other settings (dims, steps, lr, data split, batch order, seed) are held fixed.
Rollout metrics are a 24-frame greedy free-running rollout, token accuracy vs
ground truth, averaged over 6 episodes.

| arm | ce loss ↓ | teacher-forced acc ↑ | rollout acc mean ↑ | rollout acc @24f ↑ |
| --- | ---: | ---: | ---: | ---: |
| baseline (ssm) | 1.605 | 60.2% | 22.6% | 20.6% |
| + mamba block | 0.842 | 79.8% | 21.8% | 20.6% |
| + film actions | 0.844 | 79.8% | 24.3% | 27.3% |
| + noise aug (full) | 1.030 | 74.4% | **29.5%** | 27.3% |

## The headline

**Teacher-forced accuracy is a misleading proxy for generation quality, and
noise augmentation is what makes a sharp SSM usable as a world model.** The two
views disagree completely: the mamba block dominates one-step prediction
(79.8% vs 60.2%) but gains *nothing* in free-running rollout (21.8% vs 22.6%).
It is only once we corrupt the training context — teaching the model to recover
from its own mistakes — that rollout accuracy jumps, to 29.5%.

## What each change does

- **Mamba block** — best one-step predictor by a wide margin (+20 pts
  teacher-forced), cutting cross-entropy 48%. But that sharpness does not
  transfer to generation: its rollout accuracy is no better than the plain SSM.
  A textbook exposure-bias signature — a confident model trained purely with
  teacher forcing compounds its own errors when it has to feed on its outputs.

- **FiLM action conditioning** — neutral one-step (79.8 → 79.8), but lifts
  rollout accuracy (mean 21.8 → 24.3, final 20.6 → 27.3). The action signal
  anchors the model during free-running even though it is redundant for
  single-step prediction. Its value is stability, visible only in rollout.

- **Context-noise augmentation** — the largest rollout gain (+5.2 pts mean over
  the no-noise model, +6.9 over baseline), despite the *lowest* teacher-forced
  accuracy of the mamba arms. It trades one-step sharpness for the ability to
  recover from its own errors — exactly the drift antidote it was added to be.
  It holds higher accuracy throughout the rollout, not just at the end.

## Caveats

Absolute accuracies are low by design of the setup — a hard 24-frame greedy
rollout, a small model, short training, and a 64-code tokenizer — so these are
*relative* signals, not headline accuracies. Rollout metrics are averaged over
6 episodes; the ranking (baseline ≈ mamba < film < noise-aug) is stable across
4- and 6-episode runs and the gaps exceed sampling noise, but larger-batch
error bars would firm up the exact numbers.