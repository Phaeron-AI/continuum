from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from engine.models.world_model.world_model import WorldModel
from engine.models.world_model.rollout import rollout

@dataclass
class DriftReport:
  token_accuracy_per_step: list[float] = field(default_factory=list)
  psnr_per_step: list[float] = field(default_factory=list)
  drift_horizon: int = 0
  psnr_threshold: float = 20.0

  def summary(self) -> str:
    acc = ", ".join(f"{a:.0%}" for a in self.token_accuracy_per_step)
    psnr = ", ".join(f"{p:.1f}" for p in self.psnr_per_step)
    return (
      f"drift_horizon={self.drift_horizon} frames "
      f"(PSNR>{self.psnr_threshold:.0f}dB) | "
      f"token_acc/step=[{acc}] | psnr/step=[{psnr}]"
    )
  

def _psnr(a: Tensor, b: Tensor, data_range: float = 2.0) -> float:
  mse = float(torch.mean((a - b) ** 2))
  if mse <= 0:
      return float("inf")
  return 10.0 * math.log10((data_range**2) / mse)

@torch.no_grad()
def evaluate_drift(
  model: WorldModel,
  tokenizer,  # FrozenTokenizer
  seed_tokens: Tensor,
  actions: Tensor,
  true_frames_tokens: Tensor,
  num_frames: int,
  psnr_threshold: float = 20.0,
  temperature: float = 0.0,
) -> DriftReport:
  spec = tokenizer.token_spec
  tpf = spec.grid_height * spec.grid_width

  generated = rollout(model, seed_tokens, actions, num_frames, tpf, temperature)

  if true_frames_tokens.shape != generated.shape:
    raise ValueError(
      f"truth shape {tuple(true_frames_tokens.shape)} != generated "
      f"{tuple(generated.shape)}"
    )

  report = DriftReport(psnr_threshold=psnr_threshold)
  horizon = 0

  for f in range(num_frames):
    gen_f = generated[:, f]  # (B, tpf)
    true_f = true_frames_tokens[:, f]  # (B, tpf)

    acc = float((gen_f == true_f).float().mean())
    report.token_accuracy_per_step.append(acc)

    gen_px = tokenizer.decode(
      gen_f.reshape(-1, spec.grid_height, spec.grid_width)
    )
    true_px = tokenizer.decode(
      true_f.reshape(-1, spec.grid_height, spec.grid_width)
    )
    psnr = _psnr(gen_px, true_px)
    report.psnr_per_step.append(psnr)

    if psnr >= psnr_threshold and horizon == f:
      horizon = f + 1

  report.drift_horizon = horizon
  return report
