from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from engine.models.device import resolve_device
from engine.models.world_model.world_model import WorldModel

@dataclass
class WMEvalReport:
  ce_loss: float
  perplexity: float
  next_token_accuracy: float
  num_visual_targets: int
  vocab_size: int

  def summary(self) -> str:
    return (
      f"visual_targets={self.num_visual_targets} | "
      f"ce_loss={self.ce_loss:.4f} | "
      f"perplexity={self.perplexity:.1f} (random={self.vocab_size}) | "
      f"next_token_acc={self.next_token_accuracy:.1%}"
    )

@torch.no_grad()
def evaluate_world_model(
  model: WorldModel,
  loader: DataLoader,
  device: str = "auto",
  max_batches: int | None = None,
) -> WMEvalReport:
  resolved = resolve_device(device)
  model.to(resolved)
  model.eval()

  vocab = model.config.vocab_size
  total_loss_sum = 0.0
  total_correct = 0
  total_targets = 0

  for i, ids in enumerate(loader):
    if max_batches is not None and i >= max_batches:
      break
    ids = ids.to(resolved)

    # Same shift + mask as training (stages 05, 08).
    inputs, targets = ids[:, :-1], ids[:, 1:]
    logits = model(inputs)
    is_visual = ~model.embedding.is_action_token(targets)

    flat_logits = logits.reshape(-1, vocab)[is_visual.reshape(-1)]
    flat_targets = targets.reshape(-1)[is_visual.reshape(-1)]
    if flat_targets.numel() == 0:
      continue

    loss = F.cross_entropy(flat_logits, flat_targets, reduction="sum")
    total_loss_sum += float(loss.cpu())
    total_correct += int((flat_logits.argmax(dim=-1) == flat_targets).sum().cpu())
    total_targets += int(flat_targets.numel())

  if total_targets == 0:
    raise ValueError("no visual targets in the eval set")

  ce = total_loss_sum / total_targets
  return WMEvalReport(
    ce_loss=ce,
    perplexity=math.exp(min(ce, 700.0)),  # guard overflow on a broken model
    next_token_accuracy=total_correct / total_targets,
    num_visual_targets=total_targets,
    vocab_size=vocab,
  )