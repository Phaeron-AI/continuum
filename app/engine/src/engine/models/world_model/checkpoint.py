from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import asdict

import torch
from torch.optim.optimizer import Optimizer
from torch.amp.grad_scaler import GradScaler

from engine.models.world_model.world_model import WorldModel
from engine.models.world_model.config import WorldModelConfig

def save_checkpoint(
  path: Path | str,
  model: WorldModel,
  optimizer: Optimizer | None,
  scaler: GradScaler | None,
  step: int,
  extra: dict[str, Any] | None = None
)-> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  checkpoint = {
    "step": step,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),  # type: ignore
    "scaler_state": scaler.state_dict() if scaler is not None else None,
    "config": asdict(model.config),
    # The guard: which tokenizer's token ids this model speaks.
    "tokenizer_version": model.config.tokenizer_version,
    "extra": extra or {},
  }

  torch.save(checkpoint, path)

def load_checkpoint(
  path: Path | str,
  map_location: str | torch.device = "cpu"
)-> dict[ str, Any]:
  return torch.load(Path(path), map_location=map_location, weights_only=False)

def build_world_model_from_checkpoint(
  path: Path | str,
  map_location: str | torch.device = "cpu",
  expect_tokenizer_version: str | None = None,
)-> tuple[WorldModel, dict[str, Any]]:
  ckpt = load_checkpoint(path, map_location=map_location)

  saved_version = ckpt.get("tokenizer_version")
  if (
    expect_tokenizer_version is not None
    and saved_version != expect_tokenizer_version
  ):
    raise ValueError(
      f"world model was trained against tokenizer_version "
      f"{saved_version!r}, but {expect_tokenizer_version!r} is in use — "
      f"the token ids do not mean the same thing; retrain or rebuild"
    )

  config = WorldModelConfig(**ckpt["config"])
  model = WorldModel(config)
  model.load_state_dict(ckpt["model_state"])
  return model, ckpt