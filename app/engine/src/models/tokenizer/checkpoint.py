from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from models.tokenizer.config import TokenizerConfig
from models.tokenizer.tokenizer import Tokenizer


def save_checkpoint(
  path: Path | str,
  tokenizer: Tokenizer,
  optimizer: torch.optim.Optimizer,
  scaler: torch.cuda.amp.GradScaler | None,
  step: int,
  extra: dict[str, Any] | None = None,
) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  checkpoint = {
    "step": step,
    "model_state": tokenizer.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scaler_state": scaler.state_dict() if scaler is not None else None,
    "config": asdict(tokenizer.config),
    "token_spec": tokenizer.token_spec.to_dict(),
    "extra": extra or {},
  }
  torch.save(checkpoint, path)


def load_checkpoint(
  path: Path | str,
  map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
  return torch.load(Path(path), map_location=map_location, weights_only=False)


def build_tokenizer_from_checkpoint(
    path: Path | str,
    map_location: str | torch.device = "cpu",
) -> tuple[Tokenizer, dict[str, Any]]:
  ckpt = load_checkpoint(path, map_location=map_location)
  config = TokenizerConfig(**{**ckpt["config"], "levels": tuple(ckpt["config"]["levels"])})
  model = Tokenizer(config)
  model.load_state_dict(ckpt["model_state"])
  return model, ckpt