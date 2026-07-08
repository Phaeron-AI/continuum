from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler

from engine.models.device import resolve_device
from engine.models.tokenizer.tokenizer import Tokenizer
from engine.models.tokenizer.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
  max_steps: int = 20_000
  lr: float = 3e-4
  log_every: int = 100
  checkpoint_every: int = 2_000
  device: str = "auto"
  amp: bool = True  # Mixed-Precision Training
  grad_clip: float = 1.0
  checkpoint_dir: str = "checkpoints/tokenizer"

  def __post_init__(self)-> None:
    if self.max_steps <= 0:
      raise ValueError(f"max_steps must be positive; Got: {self.max_steps}")
    

@dataclass
class TrainState:
  step: int = 0
  last_loss: float = float("nan")


def _infinite(loader: DataLoader):
  while True:
    yield from loader

def train_tokenizer(tokenizer: Tokenizer, loader: DataLoader, config: TrainConfig, resume_state: dict | None = None)-> TrainState:
  device = resolve_device(config.device)
  tokenizer.to(device)
  tokenizer.train()

  optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=config.lr)
  use_amp = config.amp and device.type == "cuda"
  scaler = GradScaler(device.type, enabled=use_amp)

  state = TrainState()
  if resume_state is not None:
    optimizer.load_state_dict(resume_state["optimizer_state"])
    if resume_state.get("scaler_state") is not None:
      scaler.load_state_dict(resume_state["scaler_state"])
    state.step = resume_state["step"]
    logger.info("Resumed from step %d", state.step)
  
  batches = _infinite(loader)
  ckpt_dir = Path(config.checkpoint_dir)

  while state.step <= config.max_steps:
    x = next(batches).to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=use_amp):
      recon, _indices = tokenizer(x)
      loss = F.mse_loss(recon, x)
    scaler.scale(loss).backward()

    if config.grad_clip > 0:
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), config.grad_clip)
    scaler.step(optimizer)
    scaler.update()

    state.step += 1
    state.last_loss = float(loss.detach().cpu())

    if state.step % config.log_every == 0:
      logger.info("step %d | recon_mse %.6f", state.step, state.last_loss)

    if state.step % config.checkpoint_every == 0:
      save_checkpoint(
        ckpt_dir / f"step_{state.step:07d}.pt",
        tokenizer, optimizer, scaler, state.step, # type: ignore
      )
  
  save_checkpoint(
    ckpt_dir / f"step_{state.step:07d}.pt",
    tokenizer, optimizer, scaler, state.step, # type: ignore
  )

  return state