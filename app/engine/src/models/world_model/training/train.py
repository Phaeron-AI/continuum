from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler

from models.device import resolve_device
from models.world_model.model.checkpoint import save_checkpoint
from models.world_model.model.world_model import WorldModel

logger = getLogger(__name__)

@dataclass(frozen=True)
class TrainConfig:
  max_steps: int = 20_000
  lr: float = 3e-4
  log_every: int = 100
  checkpoint_every: int = 2_000
  device: str = "auto"
  amp: bool = True
  grad_clip: float = 1.0
  checkpoint_dir: str = "checkpoints/world_model"

  def __post_init__(self)-> None:
    if self.max_steps <= 0:
      raise ValueError(f"max steps must be positive, Got: {self.max_steps}")
  
@dataclass
class TrainState:
  step: int = 0
  last_loss: float = float("nan")

def _infinite(loader: DataLoader):
  while True:
    yield from loader

def train_world_model(
  model: WorldModel,
  loader: DataLoader,
  config: TrainConfig,
  resume_state: dict | None = None,
) -> TrainState:
  device = resolve_device(config.device)
  model.to(device)
  model.train()

  optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
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

  while state.step < config.max_steps:
    ids = next(batches).to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        loss = model.loss(ids)

    scaler.scale(loss).backward()
    if config.grad_clip > 0:
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    scaler.step(optimizer)
    scaler.update()

    state.step += 1
    state.last_loss = float(loss.detach().cpu())

    if state.step % config.log_every == 0:
      logger.info("step %d | ce_loss %.4f", state.step, state.last_loss)

    if state.step % config.checkpoint_every == 0:
      save_checkpoint(
        ckpt_dir / f"step_{state.step:07d}.pt",
        model, optimizer, scaler, state.step,
      )

  save_checkpoint(
    ckpt_dir / f"step_{state.step:07d}.pt",
    model, optimizer, scaler, state.step,
  )
  return state