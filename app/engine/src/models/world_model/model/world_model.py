from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.world_model.layers.block import SSMBlock
from models.world_model.layers.embedding import TokenEmbedding
from models.world_model.layers.mixer import make_mixer
from models.world_model.model.config import WorldModelConfig


class WorldModel(nn.Module):
  def __init__(self, config: WorldModelConfig)-> None:
    super().__init__()
    self.config = config

    self.embedding = TokenEmbedding(config.vocab_size, config.num_actions, config.d_model)
    self.blocks = nn.ModuleList([
      SSMBlock(
        make_mixer(
          config.mixer,config.d_model, d_state=config.d_state
        ),
        config.d_model,
        config.ffn_mult
      ) for _ in range(config.n_layers)
    ])

    self.norm_out = nn.LayerNorm(config.d_model)
    self.head = nn.Linear(config.d_model, config.vocab_size)
  
  def forward(self, ids: Tensor)-> Tensor:
    x = self.embedding(ids)  # (B, L, D)
    for block in self.blocks:
      x = block(x)
    x = self.norm_out(x)
    return self.head(x)  # (B, L, vocab_size)
  
  def loss(self, ids: Tensor)-> Tensor:
    if ids.dim() != 2:
      raise ValueError(f"expected (B, L) ids, got {tuple(ids.shape)}")
    if ids.shape[1] < 2:
      raise ValueError("sequence must have at least 2 tokens to shift")
    
    inputs = ids[:, :-1]  # (B, L-1)
    targets = ids[:, 1:]  # (B, L-1)

    logits = self(inputs)  # (B, L-1, vocab_size)

    is_visual = ~self.embedding.is_action_token(targets)  # (B, L-1)

    flat_logits = logits.reshape(-1, self.config.vocab_size)
    flat_targets = targets.reshape(-1)
    flat_mask = is_visual.reshape(-1)

    if not bool(flat_mask.any()):
      raise ValueError("no visual targets in batch — nothing to train on")

    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
  
  @torch.no_grad()
  def predict_next(self, ids: Tensor)-> Tensor:
    logits = self(ids)[:, -1]  # (B, vocab_size) — last position
    return logits.argmax(dim=-1)