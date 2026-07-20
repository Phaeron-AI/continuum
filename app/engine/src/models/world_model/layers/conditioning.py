from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

def positional_action_ids(ids: Tensor, vocab_size: Tensor)-> Tensor:
  if ids.dim() != 2:
    raise ValueError(f"Expected: (B, L); got: {tuple(ids.shape)}")
  
  batch, length = ids.shape # (B, L)
  is_action = ids >= vocab_size

  action_val = (ids - vocab_size).clamp(min=0)

  positions = torch.arange(length, device=ids.device).expand(batch, length)

  last_action_pos = torch.cummax(torch.where(is_action, positions, 0), dim=1).values
  return action_val.gather(1, last_action_pos)

class ActionFiLM(nn.Module):
  def __init__(self, num_actions: int, d_model: int) -> None:
    super().__init__()
    self.gamma = nn.Embedding(num_actions, d_model)
    self.beta = nn.Embedding(num_actions, d_model)
    nn.init.zeros_(self.gamma.weight)
    nn.init.zeros_(self.beta.weight)
 
  def forward(self, x: Tensor, action_ids: Tensor) -> Tensor:
    g = self.gamma(action_ids)  # (B, L, D)
    b = self.beta(action_ids)  # (B, L, D)
    return x * (1.0 + g) + b
