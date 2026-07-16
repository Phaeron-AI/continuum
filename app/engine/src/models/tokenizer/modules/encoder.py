from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
  def __init__(self, in_channels: int = 3, hidden: int = 64, latent_dim: int = 5)-> None:
    super().__init__()
    # Three stride-2 halvings: 64 -> 32 -> 16 -> 8.
    self.net = nn.Sequential(
      nn.Conv2d(in_channels, hidden, kernel_size=4, stride=2, padding=1),
      nn.GroupNorm(8, hidden),
      nn.SiLU(),
      nn.Conv2d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1),
      nn.GroupNorm(8, hidden * 2),
      nn.SiLU(),
      nn.Conv2d(hidden * 2, hidden * 2, kernel_size=4, stride=2, padding=1),
      nn.GroupNorm(8, hidden * 2),
      nn.SiLU(),
      # Project to the low FSQ dimension (1x1, preserves the 8x8 grid).
      nn.Conv2d(hidden * 2, latent_dim, kernel_size=1),
    )
  
  def forward(self, x: Tensor) -> Tensor:
    return self.net(x)