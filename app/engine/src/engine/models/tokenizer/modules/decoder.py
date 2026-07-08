from __future__ import annotations

from torch import Tensor
import torch.nn as nn

class Decoder(nn.Module):
  def __init__(self, out_channels: int = 3, hidden: int = 64, latent_dim: int = 5)-> None:
    super().__init__()

    self.net = nn.Sequential(
      nn.Conv2d(latent_dim, hidden * 2, kernel_size=1),
      nn.GroupNorm(8, hidden * 2),
      nn.SiLU(),
      nn.ConvTranspose2d(hidden * 2, hidden * 2, kernel_size=4, stride=2, padding=1),
      nn.GroupNorm(8, hidden * 2),
      nn.SiLU(),
      nn.ConvTranspose2d(hidden * 2, hidden, kernel_size=4, stride=2, padding=1),
      nn.GroupNorm(8, hidden),
      nn.SiLU(),
      nn.ConvTranspose2d(hidden, out_channels, kernel_size=4, stride=2, padding=1),
      nn.Tanh(),
    )
  
  def forward(self, z: Tensor)-> Tensor:
    return self.net(z)