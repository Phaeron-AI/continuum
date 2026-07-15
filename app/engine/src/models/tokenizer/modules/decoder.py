from __future__ import annotations

from torch import nn
from torch import Tensor

class Decoder(nn.Module):
  def __init__(
    self,
    out_channels: int = 3,
    hidden: int = 64,
    latent_dim: int = 5,
  ) -> None:
    super().__init__()
    self.net = nn.Sequential(
      # Lift the low FSQ dim back up (1x1, preserves 8x8).
      nn.Conv2d(latent_dim, hidden * 2, kernel_size=1),
      nn.GroupNorm(8, hidden * 2),
      nn.SiLU(),
      # Three stride-2 doublings: 8 -> 16 -> 32 -> 64.
      nn.ConvTranspose2d(hidden * 2, hidden * 2, kernel_size=4, stride=2, padding=1),
      nn.GroupNorm(8, hidden * 2),
      nn.SiLU(),
      nn.ConvTranspose2d(hidden * 2, hidden, kernel_size=4, stride=2, padding=1),
      nn.GroupNorm(8, hidden),
      nn.SiLU(),
      nn.ConvTranspose2d(hidden, out_channels, kernel_size=4, stride=2, padding=1),
      nn.Tanh(),  # -> [-1, 1], matches data normalization (stage 05)
    )

  def forward(self, z: Tensor) -> Tensor:
    return self.net(z)