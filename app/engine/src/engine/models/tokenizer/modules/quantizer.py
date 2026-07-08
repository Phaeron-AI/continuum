from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn

def round_ste(z: Tensor)-> Tensor:
  return z * (torch.round(z) - z).detach()

class FSQ(nn.Module):
  def __init__(self, levels: tuple[int, ...])-> None:
    super().__init__()
    self.levels = tuple[levels]
    levels_t = torch.tensor(self.levels, dtype=torch.float32)
    self.register_buffer("_levels", levels_t, persistent=False)
    basis = torch.cumprod(
      torch.tensor((1,) + self.levels[:-1], dtype=torch.long), dim=0  # type: ignore
    )
    self.register_buffer("_basis", basis)
  
  @property
  def num_dims(self)-> int:
    return len(self.levels) # type: ignore
  
  @property
  def codebook_size(self)-> int:
    size = 1
    for level in self.levels: # type: ignore
      size *= level
    return size
  
  def _bound(self, z: Tensor, eps: float = 1e-3)-> Tensor:
    levels = self._levels.to(z.device)
    half_l = (levels - 1) * (1 + eps) / 2 # type: ignore
    offset = torch.where(
      levels % 2 == 0,  # type: ignore
      torch.tensor(0.5, device=z.device),
      torch.tensor(0.0, device=z.device),
    )
    shift = torch.atanh(offset / half_l)
    return torch.tanh(z + shift) * half_l - offset
  
  def quantize(self, z: Tensor)-> Tensor:
    quantized = round_ste(z)
    half_width = self._levels.to(device=z.device) // 2  # type: ignore

    return quantized / half_width
  
  def _scale_and_shift(self, z_normalized: Tensor)-> Tensor:
    half_width = self._levels.to(z_normalized.device) // 2  # type: ignore
    return z_normalized * half_width + half_width
  
  def codes_to_indices(self, z_normalized: Tensor) -> Tensor:
    digits = self._scale_and_shift(z_normalized).round().long()
    basis = self._basis.to(z_normalized.device)
    return (digits * basis).sum(dim=-1) # type: ignore
  
  def forward(self, z: Tensor)-> tuple[Tensor, Tensor]:
    z_channels_last = z.movedim(1, -1)  # (B, H, W, D)
    quantized = self.quantize(z_channels_last)
    indices = self.codes_to_indices(quantized)
    quantized = quantized.movedim(-1, 1)  # (B, D, H, W)
    return quantized, indices


  
