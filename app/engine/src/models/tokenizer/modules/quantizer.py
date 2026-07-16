"""Finite Scalar Quantization (roadmap stages 03-04).

Canonical FSQ (Mentzer et al., 2023): bound each latent coordinate with tanh,
scale to a symmetric integer grid, round, and pass gradients through the
non-differentiable round() via the straight-through estimator. No learned
codebook, no commitment loss, no collapse (decision 0006).

The implementation matches the reference exactly, including the half-level
offset (via atanh shift) that keeps even-L grids correctly centered, and the
floor-division half-width normalization. Verified against the published
FSQ(levels=[3,5,4]) example: quantize([0.25, 0.6, -7]) -> [0.0, 0.5, -1.0],
index 10.

Shapes: operates on the last dim. Input (..., D) with D == len(levels);
output same shape, values on the finite grid. codes_to_indices collapses the
D-vector at each cell into a single integer token in [0, prod(levels)).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def round_ste(z: Tensor) -> Tensor:
  return z + (torch.round(z) - z).detach()


class FSQ(nn.Module):
  def __init__(self, levels: tuple[int, ...]) -> None:
    super().__init__()
    self.levels = tuple(levels)
    levels_t = torch.tensor(self.levels, dtype=torch.float32)
    self.register_buffer("_levels", levels_t, persistent=False)
    basis = torch.cumprod(
      torch.tensor((1,) + self.levels[:-1], dtype=torch.long), dim=0
    )
    self.register_buffer("_basis", basis, persistent=False)

  @property
  def num_dims(self) -> int:
    return len(self.levels)

  @property
  def codebook_size(self) -> int:
    size = 1
    for level in self.levels:
      size *= level
    return size

  def _bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
    """Bound z into the grid range with tanh, plus the atanh half-step
    shift for even L so round() lands symmetrically (matches reference)."""
    levels = self._levels.to(z.device)
    half_l = (levels - 1) * (1 + eps) / 2 # type: ignore
    offset = torch.where(
      levels % 2 == 0,  # type: ignore
      torch.tensor(0.5, device=z.device),
      torch.tensor(0.0, device=z.device),
    )
    shift = torch.atanh(offset / half_l)
    return torch.tanh(z + shift) * half_l - offset

  def quantize(self, z: Tensor) -> Tensor:
    """Bound then round-with-STE, normalized onto a stable width for the
    decoder. Channel dim must be last (..., D)."""
    quantized = round_ste(self._bound(z))
    half_width = self._levels.to(z.device) // 2 # type: ignore
    return quantized / half_width

  def _scale_and_shift(self, z_normalized: Tensor) -> Tensor:
    half_width = self._levels.to(z_normalized.device) // 2  # type: ignore
    return z_normalized * half_width + half_width

  def codes_to_indices(self, z_normalized: Tensor) -> Tensor:
    """Map quantized codes (normalized) to integer token ids via
    mixed-radix encoding. Input (..., D) -> output (...) long."""
    digits = self._scale_and_shift(z_normalized).round().long()
    basis = self._basis.to(z_normalized.device)
    return (digits * basis).sum(dim=-1) # type: ignore
  
  def indices_to_codes(self, indices: Tensor) -> Tensor:
    """Inverse of codes_to_indices: integer token ids -> normalized codes.

    Mixed-radix decode each id into per-dimension digits in [0, level), then
    un-shift back onto the normalized grid the decoder expects — the exact
    inverse of _scale_and_shift (digit = z_norm * half_width + half_width).
    Input (...) long -> output (..., D).
    """
    basis = self._basis.to(indices.device)          # (D,) long
    levels = self._levels.to(indices.device).long()  # (D,) # type: ignore
    # recover digits in [0, level) by mixed-radix division
    digits = (indices.unsqueeze(-1) // basis) % levels  # (..., D)  # type: ignore
    # invert _scale_and_shift:  z_norm = (digit - half_width) / half_width
    half_width = levels // 2
    return (digits.float() - half_width.float()) / half_width.float()

  def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
    """z: (B, D, H, W). Returns (quantized (B, D, H, W), indices (B, H, W)).

    Moves the channel dim last for per-dimension quantization, then back.
    """
    z_channels_last = z.movedim(1, -1)  # (B, H, W, D)
    quantized = self.quantize(z_channels_last)
    indices = self.codes_to_indices(quantized)  # (B, H, W)
    quantized = quantized.movedim(-1, 1)  # back to (B, D, H, W)
    return quantized, indices