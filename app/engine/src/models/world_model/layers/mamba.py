from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.world_model.layers import SelectiveSSM, SequenceMixer

MambaState = tuple[Tensor, Tensor]

class MambaMixer(SequenceMixer):
  def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4)-> None:
    super().__init__(d_model)

    if d_conv < 1:
      raise ValueError(f"d_conv must be >= 1, got {d_conv}")
    
    self.d_state = d_state
    self.d_conv = d_conv

    self.in_proj = nn.Linear(d_model, d_model, bias=False)
    self.conv = nn.Conv1d(
      d_model, d_model, kernel_size=d_conv, groups=d_model, padding=d_conv-1
    )

    self.ssm = SelectiveSSM(d_model, d_state)
    self.z_proj = nn.Linear(d_model, d_model, bias=False)
    self.out_proj = nn.Linear(d_model, d_model, bias=False)
  
  def forward(self, x: Tensor)-> Tensor:
    if x.dim() != 3:
      raise ValueError(f"Expected: (B, L, D); Got: {tuple(x.shape)}")
    if x.shape[-1] != self.d_model:
      raise ValueError(f"last dim: {x.shape[-1]} != d_model: {self.d_model}")
    
    length = x.shape[1]
    xin = self.in_proj(x)

    xc = self.conv(xin.transpose(1, 2))[..., :length].transpose(1, 2)
    xc = F.silu(xc)
    y = self.ssm(xc)  # (B, L, D) — parallel selective scan

    y = y * F.silu(self.z_proj(x))

    return self.out_proj(y)
  
  def init_state(self, batch: int, device: torch.device, dtype: torch.dtype)-> MambaState:
    conv_buf = torch.zeros(
      batch, self.d_model, self.d_conv-1, device=device, dtype=dtype
    )

    return conv_buf, self.ssm.init_state(batch, device, dtype)
  
  def step(self, x_t: Tensor, state: MambaState | None = None)-> tuple[Tensor, MambaState]:
    if x_t.dim() != 2:
      raise ValueError(f"step expects: (B, D); Got: {tuple(x_t.shape)}")
    if state is None:
      state = self.init_state(x_t.shape[0], x_t.device, x_t.dtype)
    conv_buf, h = state

    xin_t = self.in_proj(x_t)

    window = torch.cat([conv_buf, xin_t.unsqueeze(-1)], dim=-1)  # (B, D, k)
    weight = self.conv.weight.squeeze(1)  # (D, k) — depthwise
    xc_t = (window * weight).sum(dim=-1)  # (B, D)
    if self.conv.bias is not None:
      xc_t = xc_t + self.conv.bias
    
    xc_t = F.silu(xc_t)

    new_conv_buf = window[..., 1:]  # drop the oldest input
    y_t, new_h = self.ssm.step(xc_t, h)  # (B, D), (B, D, N)
    y_t = y_t * F.silu(self.z_proj(x_t))
    return self.out_proj(y_t), (new_conv_buf, new_h)