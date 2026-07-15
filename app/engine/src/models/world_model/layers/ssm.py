from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.world_model.layers.mixer import SequenceMixer


class SelectiveSSM(SequenceMixer):
  def __init__(self, d_model: int, d_state: int = 16)-> None:
    super().__init__(d_model)
    self.d_state = d_state
    
    a_init = torch.arange(1, d_state + 1, dtype=torch.float32)
    a_init = a_init.unsqueeze(0).repeat(d_model, 1)  # (D, N)
    self.A_log = nn.Parameter(torch.log(a_init))

    self.B_proj = nn.Linear(d_model, d_state, bias=False)
    self.C_proj = nn.Linear(d_model, d_state, bias=False)
    self.delta_proj = nn.Linear(d_model, d_model, bias=True)

    self.D = nn.Parameter(torch.ones(d_model))

  def _discretize(self, x: Tensor)-> tuple[Tensor, Tensor, Tensor, Tensor]:
    A = -torch.exp(self.A_log)

    B_t = self.B_proj(x)
    C_t = self.C_proj(x)

    delta = F.softplus(self.delta_proj(x))

    A_bar = torch.exp(delta.unsqueeze(-1)*A)

    return A_bar, B_t, C_t, delta
  
  def _scan(self, x: Tensor, A_bar: Tensor, B_t: Tensor, C_t: Tensor, delta: Tensor)-> Tensor:
    batch, length, d_model = x.shape
    h = torch.zeros(batch, d_model, self.d_state, device=x.device, dtype=x.dtype)
    outputs = []
    
    for t in range(length):
      b_bar = delta[:, t].unsqueeze(-1) * B_t[:, t].unsqueeze(1)  # (B, D, N)
      h = A_bar[:, t] * h + b_bar * x[:, t].unsqueeze(-1)  # (B, D, N)
      # y_t = C_t . h_t  (contract over the state dim)
      y_t = torch.einsum("bdn,bn->bd", h, C_t[:, t])  # (B, D)
      outputs.append(y_t)
    
    y = torch.stack(outputs, dim=1)
    return y + self.D*x
  
  def forward(self, x: Tensor)-> Tensor:
    if x.dim() != 3:
      raise ValueError(f"expected (B, L, D), got shape {tuple(x.shape)}")
    if x.shape[-1] != self.d_model:
      raise ValueError(f"last dim {x.shape[-1]} != d_model {self.d_model}")

    A_bar, B_t, C_t, delta = self._discretize(x)
    return self._scan(x, A_bar, B_t, C_t, delta)
  
  @torch.no_grad()
  def naive_reference(self, x: Tensor)-> Tensor:
    batch, length, d_model = x.shape
    n = self.d_state
    A = -torch.exp(self.A_log)  # (D, N)

    B_t = self.B_proj(x)  # (B, L, N)
    C_t = self.C_proj(x)  # (B, L, N)
    delta = F.softplus(self.delta_proj(x))  # (B, L, D)

    y = torch.zeros(batch, length, d_model, device=x.device, dtype=x.dtype)
    for b in range(batch):
      # explicit state: (D, N)
      h = torch.zeros(d_model, n, device=x.device, dtype=x.dtype)
      for t in range(length):
        for d in range(d_model):
          dt = delta[b, t, d]
          for i in range(n):
            a_bar = torch.exp(dt * A[d, i])
            b_bar = dt * B_t[b, t, i]
            h[d, i] = a_bar * h[d, i] + b_bar * x[b, t, d]
          y[b, t, d] = torch.dot(h[d], C_t[b, t])
    return y + self.D * x