from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.world_model.layers.mixer import SequenceMixer
from models.world_model.layers.scan import associative_scan


class SelectiveSSM(SequenceMixer):
  def __init__(self, d_model: int, d_state: int = 16) -> None:
    super().__init__(d_model)
    self.d_state = d_state

    a_init = torch.arange(1, d_state + 1, dtype=torch.float32)
    a_init = a_init.unsqueeze(0).repeat(d_model, 1)  # (D, N)
    self.A_log = nn.Parameter(torch.log(a_init))
    self.B_proj = nn.Linear(d_model, d_state, bias=False)
    self.C_proj = nn.Linear(d_model, d_state, bias=False)
    self.delta_proj = nn.Linear(d_model, d_model, bias=True)
    self.D = nn.Parameter(torch.ones(d_model))

  def _discretize(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    A = -torch.exp(self.A_log)
    B_t = self.B_proj(x)
    C_t = self.C_proj(x)
    delta = F.softplus(self.delta_proj(x))
    A_bar = torch.exp(delta.unsqueeze(-1) * A)
    return A_bar, B_t, C_t, delta

  def _recurrence_step(
    self,
    h: Tensor,        # (B, D, N) carried state
    x_t: Tensor,      # (B, D) current input
    A_bar_t: Tensor,  # (B, D, N) discretised transition for this step
    B_t: Tensor,      # (B, N)
    C_t: Tensor,      # (B, N)
    delta_t: Tensor,  # (B, D)
  ) -> tuple[Tensor, Tensor]:
    """One step of the recurrence: advance state, read out. The single source
    of truth for the SSM update, shared by _scan and step() so the parallel
    and recurrent paths cannot drift apart."""
    b_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, D, N)
    h = A_bar_t * h + b_bar * x_t.unsqueeze(-1)  # (B, D, N)
    y_t = torch.einsum("bdn,bn->bd", h, C_t)  # (B, D)
    return h, y_t

  def _scan(
    self, x: Tensor, A_bar: Tensor, B_t: Tensor, C_t: Tensor, delta: Tensor
  ) -> Tensor:
    """Sequential O(L) reference scan. Retained as a readable oracle; the
    hot path is now the parallel associative scan in forward()."""
    batch, length, d_model = x.shape
    h = torch.zeros(batch, d_model, self.d_state, device=x.device, dtype=x.dtype)
    outputs = []

    for t in range(length):
      h, y_t = self._recurrence_step(
        h, x[:, t], A_bar[:, t], B_t[:, t], C_t[:, t], delta[:, t]
      )
      outputs.append(y_t)

    y = torch.stack(outputs, dim=1)
    return y + self.D * x

  def _readout(self, x: Tensor, h: Tensor, C_t: Tensor) -> Tensor:
    """State sequence (B, L, D, N) -> output (B, L, D), with the D skip."""
    y = torch.einsum("bldn,bln->bld", h, C_t)
    return y + self.D * x

  def forward(self, x: Tensor) -> Tensor:
    if x.dim() != 3:
      raise ValueError(f"expected (B, L, D), got shape {tuple(x.shape)}")
    if x.shape[-1] != self.d_model:
      raise ValueError(f"last dim {x.shape[-1]} != d_model {self.d_model}")

    A_bar, B_t, C_t, delta = self._discretize(x)  # A_bar (B,L,D,N)
    # u_t = (delta_t * B_t) * x_t, broadcast to (B, L, D, N) — the additive
    # input term of the recurrence h_t = A_bar_t * h_{t-1} + u_t.
    u = (delta.unsqueeze(-1) * B_t.unsqueeze(2)) * x.unsqueeze(-1)
    h = associative_scan(A_bar, u)  # (B, L, D, N), parallel over L
    return self._readout(x, h, C_t)

  def init_state(
    self, batch: int, device: torch.device, dtype: torch.dtype
  ) -> Tensor:
    """Fresh zero hidden state for recurrent generation. (B, D, N)."""
    return torch.zeros(batch, self.d_model, self.d_state, device=device, dtype=dtype)

  def step(self, x_t: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor]:
    """Single-token recurrent generation step (roadmap stage 01).

    x_t: (B, D). state: (B, D, N) carried state, or None to start fresh.
    Returns (y_t (B, D), new_state (B, D, N)). O(1) per token, regardless of
    how many tokens preceded it — this is what makes real-time generation
    possible.
    """
    if x_t.dim() != 2:
      raise ValueError(f"step expects (B, D), got {tuple(x_t.shape)}")
    batch = x_t.shape[0]
    if state is None:
      state = self.init_state(batch, x_t.device, x_t.dtype)

    # Discretise this single token (same math as _discretize, one position).
    A = -torch.exp(self.A_log)  # (D, N)
    B_t = self.B_proj(x_t)  # (B, N)
    C_t = self.C_proj(x_t)  # (B, N)
    delta = F.softplus(self.delta_proj(x_t))  # (B, D)
    A_bar = torch.exp(delta.unsqueeze(-1) * A)  # (B, D, N)

    new_state, y_t = self._recurrence_step(state, x_t, A_bar, B_t, C_t, delta)
    y_t = y_t + self.D * x_t  # skip term, matching forward
    return y_t, new_state

  @torch.no_grad()
  def naive_reference(self, x: Tensor) -> Tensor:
    batch, length, d_model = x.shape
    n = self.d_state
    A = -torch.exp(self.A_log)  # (D, N)

    B_t = self.B_proj(x)  # (B, L, N)
    C_t = self.C_proj(x)  # (B, L, N)
    delta = F.softplus(self.delta_proj(x))  # (B, L, D)

    y = torch.zeros(batch, length, d_model, device=x.device, dtype=x.dtype)
    for b in range(batch):
      h = torch.zeros(d_model, n, device=x.device, dtype=x.dtype)  # (D, N)
      for t in range(length):
        for d in range(d_model):
          dt = delta[b, t, d]
          for i in range(n):
            a_bar = torch.exp(dt * A[d, i])
            b_bar = dt * B_t[b, t, i]
            h[d, i] = a_bar * h[d, i] + b_bar * x[b, t, d]
          y[b, t, d] = torch.dot(h[d], C_t[b, t])
    return y + self.D * x