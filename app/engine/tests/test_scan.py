"""Verification for the parallel associative scan that replaced the O(L)
Python recurrence in SelectiveSSM.forward.

The load-bearing test is `test_parallel_matches_sequential`: the Hillis-Steele
prefix scan must compute EXACTLY the sequential recurrence h_t = a_t*h_{t-1}+u_t.
A wrong shift or pad value would corrupt training silently. Non-power-of-2
lengths are included on purpose — that is where an off-by-one in the doubling
schedule shows up.
"""

from __future__ import annotations

import torch

from models.world_model.layers.scan import associative_scan


def _sequential(a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
  h = torch.zeros_like(a[:, 0])
  outs = []
  for t in range(a.shape[1]):
    h = a[:, t] * h + u[:, t]
    outs.append(h)
  return torch.stack(outs, dim=1)


def test_parallel_matches_sequential() -> None:
  torch.manual_seed(0)
  for length in (1, 2, 5, 8, 17, 33):  # include non-powers-of-two
    a = torch.rand(2, length, 3, 4) * 0.9 + 0.05  # in (0, 1)
    u = torch.randn(2, length, 3, 4)
    fast = associative_scan(a, u)
    slow = _sequential(a, u)
    assert torch.allclose(fast, slow, atol=1e-5), (
      f"scan mismatch at L={length}: {(fast - slow).abs().max().item()}"
    )


def test_forward_still_matches_naive_reference() -> None:
  """The parallel forward must still equal the fully-explicit oracle — this is
  the same guarantee test_ssm makes, restated against the new scan path."""
  from models.world_model.layers.ssm import SelectiveSSM

  torch.manual_seed(0)
  m = SelectiveSSM(d_model=4, d_state=3).eval()
  x = torch.randn(2, 9, 4)
  with torch.no_grad():
    assert torch.allclose(m(x), m.naive_reference(x), atol=1e-5)


def test_scan_is_causal() -> None:
  torch.manual_seed(0)
  a = torch.rand(1, 6, 2, 2) * 0.9 + 0.05
  u = torch.randn(1, 6, 2, 2)
  h1 = associative_scan(a, u)
  u2 = u.clone()
  u2[0, 4] += 3.0
  h2 = associative_scan(a, u2)
  assert torch.allclose(h1[:, :4], h2[:, :4]), "future leaked backward"
  assert not torch.allclose(h1[:, 4:], h2[:, 4:])