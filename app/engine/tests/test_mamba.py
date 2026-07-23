"""Verification for the true Mamba mixer (kind 'mamba'): depthwise causal conv
+ selective SSM + gate.

The load-bearing test is `test_step_matches_forward`: token-by-token recurrent
generation — carrying BOTH the conv window and the SSM state — must reproduce
the parallel forward exactly. A bug in the conv-buffer sliding, the single
place this mixer adds recurrent state beyond the SSM, shows up here and nowhere
else. `test_causality` guards the conv's left-pad-then-trim: a right-leaking
conv would invalidate teacher forcing.
"""

from __future__ import annotations

import pytest
import torch

from models.world_model.layers.mamba import MambaMixer
from models.world_model.layers.mixer import SequenceMixer, make_mixer


def _mixer(d_model: int = 8, d_state: int = 4, d_conv: int = 4) -> MambaMixer:
  torch.manual_seed(0)
  return MambaMixer(d_model=d_model, d_state=d_state, d_conv=d_conv)


def test_is_a_sequence_mixer_via_factory() -> None:
  m = make_mixer("mamba", 8, d_state=4, d_conv=4)
  assert isinstance(m, MambaMixer)
  assert isinstance(m, SequenceMixer)


def test_shape_preserved() -> None:
  m = _mixer(d_model=8)
  x = torch.randn(3, 12, 8)
  assert m(x).shape == (3, 12, 8)


def test_rejects_bad_shape() -> None:
  m = _mixer(d_model=8)
  with pytest.raises(ValueError):
    m(torch.randn(3, 12))
  with pytest.raises(ValueError):
    m(torch.randn(3, 12, 5))


def test_causality() -> None:
  """Perturbing token t must not change any output before t — the conv must be
  causal (left-padded), not centred."""
  m = _mixer(d_model=6, d_conv=4).eval()
  x = torch.randn(1, 8, 6)
  with torch.no_grad():
    y1 = m(x)
    x2 = x.clone()
    x2[0, 5] = torch.randn(6)
    y2 = m(x2)
  assert torch.allclose(y1[0, :5], y2[0, :5], atol=1e-6), "conv leaked the future"
  assert not torch.allclose(y1[0, 5:], y2[0, 5:])


def test_step_matches_forward() -> None:
  """Recurrent generation (conv window + SSM state) must equal parallel
  forward. This is the conv-buffer correctness oracle."""
  m = _mixer(d_model=8, d_state=4, d_conv=4).eval()
  x = torch.randn(2, 15, 8)
  with torch.no_grad():
    parallel = m(x)
    state = None
    outs = []
    for t in range(x.shape[1]):
      y_t, state = m.step(x[:, t], state)
      outs.append(y_t)
    recurrent = torch.stack(outs, dim=1)
  assert torch.allclose(parallel, recurrent, atol=1e-4), (
    f"step diverges from forward: {(parallel - recurrent).abs().max().item()}"
  )


@pytest.mark.slow
def test_step_matches_forward_across_shapes() -> None:
  for d_model, d_state, d_conv, length in [(4, 2, 2, 6), (16, 8, 4, 12), (12, 4, 3, 9)]:
    torch.manual_seed(1)
    m = MambaMixer(d_model=d_model, d_state=d_state, d_conv=d_conv).eval()
    x = torch.randn(2, length, d_model)
    with torch.no_grad():
      state = None
      outs = []
      for t in range(length):
        y_t, state = m.step(x[:, t], state)
        outs.append(y_t)
      rec = torch.stack(outs, dim=1)
      assert torch.allclose(m(x), rec, atol=1e-4), (
        f"mismatch d_model={d_model} d_conv={d_conv} L={length}"
      )


def test_gradients_flow() -> None:
  m = _mixer()
  x = torch.randn(2, 6, 8)
  m(x).sum().backward()
  for name, p in m.named_parameters():
    assert p.grad is not None, f"no gradient reached {name}"