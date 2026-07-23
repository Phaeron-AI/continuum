"""Verification for FiLM action conditioning (action_conditioning='film').

`test_positional_action_ids_forward_fills` pins the per-position action-in-
effect derivation. `test_film_is_identity_at_init` guards the property that a
fresh FiLM model matches the inline model numerically (so it cannot regress the
near-random-perplexity anchor). `test_film_strengthens_action_effect` is the
point of the feature: once the modulation is non-trivial, the action changes
the prediction directly rather than only through the lone inline token.
"""

from __future__ import annotations

import pytest
import torch

from models.world_model.layers.conditioning import ActionFiLM, positional_action_ids
from models.world_model.model.config import WorldModelConfig
from models.world_model.model.world_model import WorldModel


def test_positional_action_ids_forward_fills() -> None:
  vocab = 10
  # frame0 (4 visual) | action 2 | frame1 (4) | action 0 | frame2 (4)
  ids = torch.tensor([[1, 2, 3, 4, vocab + 2, 5, 6, 7, 8, vocab + 0, 1, 1, 1, 1]])
  eff = positional_action_ids(ids, vocab_size=vocab)
  expected = torch.tensor([[0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0]])
  assert torch.equal(eff, expected)


def test_film_is_identity_at_init() -> None:
  film = ActionFiLM(num_actions=5, d_model=8)
  x = torch.randn(2, 6, 8)
  a = torch.randint(0, 5, (2, 6))
  assert torch.allclose(film(x, a), x), "FiLM must be identity at init"


def test_film_is_identity_within_model_at_init() -> None:
  """Adding FiLM changes nothing at init: zero-init modulation means the film
  model's output equals the same model with conditioning bypassed. (Comparing
  two *separately constructed* models would not work — building ActionFiLM
  draws from the RNG and desyncs the downstream block/head init.)"""
  torch.manual_seed(0)
  m = WorldModel(
    WorldModelConfig(
      vocab_size=32, num_actions=5, d_model=16, d_state=4, n_layers=2,
      action_conditioning="film",
    )
  ).eval()
  ids = torch.randint(0, 37, (2, 10))
  with torch.no_grad():
    with_film = m(ids)
    m.action_film = None  # bypass conditioning; weights otherwise identical
    without_film = m(ids)
  assert torch.allclose(with_film, without_film, atol=1e-6)


def test_film_strengthens_action_effect() -> None:
  """With non-trivial modulation, changing the in-effect action changes the
  logits even at the same positions — direct conditioning."""
  torch.manual_seed(0)
  cfg = WorldModelConfig(
    vocab_size=32, num_actions=5, d_model=16, d_state=4, n_layers=2,
    action_conditioning="film",
  )
  m = WorldModel(cfg).eval()
  torch.nn.init.normal_(m.action_film.gamma.weight, std=0.5)  # type: ignore
  torch.nn.init.normal_(m.action_film.beta.weight, std=0.5) # type: ignore

  vocab = 32
  base = torch.randint(0, vocab, (1, 9))
  a = base.clone()
  a[0, 0] = vocab + 1  # action 1 in effect from position 0
  b = base.clone()
  b[0, 0] = vocab + 3  # action 3 in effect from position 0
  with torch.no_grad():
    assert not torch.allclose(m(a), m(b)), "action did not affect the logits"


def test_film_step_is_guarded() -> None:
  """Recurrent step() under FiLM is a documented not-yet — it must fail loudly,
  not silently drop conditioning."""
  cfg = WorldModelConfig(
    vocab_size=32, num_actions=5, d_model=16, d_state=4, n_layers=1,
    action_conditioning="film",
  )
  m = WorldModel(cfg).eval()
  with pytest.raises(NotImplementedError):
    m.step(torch.randint(0, 32, (1,)), None)