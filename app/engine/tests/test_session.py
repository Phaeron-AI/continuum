"""Sub-phase 3.2 verification: GenerationSession.

THE load-bearing test is `test_session_matches_rollout_greedy`: generating
frame-by-frame through the session's O(1) recurrent path must produce the
identical tokens as Phase 2's O(L^2) rollout() on the same seed and actions.
This lifts 3.1's whole-model step equivalence up to the interactive API —
a bug in how advance() sequences action/prediction tokens would otherwise
be an invisible generation error, not a crash.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from models.tokenizer.spec import TokenSpec
from models.world_model.inference.rollout import rollout
from models.world_model.model.config import WorldModelConfig
from models.world_model.model.world_model import WorldModel
from models.world_model.session import GenerationSession


def _model(n_layers: int = 3, **kw) -> WorldModel:
  torch.manual_seed(0)
  base = dict(vocab_size=32, num_actions=5, d_model=16, d_state=4, n_layers=n_layers)
  base.update(kw)
  return WorldModel(WorldModelConfig(**base))  # type: ignore


class _FakeTokenizer:
  """Minimal FrozenTokenizer stand-in. GenerationSession only touches
  token_spec and decode(); decode()'s content is never exercised on the
  equivalence path (that path uses return_pixels=False and is checked
  separately from decode's own shape contract)."""

  def __init__(self, spec: TokenSpec) -> None:
    self._spec = spec

  @property
  def token_spec(self) -> TokenSpec:
    return self._spec

  def decode(self, indices: Tensor) -> Tensor:
    batch = indices.shape[0]
    spec = self._spec
    return torch.zeros(batch, spec.input_channels, spec.input_height, spec.input_width)


def _tokenizer(grid_height: int = 2, grid_width: int = 2) -> _FakeTokenizer:
  spec = TokenSpec(grid_height=grid_height, grid_width=grid_width, levels=(2, 2, 2, 2, 2))
  return _FakeTokenizer(spec)


# ---- THE session-vs-rollout equivalence test ----

def test_session_matches_rollout_greedy() -> None:
  """Session-vs-rollout token equivalence (greedy) — the load-bearing proof
  that the fast stateful path computes exactly what rollout() computes."""
  model = _model(n_layers=3).eval()
  tokenizer = _tokenizer(grid_height=2, grid_width=2)
  batch, seed_len, num_frames = 2, 5, 3

  torch.manual_seed(42)
  seed_tokens = torch.randint(0, model.config.vocab_size, (batch, seed_len))
  actions = torch.randint(0, model.config.num_actions, (batch, num_frames))

  with torch.no_grad():
    expected = rollout(
      model,
      seed_tokens,
      actions,
      num_frames,
      tokenizer.token_spec.tokens_per_frame,
      temperature=0.0,
    )  # (B, num_frames, tokens_per_frame)

  session = GenerationSession(model, tokenizer)  # type: ignore[arg-type]
  session.open(seed_tokens)
  frames = []
  for f in range(num_frames):
    grid = session.advance(actions[:, f], return_pixels=False, temperature=0.0)
    frames.append(grid.reshape(batch, -1))
  session.close()

  actual = torch.stack(frames, dim=1)
  assert torch.equal(actual, expected), (
    "session output diverges from rollout() — the O(1) path is no longer "
    "provably equivalent to the O(L^2) reference"
  )


# ---- lifecycle ----

def test_open_sets_is_open() -> None:
  model = _model().eval()
  session = GenerationSession(model, _tokenizer())  # type: ignore[arg-type]
  assert not session.is_open
  seed = torch.randint(0, model.config.vocab_size, (1, 4))
  session.open(seed)
  assert session.is_open


def test_close_discards_state_and_requires_reopen() -> None:
  model = _model().eval()
  session = GenerationSession(model, _tokenizer())  # type: ignore[arg-type]
  seed = torch.randint(0, model.config.vocab_size, (1, 4))
  session.open(seed)
  session.advance(torch.zeros(1, dtype=torch.long))
  session.close()

  assert not session.is_open
  with pytest.raises(RuntimeError):
    session.advance(torch.zeros(1, dtype=torch.long))


def test_advance_before_open_raises() -> None:
  model = _model().eval()
  session = GenerationSession(model, _tokenizer())  # type: ignore[arg-type]
  with pytest.raises(RuntimeError):
    session.advance(torch.zeros(1, dtype=torch.long))


def test_advance_rejects_batch_mismatch() -> None:
  model = _model().eval()
  session = GenerationSession(model, _tokenizer())  # type: ignore[arg-type]
  seed = torch.randint(0, model.config.vocab_size, (2, 4))
  session.open(seed)
  with pytest.raises(ValueError):
    session.advance(torch.zeros(3, dtype=torch.long))  # wrong batch


# ---- return_pixels flag ----

def test_return_pixels_true_shape() -> None:
  model = _model().eval()
  tokenizer = _tokenizer(grid_height=2, grid_width=2)
  session = GenerationSession(model, tokenizer)  # type: ignore[arg-type]
  seed = torch.randint(0, model.config.vocab_size, (2, 3))
  session.open(seed)

  pixels = session.advance(torch.zeros(2, dtype=torch.long), return_pixels=True)

  spec = tokenizer.token_spec
  assert pixels.shape == (2, spec.input_channels, spec.input_height, spec.input_width)


def test_return_pixels_false_shape() -> None:
  model = _model().eval()
  tokenizer = _tokenizer(grid_height=2, grid_width=3)
  session = GenerationSession(model, tokenizer)  # type: ignore[arg-type]
  seed = torch.randint(0, model.config.vocab_size, (2, 3))
  session.open(seed)

  grid = session.advance(torch.zeros(2, dtype=torch.long), return_pixels=False)

  assert grid.shape == (2, 2, 3)  # (B, grid_height, grid_width)


# ---- temperature=0 determinism ----

def test_temperature_zero_is_deterministic() -> None:
  """Same seed/actions through two independently-constructed sessions must
  produce identical output at temperature=0 — greedy, no sampling noise."""

  def _run() -> Tensor:
    model = _model(n_layers=2).eval()
    tokenizer = _tokenizer(grid_height=2, grid_width=2)
    session = GenerationSession(model, tokenizer)  # type: ignore[arg-type]
    seed = torch.zeros(1, 3, dtype=torch.long)
    actions = torch.zeros(1, 2, dtype=torch.long)

    session.open(seed)
    grids = [
      session.advance(actions[:, f], return_pixels=False, temperature=0.0)
      for f in range(2)
    ]
    session.close()
    return torch.stack(grids, dim=1)

  assert torch.equal(_run(), _run())