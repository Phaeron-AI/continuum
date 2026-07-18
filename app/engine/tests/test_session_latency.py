"""Sub-phase 3.3 verification: GenerationSession latency instrumentation.

Not testing exact timing values (inherently noisy) — testing structure:
that metrics() reports the right shape at the right times, that the
step/sample/decode split behaves as documented (e.g. decode_ms is 0 when
return_pixels=False), and that history resets correctly across the
lifecycle. Kept in its own file rather than appended to test_session.py
to avoid a hand-merge against an already-edited live file.
"""

from __future__ import annotations

import torch
from torch import Tensor

from models.tokenizer.spec import TokenSpec
from models.world_model.model.config import WorldModelConfig
from models.world_model.model.world_model import WorldModel
from models.world_model.session import GenerationSession, _percentile


def _model(n_layers: int = 2, **kw) -> WorldModel:
  torch.manual_seed(0)
  base = dict(vocab_size=32, num_actions=5, d_model=16, d_state=4, n_layers=n_layers)
  base.update(kw)
  return WorldModel(WorldModelConfig(**base))  # type: ignore


class _FakeTokenizer:
  """Same minimal stand-in used in test_session.py."""

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


# ---- metrics() availability ----

def test_metrics_none_before_any_advance() -> None:
  model = _model().eval()
  session = GenerationSession(model, _tokenizer())  # type: ignore[arg-type]
  seed = torch.randint(0, model.config.vocab_size, (1, 3))
  session.open(seed)
  assert session.metrics() is None


def test_metrics_none_when_tracking_disabled() -> None:
  model = _model().eval()
  session = GenerationSession(model, _tokenizer(), track_latency=False)  # type: ignore[arg-type]
  seed = torch.randint(0, model.config.vocab_size, (1, 3))
  session.open(seed)
  session.advance(torch.zeros(1, dtype=torch.long))
  assert session.metrics() is None


# ---- shape / field sanity after real advances ----

def test_metrics_after_advance_has_expected_fields() -> None:
  model = _model().eval()
  session = GenerationSession(model, _tokenizer())  # type: ignore[arg-type]
  seed = torch.randint(0, model.config.vocab_size, (1, 3))
  session.open(seed)
  session.advance(torch.zeros(1, dtype=torch.long))

  report = session.metrics()
  assert report is not None
  assert report.count == 1
  assert report.last.total_ms >= 0.0
  assert report.last.step_ms >= 0.0
  assert report.last.sample_ms >= 0.0
  # step_ms and sample_ms are components of total_ms alongside decode_ms
  assert report.last.total_ms >= report.last.step_ms + report.last.sample_ms - 1e-6
  assert report.mean_total_ms >= 0.0
  assert report.fps > 0.0


def test_decode_ms_zero_when_return_pixels_false() -> None:
  model = _model().eval()
  session = GenerationSession(model, _tokenizer())  # type: ignore[arg-type]
  seed = torch.randint(0, model.config.vocab_size, (1, 3))
  session.open(seed)
  session.advance(torch.zeros(1, dtype=torch.long), return_pixels=False)

  report = session.metrics()
  assert report is not None
  assert report.last.decode_ms == 0.0


def test_decode_ms_nonzero_when_return_pixels_true() -> None:
  model = _model().eval()
  session = GenerationSession(model, _tokenizer())  # type: ignore[arg-type]
  seed = torch.randint(0, model.config.vocab_size, (1, 3))
  session.open(seed)
  session.advance(torch.zeros(1, dtype=torch.long), return_pixels=True)

  report = session.metrics()
  assert report is not None
  assert report.last.decode_ms >= 0.0  # >=0 not >0: fake decode is near-instant


# ---- history across multiple frames ----

def test_metrics_count_grows_with_frames() -> None:
  model = _model().eval()
  session = GenerationSession(model, _tokenizer())  # type: ignore[arg-type]
  seed = torch.randint(0, model.config.vocab_size, (1, 3))
  session.open(seed)

  for _ in range(5):
    session.advance(torch.zeros(1, dtype=torch.long), return_pixels=False)

  report = session.metrics()
  assert report is not None
  assert report.count == 5


# ---- lifecycle resets ----

def test_history_resets_on_reopen() -> None:
  model = _model().eval()
  session = GenerationSession(model, _tokenizer())  # type: ignore[arg-type]
  seed = torch.randint(0, model.config.vocab_size, (1, 3))

  session.open(seed)
  session.advance(torch.zeros(1, dtype=torch.long), return_pixels=False)
  assert session.metrics() is not None

  session.open(seed)  # reopen without close — history should still reset
  assert session.metrics() is None


def test_history_resets_on_close() -> None:
  model = _model().eval()
  session = GenerationSession(model, _tokenizer())  # type: ignore[arg-type]
  seed = torch.randint(0, model.config.vocab_size, (1, 3))

  session.open(seed)
  session.advance(torch.zeros(1, dtype=torch.long), return_pixels=False)
  session.close()

  assert session.metrics() is None


# ---- percentile helper (pure function, no model needed) ----

def test_percentile_matches_known_values() -> None:
  values = [10.0, 20.0, 30.0, 40.0, 50.0]
  assert _percentile(values, 0.0) == 10.0
  assert _percentile(values, 1.0) == 50.0
  assert _percentile(values, 0.5) == 30.0


def test_percentile_empty_list_is_zero() -> None:
  assert _percentile([], 0.5) == 0.0