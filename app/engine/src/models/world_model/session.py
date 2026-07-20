from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor

from models.tokenizer.frozen import FrozenTokenizer
from models.world_model.inference.rollout import _pick
from models.world_model.model.world_model import GenerationState, WorldModel

_HISTORY_MAXLEN = 512  # bounded so long-running sessions don't leak memory


@dataclass(frozen=True)
class FrameLatency:
  step_ms: float
  sample_ms: float
  decode_ms: float
  total_ms: float


@dataclass(frozen=True)
class LatencyReport:
  last: FrameLatency
  count: int
  mean_total_ms: float
  p50_total_ms: float
  p95_total_ms: float
  fps: float  # 1000 / mean_total_ms; 0.0 if no frames yet


def _percentile(sorted_values: list[float], q: float) -> float:
  if not sorted_values:
    return 0.0
  if len(sorted_values) == 1:
    return sorted_values[0]
  idx = q * (len(sorted_values) - 1)
  lo = int(idx)
  hi = min(lo + 1, len(sorted_values) - 1)
  frac = idx - lo
  return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


class GenerationSession:
  """Interactive session over a WorldModel — the object a live client drives.

  Lifecycle: open(seed) warms state once by stepping every seed token,
  advance(action) generates one frame per call, close() discards state.
  metrics() reports per-frame latency (step/sample/decode) since open().
  """

  def __init__(
    self,
    model: WorldModel,
    tokenizer: FrozenTokenizer,
    track_latency: bool = True,
  )-> None:
    self._model = model
    self._tokenizer = tokenizer
    self._vocab = model.config.vocab_size
    self._tokens_per_frame = tokenizer.token_spec.tokens_per_frame
    self._grid_height = tokenizer.token_spec.grid_height
    self._grid_width = tokenizer.token_spec.grid_width
    self._device = next(model.parameters()).device

    self._state: GenerationState | None = None
    self._is_open = False
    self._batch = 0

    self._track_latency = track_latency
    self._history: deque[FrameLatency] = deque(maxlen=_HISTORY_MAXLEN)

  @property
  def is_open(self)-> bool:
    return self._is_open

  def _sync(self)-> None:
    if self._device.type == "cuda":
      torch.cuda.synchronize(self._device)

  def _now(self)-> float:
    return time.perf_counter()

  @torch.no_grad()
  def open(self, seed_tokens: Tensor)-> None:
    """Warm the recurrent state by stepping every seed token, in order.

    seed_tokens: (B, L) token ids — same shape rollout() expects.
    """
    if seed_tokens.dim() != 2:
      raise ValueError(f"expected (B, L) seed_tokens, got {tuple(seed_tokens.shape)}")

    self._model.eval()
    state: GenerationState | None = None
    for t in range(seed_tokens.shape[1]):
      _logits, state = self._model.step(seed_tokens[:, t], state)

    self._state = state
    self._batch = seed_tokens.shape[0]
    self._is_open = True
    self._history.clear()

  @torch.no_grad()
  def advance(
    self,
    action: Tensor,
    return_pixels: bool = True,
    temperature: float = 0.0,
  )-> Tensor:
    if not self._is_open:
      raise RuntimeError("advance() called before open() — session is not started")
    if action.dim() != 1:
      raise ValueError(f"expected (B,) action, got {tuple(action.shape)}")
    if action.shape[0] != self._batch:
      raise ValueError(
        f"action batch {action.shape[0]} does not match session batch {self._batch}"
      )

    track = self._track_latency
    step_ms = 0.0
    sample_ms = 0.0
    decode_ms = 0.0

    if track:
      self._sync()
      t_start = self._now()

    # The action that produces this frame is stepped in first — mirrors
    # rollout()'s "append action token before generating" ordering.
    action_tok = self._vocab + action

    if track:
      self._sync()
      t0 = self._now()
    logits, self._state = self._model.step(action_tok, self._state)
    if track:
      self._sync()
      step_ms += (self._now() - t0) * 1000.0

    frame_tokens: list[Tensor] = []
    for _ in range(self._tokens_per_frame):
      if track:
        self._sync()
        t0 = self._now()
      nxt = _pick(logits, temperature)  # (B,)
      if track:
        self._sync()
        sample_ms += (self._now() - t0) * 1000.0

      frame_tokens.append(nxt)

      # Feed the model's own prediction back in — same compounding-error
      # step as rollout(), just via step() instead of full reprocessing.
      if track:
        self._sync()
        t0 = self._now()
      logits, self._state = self._model.step(nxt, self._state)
      if track:
        self._sync()
        step_ms += (self._now() - t0) * 1000.0

    grid = torch.stack(frame_tokens, dim=1).reshape(
      self._batch, self._grid_height, self._grid_width
    )  # (B, h, w)

    if return_pixels:
      if track:
        self._sync()
        t0 = self._now()
      result: Tensor = self._tokenizer.decode(grid)  # (B, C, H, W)
      if track:
        self._sync()
        decode_ms = (self._now() - t0) * 1000.0
    else:
      result = grid

    if track:
      self._sync()
      total_ms = (self._now() - t_start) * 1000.0
      self._history.append(
        FrameLatency(step_ms=step_ms, sample_ms=sample_ms, decode_ms=decode_ms, total_ms=total_ms)
      )

    return result

  def metrics(self)-> LatencyReport | None:
    """Latency since open(), over the retained history. None if no frame
    has been generated yet, or if track_latency=False."""
    if not self._history:
      return None

    totals = sorted(f.total_ms for f in self._history)
    mean_total = sum(totals) / len(totals)
    return LatencyReport(
      last=self._history[-1],
      count=len(self._history),
      mean_total_ms=mean_total,
      p50_total_ms=_percentile(totals, 0.50),
      p95_total_ms=_percentile(totals, 0.95),
      fps=1000.0 / mean_total if mean_total > 0 else 0.0,
    )

  def close(self)-> None:
    """Discard recurrent state and latency history."""
    self._state = None
    self._is_open = False
    self._batch = 0
    self._history.clear()