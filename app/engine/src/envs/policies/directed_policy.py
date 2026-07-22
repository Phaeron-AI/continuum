from __future__ import annotations

import numpy as np

from .base import Policy

# Grid2D action ids (see envs/base.py Action): UP=0 DOWN=1 LEFT=2 RIGHT=3 INTERACT=4.
_OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}


class DirectedPolicy(Policy):
  """Movement-biased, open-loop policy: commit to one direction for a run of
  several steps, then turn — never immediately reversing, and skipping the
  INTERACT no-op by default."""

  def __init__(
    self,
    num_actions: int,
    rng: np.random.Generator | None = None,
    min_hold: int = 4,
    max_hold: int = 12,
    interact_prob: float = 0.0,
  ) -> None:
    if num_actions < 2:
      raise ValueError(f"num_actions must be >= 2, got {num_actions}")
    if not 1 <= min_hold <= max_hold:
      raise ValueError(f"require 1 <= min_hold <= max_hold, got {min_hold}, {max_hold}")
    if not 0.0 <= interact_prob < 1.0:
      raise ValueError(f"interact_prob must be in [0, 1), got {interact_prob}")

    self._rng = rng if rng is not None else np.random.default_rng()
    self._moves = list(range(min(4, num_actions)))
    self._interact = num_actions - 1 if num_actions >= 5 else None
    self._interact_prob = interact_prob
    self._min_hold = min_hold
    self._max_hold = max_hold

    self._direction = int(self._rng.choice(self._moves))
    self._hold = self._new_hold()

  def _new_hold(self) -> int:
    return int(self._rng.integers(self._min_hold, self._max_hold + 1))

  def _pick_direction(self) -> int:
    banned = _OPPOSITE.get(self._direction)
    choices = [m for m in self._moves if m != banned] or self._moves
    return int(self._rng.choice(choices))

  def act(self, observation: np.ndarray) -> int:
    if self._interact is not None and self._rng.random() < self._interact_prob:
      return self._interact
    if self._hold <= 0:
      self._direction = self._pick_direction()
      self._hold = self._new_hold()
    self._hold -= 1
    return self._direction