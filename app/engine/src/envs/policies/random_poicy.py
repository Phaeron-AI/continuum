from __future__ import annotations

import numpy as np

from .base import Policy

class RandomPolicy(Policy):
  def __init__(self, num_actions: int, rng: np.random.Generator | None = None):
    super().__init__()
    if num_actions <= 0:
      raise ValueError("num_actions must be positive")
    self._num_actions = num_actions
    self._rng = rng if rng is not None else np.random.default_rng()

  def act(self, observation: np.ndarray)-> int:
    return int(self._rng.integers(0, self._num_actions))