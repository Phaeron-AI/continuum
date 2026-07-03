"""Policy tests: decoupled from any concrete Action enum."""

from __future__ import annotations

import numpy as np
import pytest

from ..src.envs.policies.random_policy import RandomPolicy


def test_random_policy_stays_in_range() -> None:
  policy = RandomPolicy(num_actions=5, rng=np.random.default_rng(0))
  obs = np.zeros((8, 8, 3), dtype=np.uint8)
  actions = [policy.act(obs) for _ in range(200)]
  assert all(0 <= a < 5 for a in actions)
  # sanity: with 200 draws over 5 actions, every action should appear
  assert set(actions) == {0, 1, 2, 3, 4}


def test_random_policy_seeded_reproducibility() -> None:
  obs = np.zeros((8, 8, 3), dtype=np.uint8)
  p1 = RandomPolicy(5, np.random.default_rng(7))
  p2 = RandomPolicy(5, np.random.default_rng(7))
  seq1 = [p1.act(obs) for _ in range(50)]
  seq2 = [p2.act(obs) for _ in range(50)]
  assert seq1 == seq2


def test_random_policy_rejects_bad_action_count() -> None:
  with pytest.raises(ValueError):
    RandomPolicy(num_actions=0)