"""Verification for the movement-biased DirectedPolicy.

The load-bearing tests are `test_commits_to_runs` (the agent moves in sustained
directional runs rather than jittering — the whole point, since jitter is what
makes frames near-duplicates) and `test_never_immediately_reverses` (a turn
never flips to the exact opposite, so runs accumulate net displacement).
"""

from __future__ import annotations

import numpy as np

from envs.policies.directed_policy import DirectedPolicy
from envs.policies.registry import make_policy

_OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}
_OBS = np.zeros((4, 4, 3), dtype=np.uint8)


def _roll(policy, n: int = 600) -> list[int]:
  return [policy.act(_OBS) for _ in range(n)]


def test_registered() -> None:
  p = make_policy("directed", 5, np.random.default_rng(0))
  assert isinstance(p, DirectedPolicy)


def test_actions_in_range() -> None:
  acts = _roll(DirectedPolicy(5, np.random.default_rng(0)))
  assert all(0 <= a < 5 for a in acts)


def test_skips_interact_by_default() -> None:
  # interact_prob defaults to 0.0 -> the no-op (id 4) never appears.
  assert 4 not in _roll(DirectedPolicy(5, np.random.default_rng(0)))


def test_never_immediately_reverses() -> None:
  acts = _roll(DirectedPolicy(5, np.random.default_rng(1)))
  for a, b in zip(acts[:-1], acts[1:], strict=True):
    assert b != _OPPOSITE.get(a), f"immediate reversal {a} -> {b}"


def test_commits_to_runs() -> None:
  # Far fewer direction changes than steps => sustained runs, not jitter.
  acts = _roll(DirectedPolicy(5, np.random.default_rng(2), min_hold=4, max_hold=12))
  changes = sum(1 for a, b in zip(acts[:-1], acts[1:], strict=True) if a != b)
  assert changes < len(acts) * 0.34, f"too jittery: {changes} changes / {len(acts)}"


def test_deterministic_with_seed() -> None:
  a = _roll(DirectedPolicy(5, np.random.default_rng(7)))
  b = _roll(DirectedPolicy(5, np.random.default_rng(7)))
  assert a == b


def test_interact_appears_when_enabled() -> None:
  assert 4 in _roll(DirectedPolicy(5, np.random.default_rng(0), interact_prob=0.2))


def test_rejects_bad_args() -> None:
  import pytest

  with pytest.raises(ValueError):
    DirectedPolicy(1, np.random.default_rng(0))
  with pytest.raises(ValueError):
    DirectedPolicy(5, np.random.default_rng(0), min_hold=0)
  with pytest.raises(ValueError):
    DirectedPolicy(5, np.random.default_rng(0), min_hold=10, max_hold=4)
  with pytest.raises(ValueError):
    DirectedPolicy(5, np.random.default_rng(0), interact_prob=1.0)