"""Sub-phase 0.1 verification: determinism and observation-spec compliance,
plus verification that the modularity seams (config object, injected
renderer) actually hold."""

from __future__ import annotations

import numpy as np
import pytest

from engine.envs.base import Action
from engine.envs.grid2d.config import Grid2DConfig
from engine.envs.grid2d.env import Grid2DEnv


class MockRenderer:
  """Trivial renderer for pure-logic tests — no pygame involved. Returns a
  zero frame of the right shape; also records calls so tests can assert
  the env rendered when expected."""

  def __init__(self, canvas_size: int) -> None:
    self._canvas_size = canvas_size
    self.render_calls = 0

  def render(self, agent_pos: np.ndarray, obstacles: list[np.ndarray]) -> np.ndarray:
    self.render_calls += 1
    return np.zeros((self._canvas_size, self._canvas_size, 3), dtype=np.uint8)


def _run_episode(seed: int, steps: int = 20) -> list[np.ndarray]:
  env = Grid2DEnv(config=Grid2DConfig(episode_length=steps))
  rng = np.random.default_rng(seed)
  frames = [env.reset(seed=seed)]
  for _ in range(steps):
    action = Action(rng.integers(0, len(Action)))
    result = env.step(action)
    frames.append(result.observation)
    if result.done:
      break
  return frames


def test_same_seed_produces_identical_episode() -> None:
  ep_a = _run_episode(seed=42)
  ep_b = _run_episode(seed=42)
  assert len(ep_a) == len(ep_b)
  for frame_a, frame_b in zip(ep_a, ep_b):
    np.testing.assert_array_equal(frame_a, frame_b)


def test_different_seed_produces_different_episode() -> None:
  ep_a = _run_episode(seed=1)
  ep_b = _run_episode(seed=2)
  assert any(not np.array_equal(a, b) for a, b in zip(ep_a, ep_b))


def test_observation_matches_declared_spec() -> None:
  env = Grid2DEnv()
  obs = env.reset(seed=0)
  spec = env.observation_spec
  assert obs.shape == spec.shape
  assert obs.dtype == np.uint8


def test_step_before_reset_raises() -> None:
  env = Grid2DEnv()
  with pytest.raises(RuntimeError):
    env.step(Action.UP)


def test_config_drives_observation_spec() -> None:
  env = Grid2DEnv(config=Grid2DConfig(canvas_size=64))
  assert env.observation_spec.shape == (64, 64, 3)
  obs = env.reset(seed=0)
  assert obs.shape == (64, 64, 3)


def test_mock_renderer_injection_runs_without_pygame_rendering() -> None:
  """The injection seam: movement/collision logic is fully testable with a
  renderer stub."""
  config = Grid2DConfig(canvas_size=84, episode_length=5)
  mock = MockRenderer(canvas_size=config.canvas_size)
  env = Grid2DEnv(config=config, renderer=mock) # type: ignore
  env.reset(seed=0)
  for _ in range(5):
    result = env.step(Action.RIGHT)
  assert result.done
  # reset renders once, each of the 5 steps renders once
  assert mock.render_calls == 6


def test_invalid_config_rejected() -> None:
  with pytest.raises(ValueError):
    Grid2DConfig(canvas_size=4, agent_radius=3)
  with pytest.raises(ValueError):
    Grid2DConfig(episode_length=0)