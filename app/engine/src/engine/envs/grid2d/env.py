from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional

import numpy as np

from ..base import Action, Env, ObservationSpec, StepResult
from .config import Grid2DConfig

class Renderer(Protocol):
  def render(self, agent_pos: tuple[int, int], obstacles: list[np.ndarray])-> np.ndarray:...


@dataclass
class _Grid2DState:
  agent_pos: np.ndarray
  obstacles: list[np.ndarray]
  step_count: int = 0


class Grid2DEnv(Env):
  def __init__(self, config: Optional[Grid2DConfig] = None, renderer: Optional[Renderer] = None)-> None:
    self._config = config or Grid2DConfig()
    self.observation_spec = ObservationSpec(
      height=self._config.canvas_size,
      width=self._config.canvas_size,
      channels=3,
    )

    if renderer is None:
      from .renderer import Grid2DRenderer
      renderer = Grid2DRenderer(self._config)

    self._renderer = renderer
    self._state: Optional[_Grid2DState] = None
  
  @property
  def config(self)-> Grid2DConfig:
    return self._config
  
  def reset(self, seed: int | None = None) -> np.ndarray:
    cfg = self._config
    rng = np.random.default_rng(seed)
    agent_pos = rng.integers(
      cfg.agent_radius, cfg.canvas_size - cfg.agent_radius, size=2
    ).astype(np.float32)
    obstacles = [
      rng.integers(
        cfg.obstacle_radius, cfg.canvas_size - cfg.obstacle_radius, size=2
      ).astype(np.float32)
      for _ in range(cfg.num_obstacles)
    ]
    self._state = _Grid2DState(agent_pos=agent_pos, obstacles=obstacles)
    return self._render()
  
  def step(self, action: Action)-> StepResult:
    if self._state is None:
      raise RuntimeError("Env.step() called before reset()")

    cfg = self._config
    proposed = self._state.agent_pos.copy()
    if action == Action.UP:
      proposed[1] -= cfg.move_step
    elif action == Action.DOWN:
      proposed[1] += cfg.move_step
    elif action == Action.LEFT:
      proposed[0] -= cfg.move_step
    elif action == Action.RIGHT:
      proposed[0] += cfg.move_step

    proposed = np.clip(proposed, cfg.agent_radius, cfg.canvas_size - cfg.agent_radius)
    if not self._collides(proposed):
      self._state.agent_pos = proposed

    self._state.step_count += 1
    done = self._state.step_count >= cfg.episode_length
    info = {"step_count": self._state.step_count}
    return StepResult(observation=self._render(), done=done, info=info)
  
  def _collides(self, pos: np.ndarray) -> bool:
    assert self._state is not None
    threshold = self._config.agent_radius + self._config.obstacle_radius
    return any(
      np.linalg.norm(pos - obstacle) < threshold
      for obstacle in self._state.obstacles
    )

  def _render(self) -> np.ndarray:
    assert self._state is not None
    return self._renderer.render(self._state.agent_pos, self._state.obstacles)  # type: ignore