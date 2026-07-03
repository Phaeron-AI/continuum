from __future__ import annotations

import os
import numpy as np

from .config import Grid2DConfig

BACKGROUND_COLOR = (20, 20, 20)
AGENT_COLOR = (66, 135, 245)
OBSTACLE_COLOR = (200, 60, 60)

class Grid2DRenderer:
  def __init__(self, config: Grid2DConfig, headless: bool = True)-> None:
    if headless:
      os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    import pygame

    if not pygame.get_init():
      pygame.init()
    
    self._pygame = pygame
    self._config = config
    self._surface = pygame.Surface((self._config.canvas_size, self._config.canvas_size))
  
  def render(self, agent_pos: tuple[int, int], obstacles: list[np.ndarray])-> np.ndarray:
    pygame = self._pygame
    self._surface.fill(BACKGROUND_COLOR)

    for obstacle in obstacles:
      pygame.draw.circle(
        self._surface,
        OBSTACLE_COLOR,
        obstacle.astype(int), # type: ignore
        self._config.obstacle_radius,
      )
    pygame.draw.circle(
      self._surface,
      AGENT_COLOR,
      agent_pos.astype(int),  # type: ignore
      self._config.agent_radius,
    )

    frame = pygame.surfarray.array3d(self._surface)

    return np.transpose(frame, (1, 0, 2)).astype(np.uint8)