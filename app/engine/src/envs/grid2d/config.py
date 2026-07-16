from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Grid2DConfig:
  canvas_size: int = 84
  episode_length: int = 200
  num_obstacles: int = 5
  agent_radius: int = 3
  obstacle_radius: int = 3
  move_step: int = 3

  def __post_init__(self)-> None:
    if self.canvas_size <= 2 * self.agent_radius:
      raise ValueError(
        f"canvas_size ({self.canvas_size}) too small for "
        f"agent_radius ({self.agent_radius})"
      )
    if self.episode_length <= 0:
      raise ValueError(f"episode_length must be positive, got {self.episode_length}")
    if self.num_obstacles < 0:
      raise ValueError(f"num_obstacles must be >= 0, got {self.num_obstacles}")
  
  @classmethod
  def from_dict(cls, data: dict)-> Grid2DConfig:
    return cls(**data)