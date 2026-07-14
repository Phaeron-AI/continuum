from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

import numpy as np

class Action(IntEnum):
  UP = 0
  DOWN = 1
  LEFT = 2
  RIGHT = 3
  INTERACT = 4

ACTION_SPACE_VERSION = "v1"

@dataclass(frozen=True)
class ObservationSpec:
  height: int
  width: int
  channels: int
  dtype: str = "uint8"

  @property
  def shape(self)-> tuple[int, int, int]:
    return (self.height, self.width, self.channels)


@dataclass
class StepResult:
  observation: np.ndarray
  done: bool
  info: dict[str, Any]

class Env(ABC):
  observation_spec: ObservationSpec
  action_space_version: str = ACTION_SPACE_VERSION
  env_version: str

  @abstractmethod
  def reset(self, seed: Optional[int] = None)-> np.ndarray:...

  @abstractmethod
  def step(self, action: Action)-> StepResult:...