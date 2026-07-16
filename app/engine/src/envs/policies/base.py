from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
  @abstractmethod
  def act(self, observation: np.ndarray)-> int: ...