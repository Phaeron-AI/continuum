from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn as nn

class SequenceMixer(nn.Module, ABC):
  def __init__(self, d_model: int)-> None:
    super().__init__()
    self.d_model = d_model
  
  @abstractmethod
  def forward(self, x: Tensor)-> Tensor:...


class IdentityMixer(SequenceMixer):
  def forward(self, x: Tensor)-> Tensor:
    if x.ndim != 3:
      raise ValueError(f"expected (B, L, D), got shape {tuple(x.shape)}")
    if x.shape[-1] != self.d_model:
      raise ValueError(
        f"last dim {x.shape[-1]} != d_model {self.d_model}"
      )
    
    return x
  
def make_mixer(kind: str, d_model: int, **kwargs: object)-> SequenceMixer:
  if kind == "identity":
    return IdentityMixer(d_model=d_model)
  raise ValueError(
    f"unknown mixer kind: {kind!r}. Available: 'identity' "
    f"(ssm/mamba land in later sub-phases)"
  )