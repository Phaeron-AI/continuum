from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from models.world_model.layers.mixer import SequenceMixer


class SSMBlock(nn.Module):
  def __init__(self, mixer: SequenceMixer, d_model: int, ffn_mult: int = 4)-> None:
    super().__init__()
    self.norm1 = nn.LayerNorm(d_model)
    self.mixer = mixer
    self.norm2 = nn.LayerNorm(d_model)
    self.ffn = nn.Sequential(
      nn.Linear(d_model, d_model * ffn_mult),
      nn.SiLU(),
      nn.Linear(d_model * ffn_mult, d_model),
    )
  
  def forward(self, x: Tensor)-> Tensor:
    x = x + self.mixer(self.norm1(x))
    return x + self.ffn(self.norm2(x))\
    