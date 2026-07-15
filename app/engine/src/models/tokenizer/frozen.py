from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from models.tokenizer.checkpoint import build_tokenizer_from_checkpoint
from models.tokenizer.spec import TokenSpec
from models.tokenizer.tokenizer import Tokenizer


class FrozenTokenizer:
  def __init__(self, tokenzier: Tokenizer)-> None:
    self._tokenizer = tokenzier.eval()
    # Freeze Parameters (Non-Trainable Params.)
    for p in self._tokenizer.parameters():
      p.requires_grad_(False)
  
  @classmethod
  def from_checkpoint(cls, path: Path | str, device: str | torch.device = "cpu")-> FrozenTokenizer:
    model, _ckpt = build_tokenizer_from_checkpoint(path, map_location=device)
    model.to(device)
    return cls(model)
  
  @property
  def token_spec(self)-> TokenSpec:
    return self._tokenizer.token_spec
  
  @torch.no_grad()
  def encode(self, frames: Tensor)-> Tensor:
    return self._tokenizer.encode_indices(frames)
  
  @torch.no_grad()
  def decode(self, indices: Tensor)-> Tensor:
    return self._tokenizer.decode_indices(indices)