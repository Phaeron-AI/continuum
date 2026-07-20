from __future__ import annotations

import os

from torch import Tensor, nn

# Range-checking token ids means reading their min/max, which forces a
# device-to-host sync on every forward — fine for debugging, ruinous on a
# hot GPU training loop. Off by default; opt in with CONTINUUM_CHECK_TOKEN_IDS=1.
# nn.Embedding still faults loudly on an out-of-range id regardless.
_CHECK_IDS = os.environ.get("CONTINUUM_CHECK_TOKEN_IDS", "0") == "1"


class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size: int, num_actions: int, d_model: int) -> None:
    super().__init__()
    self.vocab_size = vocab_size
    self.num_actions = num_actions
    self.d_model = d_model
    # One table spanning both kinds — the stage-06 shared space.
    self.embed = nn.Embedding(vocab_size + num_actions, d_model)

  @property
  def total_vocab(self) -> int:
    return self.vocab_size + self.num_actions

  def forward(self, ids: Tensor) -> Tensor:
    """ids: (B, L) long -> (B, L, d_model) float."""
    if ids.dim() != 2:
      raise ValueError(f"expected (B, L) token ids, got {tuple(ids.shape)}")
    if _CHECK_IDS and (int(ids.max()) >= self.total_vocab or int(ids.min()) < 0):
      raise ValueError(
        f"token id out of range [0, {self.total_vocab}): "
        f"got min={int(ids.min())}, max={int(ids.max())}"
      )
    return self.embed(ids)

  def is_action_token(self, ids: Tensor) -> Tensor:
    return ids >= self.vocab_size