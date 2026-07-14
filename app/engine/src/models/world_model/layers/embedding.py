from __future__ import annotations

from torch import Tensor, nn


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
    if int(ids.max()) >= self.total_vocab or int(ids.min()) < 0:
      raise ValueError(
        f"token id out of range [0, {self.total_vocab}): "
        f"got min={int(ids.min())}, max={int(ids.max())}"
      )
    return self.embed(ids)

  def is_action_token(self, ids: Tensor) -> Tensor:
    return ids >= self.vocab_size