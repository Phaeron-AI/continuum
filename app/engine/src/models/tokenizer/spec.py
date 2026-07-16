from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TokenSpec:
  grid_height: int
  grid_width: int
  levels: tuple[int, ...]
  tokenizer_version: str = "v1"
  input_height: int = 64
  input_width: int = 64
  input_channels: int = 3
  latent_dim: int = field(default=0)

  def __post_init__(self) -> None:
    if self.grid_height <= 0 or self.grid_width <= 0:
      raise ValueError(
        f"grid dims must be positive, got "
        f"({self.grid_height}, {self.grid_width})"
      )
    if len(self.levels) == 0:
      raise ValueError("levels must be non-empty")
    if any(level < 2 for level in self.levels):
      raise ValueError(f"every FSQ level must be >= 2, got {self.levels}")
    derived_dim = len(self.levels)
    if self.latent_dim == 0:
      object.__setattr__(self, "latent_dim", derived_dim)
    elif self.latent_dim != derived_dim:
      raise ValueError(
        f"latent_dim ({self.latent_dim}) must equal len(levels) "
        f"({derived_dim})"
      )

  @property
  def vocab_size(self) -> int:
    """Total number of distinct codes = product of FSQ levels."""
    return math.prod(self.levels)

  @property
  def tokens_per_frame(self) -> int:
    return self.grid_height * self.grid_width

  @property
  def bits_per_token(self) -> float:
    return math.log2(self.vocab_size)

  @property
  def bits_per_frame(self) -> float:
    return self.tokens_per_frame * self.bits_per_token

  def to_dict(self) -> dict:
    return {
      "grid_height": self.grid_height,
      "grid_width": self.grid_width,
      "levels": list(self.levels),
      "tokenizer_version": self.tokenizer_version,
      "input_height": self.input_height,
      "input_width": self.input_width,
      "input_channels": self.input_channels,
      "latent_dim": self.latent_dim,
      "vocab_size": self.vocab_size,
    }

  @classmethod
  def from_dict(cls, data: dict) -> TokenSpec:
    return cls(
      grid_height=data["grid_height"],
      grid_width=data["grid_width"],
      levels=tuple(data["levels"]),
      tokenizer_version=data.get("tokenizer_version", "v1"),
      input_height=data.get("input_height", 64),
      input_width=data.get("input_width", 64),
      input_channels=data.get("input_channels", 3),
      latent_dim=data.get("latent_dim", 0),
    )