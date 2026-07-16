from __future__ import annotations

from dataclasses import dataclass

from models.tokenizer.spec import TokenSpec


@dataclass(frozen=True)
class TokenizerConfig:
  levels: tuple[int, ...] = (8, 8, 8, 5, 5)
  hidden: int = 64
  in_channels: int = 3
  input_size: int = 64
  grid_size: int = 8
  tokenizer_version: str = "v1"

  def __post_init__(self) -> None:
    if self.input_size % self.grid_size != 0:
      raise ValueError(
        f"input_size ({self.input_size}) must be divisible by "
        f"grid_size ({self.grid_size})"
      )
    if self.input_size // self.grid_size != 8:
      raise ValueError(
        f"encoder downsamples by 8x; input_size/grid_size must be 8, "
        f"got {self.input_size}/{self.grid_size}"
      )
  
  @property
  def latent_dim(self)-> int:
    return len(self.levels)
  
  def to_token_spec(self) -> TokenSpec:
    return TokenSpec(
      grid_height=self.grid_size,
      grid_width=self.grid_size,
      levels=self.levels,
      tokenizer_version=self.tokenizer_version,
      input_height=self.input_size,
      input_width=self.input_size,
      input_channels=self.in_channels,
    )