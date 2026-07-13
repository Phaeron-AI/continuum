from __future__ import annotations

from dataclasses import dataclass

from engine.models.tokenizer.spec import TokenSpec

@dataclass(frozen=True)
class WorldModelConfig:
  vocab_size: int
  num_actions: int = 5
  d_model: int = 256
  d_state: int = 16
  n_layers: int = 4
  ffn_mult: int = 4
  mixer: str = "ssm"
  tokenizer_version: str = "v1"

  def __post_init__(self)-> None:
    if self.vocab_size <= 0:
      raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
    if self.num_actions <= 0:
      raise ValueError(f"num_actions must be positive, got {self.num_actions}")
    if self.n_layers <= 0:
      raise ValueError(f"n_layers must be positive, got {self.n_layers}")
    
  @property
  def total_vocab(self)-> int:
    return self.vocab_size + self.num_actions
  
  @classmethod
  def from_token_sec(
    cls,
    spec: TokenSpec,
    num_actions: int = 5,
    **kwargs: object
  )-> WorldModelConfig:
    return cls(
      vocab_size=spec.vocab_size,
      num_actions=num_actions,
      tokenizer_version=spec.tokenizer_version,
      **kwargs,  # type: ignore[arg-type]
    )