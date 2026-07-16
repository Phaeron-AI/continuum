from __future__ import annotations

import torch
from torch import Tensor, nn

from models.tokenizer.config import TokenizerConfig
from models.tokenizer.modules.decoder import Decoder
from models.tokenizer.modules.encoder import Encoder
from models.tokenizer.modules.quantizer import FSQ
from models.tokenizer.spec import TokenSpec


class Tokenizer(nn.Module):
  def __init__(self, config: TokenizerConfig | None = None) -> None:
    super().__init__()
    self.config = config or TokenizerConfig()
    self.encoder = Encoder(
      in_channels=self.config.in_channels,
      hidden=self.config.hidden,
      latent_dim=self.config.latent_dim,
    )
    self.quantizer = FSQ(levels=self.config.levels)
    self.decoder = Decoder(
      out_channels=self.config.in_channels,
      hidden=self.config.hidden,
      latent_dim=self.config.latent_dim,
    )

  @property
  def token_spec(self) -> TokenSpec:
    return self.config.to_token_spec()

  def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
    z = self.encoder(x)
    quantized, indices = self.quantizer(z)
    recon = self.decoder(quantized)
    return recon, indices

  @torch.no_grad()
  def encode_indices(self, x: Tensor) -> Tensor:
    z = self.encoder(x)
    _, indices = self.quantizer(z)
    return indices

  @torch.no_grad()
  def decode_indices(self, indices: Tensor) -> Tensor:
    codes = self.quantizer.indices_to_codes(indices)  # (B, h, w, D)
    codes = codes.movedim(-1, 1)  # (B, D, h, w)
    return self.decoder(codes)