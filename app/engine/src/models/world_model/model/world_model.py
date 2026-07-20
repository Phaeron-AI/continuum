from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.world_model.layers.block import SSMBlock
from models.world_model.layers.conditioning import ActionFiLM, positional_action_ids
from models.world_model.layers.embedding import TokenEmbedding
from models.world_model.layers.mixer import make_mixer
from models.world_model.model.config import WorldModelConfig

GenerationState = list[Tensor | None]

class WorldModel(nn.Module):
  def __init__(self, config: WorldModelConfig)-> None:
    super().__init__()
    self.config = config

    self.embedding = TokenEmbedding(config.vocab_size, config.num_actions, config.d_model)
    self.action_film = (
      ActionFiLM(config.num_actions, config.d_model)
      if config.action_conditioning == "film"
      else None
    )
    self.blocks = nn.ModuleList([
      SSMBlock(
        make_mixer(
          config.mixer, config.d_model,
          d_state=config.d_state, d_conv=config.d_conv,
        ),
        config.d_model,
        config.ffn_mult
      ) for _ in range(config.n_layers)
    ])

    self.norm_out = nn.LayerNorm(config.d_model)
    self.head = nn.Linear(config.d_model, config.vocab_size)
  
  def forward(self, ids: Tensor)-> Tensor:
    x = self.embedding(ids)  # (B, L, D)
    if self.action_film is not None:
      # per-position action-in-effect, derived from the inline stream
      actions = positional_action_ids(ids, self.config.vocab_size)
      x = self.action_film(x, actions)
    for block in self.blocks:
      x = block(x)
    x = self.norm_out(x)
    return self.head(x)  # (B, L, vocab_size)
  
  def _corrupt_context(self, inputs: Tensor, prob: float) -> Tensor:
    """Replace a fraction `prob` of VISUAL input tokens with random visual
    ids, leaving action tokens (conditioning) and all targets untouched.

    This is the drift antidote: with pure teacher forcing the model only ever
    conditions on ground-truth context, so at generation time — fed its own
    imperfect tokens — it enters states it never trained on and the world melts.
    Training on corrupted context teaches it to recover from its own mistakes.
    """
    if prob <= 0.0:
      return inputs
    is_visual = ~self.embedding.is_action_token(inputs)
    corrupt = (torch.rand_like(inputs, dtype=torch.float) < prob) & is_visual
    random_ids = torch.randint_like(inputs, high=self.config.vocab_size)
    return torch.where(corrupt, random_ids, inputs)

  def loss(self, ids: Tensor, context_noise_prob: float = 0.0)-> Tensor:
    if ids.dim() != 2:
      raise ValueError(f"expected (B, L) ids, got {tuple(ids.shape)}")
    if ids.shape[1] < 2:
      raise ValueError("sequence must have at least 2 tokens to shift")
    
    inputs = ids[:, :-1]  # (B, L-1)
    targets = ids[:, 1:]  # (B, L-1) — targets stay clean

    inputs = self._corrupt_context(inputs, context_noise_prob)
    logits = self(inputs)  # (B, L-1, vocab_size)

    is_visual = ~self.embedding.is_action_token(targets)  # (B, L-1)

    flat_logits = logits.reshape(-1, self.config.vocab_size)
    flat_targets = targets.reshape(-1)
    flat_mask = is_visual.reshape(-1)

    if not bool(flat_mask.any()):
      raise ValueError("no visual targets in batch — nothing to train on")

    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
  
  @torch.no_grad()
  def predict_next(self, ids: Tensor)-> Tensor:
    logits = self(ids)[:, -1]  # (B, vocab_size) — last position
    return logits.argmax(dim=-1)
  
  def init_generation_state(
    self, batch: int, device: torch.device, dtype: torch.dtype
  ) -> GenerationState:
    """Fresh per-layer recurrent state for a generation session (Phase 3,
    stage 02). One entry per block — the state is a list because each
    block's mixer carries its own hidden state independently."""
    return [
      block.mixer.init_state(batch, device, dtype) for block in self.blocks # type: ignore
    ]

  def step(self, token: Tensor, state: GenerationState | None) -> tuple[Tensor, GenerationState]:
    if token.dim() != 1:
      raise ValueError(f"step expects (B,) token ids, got {tuple(token.shape)}")

    if self.action_film is not None:
      raise NotImplementedError(
        "FiLM conditioning is applied on the forward path (used by rollout); "
        "recurrent step() FiLM support lands in a later sub-phase"
      )
    x = self.embedding(token.unsqueeze(1)).squeeze(1)  # (B, D)
    if state is None:
      state = self.init_generation_state(token.shape[0], x.device, x.dtype)

    new_state = []
    for block, layer_state in zip(self.blocks, state, strict=True):
      x, s = block.step(x, layer_state) # type: ignore
      new_state.append(s)

    x = self.norm_out(x)
    logits = self.head(x)  # (B, vocab_size)
    return logits, new_state