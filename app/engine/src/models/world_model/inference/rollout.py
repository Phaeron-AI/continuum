from __future__ import annotations

import torch
from torch import Tensor

from models.world_model.model.world_model import WorldModel


def _pick(
  logits: Tensor,
  temperature: float,
  top_k: int = 0,
  top_p: float = 0.0,
)-> Tensor:
  if temperature <= 0.0:
    return logits.argmax(dim=-1)

  logits = logits / temperature

  if top_k > 0:
    k = min(top_k, logits.shape[-1])
    kth = logits.topk(k, dim=-1).values[..., -1, None]
    logits = logits.masked_fill(logits < kth, float("-inf"))

  if top_p > 0.0:
    ordered, order = torch.sort(logits, dim=-1, descending=True)
    cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
    remove = cumulative - torch.softmax(ordered, dim=-1) >= top_p
    ordered = ordered.masked_fill(remove, float("-inf"))
    logits = torch.empty_like(logits).scatter_(-1, order, ordered)

  probs = torch.softmax(logits, dim=-1)
  return torch.multinomial(probs, num_samples=1).squeeze(-1)

@torch.no_grad()
def rollout(
  model: WorldModel,
  seed_tokens: Tensor,
  actions: Tensor,
  num_frames: int,
  tokens_per_frame: int,
  temperature: float = 0.0,
  top_k: int = 0,
  top_p: float = 0.0,
)-> Tensor:
  model.eval()
  vocab = model.config.vocab_size

  if actions.shape[1] < num_frames:
    raise ValueError(f"need {num_frames} actions, got {actions.shape[1]}")

  context = seed_tokens.clone()
  generated: list[Tensor] = []

  for f in range(num_frames):
    # The action that produces this frame is appended BEFORE generating it
    # — that is what conditions the prediction (stage 06).
    action_tok = (vocab + actions[:, f]).unsqueeze(1)  # (B, 1)
    context = torch.cat([context, action_tok], dim=1)

    frame_tokens: list[Tensor] = []
    for _ in range(tokens_per_frame):
      logits = model(context)[:, -1]  # (B, vocab) — next-token logits
      nxt = _pick(logits, temperature, top_k=top_k, top_p=top_p)  # (B,)
      frame_tokens.append(nxt)
      # Feed the model's OWN prediction back in. This is the step where
      # error begins to compound (stage 07).
      context = torch.cat([context, nxt.unsqueeze(1)], dim=1)

    generated.append(torch.stack(frame_tokens, dim=1))  # (B, tokens_per_frame)

  return torch.stack(generated, dim=1)  # (B, num_frames, tokens_per_frame)

@torch.no_grad()
def rollout_to_frames(
  model: WorldModel,
  tokenizer, # FrozenTokenizer,
  seed_tokens: Tensor,
  actions: Tensor,
  num_frames: int,
  temperature: float=0.0
)-> Tensor:
  spec = tokenizer.token_spec
  tpf = spec.grid_height * spec.grid_width

  grids = rollout(model, seed_tokens, actions, num_frames, tpf, temperature)

  batch, frames, _ = grids.shape
  # (B*F, h, w) -> decode -> (B*F, C, H, W) -> (B, F, C, H, W)
  flat = grids.reshape(batch * frames, spec.grid_height, spec.grid_width)
  pixels = tokenizer.decode(flat)
  return pixels.reshape(batch, frames, *pixels.shape[1:])