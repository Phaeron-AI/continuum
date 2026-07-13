from __future__ import annotations

import torch
from torch import Tensor

from engine.models.world_model.world_model import WorldModel

def _pick(logits: Tensor, temperature: float)-> Tensor:
  if temperature <= 0.0:
    return logits.argmax(dim=-1)
  probs = torch.softmax(logits / temperature, dim=-1)
  return torch.multinomial(probs, num_samples=1).squeeze(-1)

@torch.no_grad()
def rollout(
  model: WorldModel,
  seed_tokens: Tensor,
  actions: Tensor,
  num_frames: int,
  tokens_per_frame: int,
  temperature: float = 0.0
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
      nxt = _pick(logits, temperature)  # (B,)
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