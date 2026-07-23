"""Smoke-test the interactive session path (open -> advance) outside the server,
so any error prints to your OWN terminal instead of hiding in the uvicorn log.

    python scripts/smoke_session.py \
      --model checkpoints/world_model/phase3_player/step_0008000.pt \
      --tokenizer checkpoints/tokenizer/phase1_v1/step_0010000.pt \
      --seed seeds/seed_00.png
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from data.storage.transforms import frame_to_tensor
from models.tokenizer.frozen import FrozenTokenizer
from models.world_model.model.checkpoint import build_world_model_from_checkpoint
from models.world_model.session import GenerationSession


def main() -> int:
  ap = argparse.ArgumentParser(description="Exercise a GenerationSession outside the server.")
  ap.add_argument("--model", required=True)
  ap.add_argument("--tokenizer", required=True)
  ap.add_argument("--seed", required=True, help="A seed PNG frame.")
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--frames", type=int, default=3)
  args = ap.parse_args()

  print(f"device = {args.device}")
  tok = FrozenTokenizer.from_checkpoint(args.tokenizer, device=args.device)
  model, _ = build_world_model_from_checkpoint(
    args.model, map_location=args.device,
    expect_tokenizer_version=tok.token_spec.tokenizer_version,
  )
  model.to(args.device).eval()
  print(
    f"model loaded: vocab={model.config.vocab_size} mixer={model.config.mixer} "
    f"action_conditioning={model.config.action_conditioning}"
  )

  # Exactly what the server does on POST /sessions.
  img = Image.open(args.seed).convert("RGB")
  arr = np.asarray(img, dtype=np.uint8)
  frame = frame_to_tensor(arr, target_size=(64, 64)).unsqueeze(0).to(args.device)  # (1,C,H,W)
  seed_tokens = tok.encode(frame).reshape(1, -1)  # (1, h*w)
  print(f"seed encoded: frame {tuple(frame.shape)} -> tokens {tuple(seed_tokens.shape)}")

  session = GenerationSession(model, tok)
  session.open(seed_tokens)
  print("session opened; advancing...")

  for i in range(args.frames):
    action = torch.tensor([4], dtype=torch.long)  # CPU tensor, exactly like the WS handler
    out = session.advance(action, return_pixels=True, temperature=0.9)
    print(f"  frame {i}: {tuple(out.shape)}  pixel range [{out.min():.2f}, {out.max():.2f}]")

  print("OK — advance() works end to end.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())