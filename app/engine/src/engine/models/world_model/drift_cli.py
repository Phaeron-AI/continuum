from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import torch

from engine.data.token_cache import load_cache_manifest
from engine.models.tokenizer.frozen import FrozenTokenizer
from engine.models.world_model.checkpoint import build_world_model_from_checkpoint
from engine.models.world_model.drift import DriftReport, evaluate_drift

logger = logging.getLogger(__name__)


def _load_rollout_batch(
  cache_dir: Path, num_frames: int, batch_size: int, tokens_per_frame: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Pulls (seed_tokens, actions, true_future_tokens) from the token cache.

  The seed is one real frame; the truth is the real continuation. The model
  never sees the truth — it only sees the seed and the actions.
  """
  seeds, actions, truths = [], [], []
  with h5py.File(cache_dir / "tokens.h5", "r") as f:
    for episode_id in f.keys():
      tokens = f[episode_id]["tokens"][:]  # (T+1, h, w)  # type: ignore
      acts = f[episode_id]["actions"][:]  # (T,)  # type: ignore
      if tokens.shape[0] < num_frames + 1 or acts.shape[0] < num_frames:  # type: ignore
        continue

      seeds.append(torch.tensor(tokens[0].reshape(-1), dtype=torch.long)) # type: ignore
      actions.append(torch.tensor(acts[:num_frames], dtype=torch.long)) # type: ignore
      truths.append(
        torch.tensor(
          tokens[1 : num_frames + 1].reshape(num_frames, tokens_per_frame), # type: ignore
          dtype=torch.long,
        )
      )
      if len(seeds) >= batch_size:
        break

  if not seeds:
    raise ValueError(
      f"no episode in the cache is long enough for a {num_frames}-frame rollout"
    )
  return torch.stack(seeds), torch.stack(actions), torch.stack(truths)


def run_drift_eval(
  world_model_checkpoint: str,
  tokenizer_checkpoint: str,
  cache_dir: str,
  num_frames: int = 8,
  batch_size: int = 8,
  psnr_threshold: float = 20.0,
  device: str = "cpu",
) -> DriftReport:
  cache = Path(cache_dir)
  manifest = load_cache_manifest(cache)

  frozen = FrozenTokenizer.from_checkpoint(tokenizer_checkpoint, device=device)
  spec = frozen.token_spec
  tpf = spec.grid_height * spec.grid_width

  # The guard: the world model must speak the same tokenizer's ids.
  model, _ = build_world_model_from_checkpoint(
    world_model_checkpoint,
    map_location=device,
    expect_tokenizer_version=manifest["tokenizer_version"],
  )
  model.to(device)

  seeds, actions, truths = _load_rollout_batch(cache, num_frames, batch_size, tpf)
  seeds, actions, truths = seeds.to(device), actions.to(device), truths.to(device)

  logger.info(
    "Rolling out %d frames from %d seeds (greedy)...", num_frames, seeds.shape[0]
  )
  report = evaluate_drift(
    model, frozen, seeds, actions, truths,
    num_frames=num_frames,
    psnr_threshold=psnr_threshold,
    temperature=0.0,  # greedy: divergence is the model's error, not noise
  )
  logger.info("Drift: %s", report.summary())
  return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Evaluate world-model rollout drift.")
  p.add_argument("--world-model", required=True, help="World model checkpoint.")
  p.add_argument("--tokenizer", required=True, help="Tokenizer checkpoint.")
  p.add_argument("--cache", required=True, help="Token cache directory.")
  p.add_argument("--frames", type=int, default=8, help="Rollout depth.")
  p.add_argument("--batch-size", type=int, default=8)
  p.add_argument("--psnr-threshold", type=float, default=20.0)
  p.add_argument("--device", default="cpu")
  p.add_argument("--log-level", default="INFO")
  return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = _parse_args(argv)
  logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(levelname)s %(name)s: %(message)s",
  )
  log = logging.getLogger("evaluate_drift")
  try:
    report = run_drift_eval(
      args.world_model,
      args.tokenizer,
      args.cache,
      num_frames=args.frames,
      batch_size=args.batch_size,
      psnr_threshold=args.psnr_threshold,
      device=args.device,
    )
  except (FileNotFoundError, ValueError) as exc:
    log.error("Drift evaluation failed: %s", exc)
    return 1
  print(report.summary())
  return 0