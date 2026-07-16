from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data.tokens.token_cache import build_token_cache, load_cache_manifest
from models.tokenizer.frozen import FrozenTokenizer

logger = logging.getLogger(__name__)


def run_build(
    dataset_dir: str,
    tokenizer_checkpoint: str,
    output_dir: str,
    device: str = "cpu",
    batch_size: int = 64,
) -> Path:
  frozen = FrozenTokenizer.from_checkpoint(tokenizer_checkpoint, device=device)
  spec = frozen.token_spec
  logger.info(
    "Tokenizing %s with tokenizer %s (vocab=%d, grid=%dx%d) -> %s",
    dataset_dir, spec.tokenizer_version, spec.vocab_size,
    spec.grid_height, spec.grid_width, output_dir,
  )
  out = build_token_cache(
    dataset_dir, frozen, output_dir, device=device, batch_size=batch_size
  )
  manifest = load_cache_manifest(out)
  logger.info(
    "Cache built: %d episodes, tokenizer_version=%s",
    manifest["num_episodes"], manifest["tokenizer_version"],
  )
  return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Build a token cache from a dataset.")
  p.add_argument("--dataset", required=True, help="Phase 0 dataset directory.")
  p.add_argument("--tokenizer", required=True, help="Trained tokenizer checkpoint (.pt).")
  p.add_argument("--output", required=True, help="Cache output directory.")
  p.add_argument("--device", default="cpu", help="Device for tokenization.")
  p.add_argument("--batch-size", type=int, default=64)
  p.add_argument("--log-level", default="INFO")
  return p.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
  args = _parse_args(argv)
  logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(levelname)s %(name)s: %(message)s",
  )
  log = logging.getLogger("build_token_cache")
  try:
    out = run_build(
      args.dataset, args.tokenizer, args.output,
      device=args.device, batch_size=args.batch_size,
    )
  except (FileNotFoundError, ValueError) as exc:
    log.error("Cache build failed: %s", exc)
    return 1
  log.info("Done. Cache at %s", out)
  return 0