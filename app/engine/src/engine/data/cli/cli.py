"""Dataset-generation CLI logic.

Lives in the package (not scripts/) so it's importable and testable. The
scripts/ entrypoint is a trivial shim over main() here. Keeping the real
command-line logic inside engine means the pipeline is fully exercised by
the test suite and reusable from other orchestration, while scripts/ holds
only launchers.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging

from engine.data.generation.config_loader import load_generation_config
from engine.data.generation.harness import generate_dataset


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Generate a Phase 0 dataset.")
  parser.add_argument(
    "--config", required=True, help="Path to a generation config YAML file."
  )
  parser.add_argument("--episodes", type=int, default=None, help="Override num_episodes.")
  parser.add_argument("--workers", type=int, default=None, help="Override num_workers.")
  parser.add_argument("--output-dir", default=None, help="Override output_dir.")
  parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = _parse_args(argv)
  logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(levelname)s %(name)s: %(message)s",
  )
  log = logging.getLogger("generate_dataset")

  try:
    config = load_generation_config(args.config)
  except (FileNotFoundError, ValueError) as exc:
    log.error("Failed to load config: %s", exc)
    return 1

  overrides = {}
  if args.episodes is not None:
    overrides["num_episodes"] = args.episodes
  if args.workers is not None:
    overrides["num_workers"] = args.workers
  if args.output_dir is not None:
    overrides["output_dir"] = args.output_dir
  if overrides:
    config = dataclasses.replace(config, **overrides)

  log.info(
    "Generating '%s': %d episodes, %d worker(s) -> %s",
    config.dataset_name,
    config.num_episodes,
    config.num_workers,
    config.output_dir,
  )

  output_dir, report = generate_dataset(config)

  log.info("Done. Wrote dataset to %s", output_dir)
  print(report.summary())
  return 0