from __future__ import annotations

import argparse
import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader, Subset

from data import SequenceDataset, load_cache_manifest
from envs import dataclass_factory
from models.world_model.model import WorldModelConfig, WorldModel, build_world_model_from_checkpoint
from models.world_model.training import TrainConfig, train_world_model, WMEvalReport, evaluate_world_model


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
  cache_dir: str = "data/token_cache"
  context_frames: int = 4
  num_actions: int = 5
  batch_size: int = 32
  num_workers: int = 4
  eval_fraction: float = 0.1
  split_seed: int = 0
  eval_on_finish: bool = True


def _load_configs(path: Path | str) -> tuple[dict, TrainConfig, DataConfig]:
  """Returns (raw model overrides, TrainConfig, DataConfig).

  The model config is returned as a raw dict, not a WorldModelConfig, because
  vocab_size is not known until the cache manifest is read — it comes from
  the tokenizer, not from the YAML.
  """
  raw = yaml.safe_load(Path(path).read_text())
  if not isinstance(raw, dict):
    raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")

  model_raw = dict(raw.get("model", {}))
  train = dataclass_factory(TrainConfig, "train")(dict(raw.get("train", {})))
  data = dataclass_factory(DataConfig, "data")(dict(raw.get("data", {})))
  return model_raw, train, data


def _split(dataset: SequenceDataset, eval_fraction: float, seed: int):
  n = len(dataset)
  rng = np.random.default_rng(seed)
  perm = rng.permutation(n)
  n_eval = max(1, int(n * eval_fraction))
  return (
    Subset(dataset, perm[n_eval:].tolist()),
    Subset(dataset, perm[:n_eval].tolist()),
  )


def run_training(
  config_path: Path | str,
  overrides: dict | None = None,
  resume_from: str | None = None,
) -> tuple[Path, WMEvalReport | None]:
  model_raw, train_cfg, data_cfg = _load_configs(config_path)
  if overrides:
    train_cfg = dataclasses.replace(train_cfg, **overrides)

  # The cache is a prerequisite — built once by build_token_cache.
  cache_dir = Path(data_cfg.cache_dir)
  if not (cache_dir / "tokens.h5").exists():
    raise FileNotFoundError(
      f"no token cache at {cache_dir} — run build_token_cache first"
    )
  manifest = load_cache_manifest(cache_dir)

  dataset = SequenceDataset(
    cache_dir,
    context_frames=data_cfg.context_frames,
    num_actions=data_cfg.num_actions,
    tokenizer_version=manifest["tokenizer_version"],  # the guard
  )
  train_set, eval_set = _split(dataset, data_cfg.eval_fraction, data_cfg.split_seed)
  train_loader = DataLoader(
    train_set,
    batch_size=data_cfg.batch_size,
    shuffle=True,
    num_workers=data_cfg.num_workers,
    drop_last=True,
  )

  # Vocab comes from the TOKENIZER (via the cache), never from the YAML.
  model_cfg = WorldModelConfig(
    vocab_size=manifest["vocab_size"],
    num_actions=data_cfg.num_actions,
    tokenizer_version=manifest["tokenizer_version"],
    **model_raw,
  )

  model = WorldModel(model_cfg)
  resume_state = None
  if resume_from:
    model, resume_state = build_world_model_from_checkpoint(
      resume_from, expect_tokenizer_version=manifest["tokenizer_version"]
    )
    logger.info("Resuming from %s", resume_from)

  logger.info(
    "Training world model: %d train / %d eval windows | vocab=%d | %d steps -> %s",
    len(train_set), len(eval_set), model_cfg.vocab_size,
    train_cfg.max_steps, train_cfg.checkpoint_dir,
  )
  train_world_model(model, train_loader, train_cfg, resume_state=resume_state)

  report: WMEvalReport | None = None
  if data_cfg.eval_on_finish:
    eval_loader = DataLoader(
      eval_set, batch_size=data_cfg.batch_size, shuffle=False,
      num_workers=data_cfg.num_workers,
    )
    report = evaluate_world_model(model, eval_loader, device=train_cfg.device)
    logger.info("Held-out eval: %s", report.summary())

  return Path(train_cfg.checkpoint_dir), report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Train a world model on a token cache.")
  p.add_argument("--config", required=True, help="Training config YAML.")
  p.add_argument("--steps", type=int, default=None, help="Override max_steps.")
  p.add_argument("--device", default=None, help="Override device.")
  p.add_argument("--resume", default=None, help="Checkpoint to resume from.")
  p.add_argument("--log-level", default="INFO")
  return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = _parse_args(argv)
  logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(levelname)s %(name)s: %(message)s",
  )
  log = logging.getLogger("train_world_model")

  overrides: dict = {}
  if args.steps is not None:
    overrides["max_steps"] = args.steps
  if args.device is not None:
    overrides["device"] = args.device

  try:
    ckpt_dir, report = run_training(
      args.config, overrides=overrides or None, resume_from=args.resume
    )
  except (FileNotFoundError, ValueError) as exc:
    log.error("Training failed: %s", exc)
    return 1

  log.info("Done. Checkpoints in %s", ckpt_dir)
  if report is not None:
    print(report.summary())
  return 0