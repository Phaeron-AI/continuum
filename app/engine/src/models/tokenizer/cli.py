from __future__ import annotations

import argparse
import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from data.loading.frame_dataset import FrameDataset
from data.loading.split import split_dataset
from envs.config_registry import dataclass_factory
from models.tokenizer.checkpoint import build_tokenizer_from_checkpoint
from models.tokenizer.config import TokenizerConfig
from models.tokenizer.evaluate import EvalReport, evaluate_tokenizer
from models.tokenizer.tokenizer import Tokenizer
from models.tokenizer.train import TrainConfig, train_tokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
  dataset_dir: str = "data/raw/phase0_grid2d_v1"
  batch_size: int = 64
  num_workers: int = 4
  eval_fraction: float = 0.1
  split_seed: int = 0
  input_size: int = 64
  eval_on_finish: bool = True


def _load_configs(
  path: Path | str,
) -> tuple[TokenizerConfig, TrainConfig, DataConfig]:
  raw = yaml.safe_load(Path(path).read_text())
  if not isinstance(raw, dict):
    raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")

  tok_raw = dict(raw.get("tokenizer", {}))
  if "levels" in tok_raw:
    tok_raw["levels"] = tuple(tok_raw["levels"])

  tok = dataclass_factory(TokenizerConfig, "tokenizer")(tok_raw)
  train = dataclass_factory(TrainConfig, "train")(dict(raw.get("train", {})))
  data = dataclass_factory(DataConfig, "data")(dict(raw.get("data", {})))
  return tok, train, data


def run_training(
  config_path: Path | str,
  overrides: dict | None = None,
  resume_from: str | None = None,
) -> tuple[Path, EvalReport | None]:
  """Executes a full training run from a config file.

  Returns (checkpoint_dir, eval_report). eval_report is None if evaluation
  was disabled. `overrides` applies onto TrainConfig; `resume_from` loads a
  checkpoint to continue.
  """
  tok_cfg, train_cfg, data_cfg = _load_configs(config_path)
  if overrides:
    train_cfg = dataclasses.replace(train_cfg, **overrides)

  dataset = FrameDataset(
    data_cfg.dataset_dir, target_size=(data_cfg.input_size, data_cfg.input_size)
  )
  train_set, eval_set = split_dataset(
    dataset, eval_fraction=data_cfg.eval_fraction, seed=data_cfg.split_seed
  )
  train_loader = DataLoader(
    train_set,
    batch_size=data_cfg.batch_size,
    shuffle=True,
    num_workers=data_cfg.num_workers,
    drop_last=True,
  )

  tokenizer = Tokenizer(tok_cfg)
  resume_state = None
  if resume_from:
    tokenizer, resume_state = build_tokenizer_from_checkpoint(resume_from)
    logger.info("Resuming from %s", resume_from)

  logger.info(
    "Training: %d train / %d eval frames, %d steps -> %s",
    len(train_set), len(eval_set), train_cfg.max_steps, train_cfg.checkpoint_dir,
  )
  train_tokenizer(tokenizer, train_loader, train_cfg, resume_state=resume_state)

  report: EvalReport | None = None
  if data_cfg.eval_on_finish:
    eval_loader = DataLoader(
      eval_set,
      batch_size=data_cfg.batch_size,
      shuffle=False,
      num_workers=data_cfg.num_workers,
    )
    report = evaluate_tokenizer(tokenizer, eval_loader, device=train_cfg.device)
    logger.info("Held-out eval: %s", report.summary())

  return Path(train_cfg.checkpoint_dir), report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Train a tokenizer on a Phase 0 dataset.")
  p.add_argument("--config", required=True, help="Path to a training config YAML.")
  p.add_argument("--steps", type=int, default=None, help="Override max_steps.")
  p.add_argument("--device", default=None, help="Override device (cpu/cuda/auto).")
  p.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
  p.add_argument("--log-level", default="INFO")
  return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = _parse_args(argv)
  logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(levelname)s %(name)s: %(message)s",
  )
  log = logging.getLogger("train_tokenizer")

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