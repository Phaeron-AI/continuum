from __future__ import annotations

import argparse
import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from data import SequenceDataset, load_cache_manifest
from models.device import resolve_device
from models.tokenizer import FrozenTokenizer
from models.world_model.inference import evaluate_drift
from models.world_model.model import WorldModel, WorldModelConfig
from models.world_model.training import TrainConfig, evaluate_world_model, train_world_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Arm:
  """One configuration in the ablation. Arms differ ONLY in the three factors
  under study; every other hyperparameter (dims, steps, lr, data, seed) is held
  fixed so each row's delta is attributable to a single change."""

  name: str
  mixer: str
  action_conditioning: str
  context_noise_prob: float


# Additive ablation: each arm turns on one more of the new capabilities, so the
# marginal effect of mamba, then film, then noise-aug is read straight off the
# table. Trim or extend this list as needed.
DEFAULT_ARMS: tuple[Arm, ...] = (
  Arm("baseline (ssm)", "ssm", "inline", 0.0),
  Arm("+ mamba block", "mamba", "inline", 0.0),
  Arm("+ film actions", "mamba", "film", 0.0),
  Arm("+ noise aug (full)", "mamba", "film", 0.15),
)


@dataclass
class ArmResult:
  name: str
  params: int
  ce_loss: float
  next_token_accuracy: float
  drift_horizon: int
  drift_summary: str


def _split(dataset: SequenceDataset, eval_fraction: float, seed: int):
  """Identical to the training CLI's split so ablation numbers line up with a
  normal training run."""
  n = len(dataset)
  perm = np.random.default_rng(seed).permutation(n)
  n_eval = max(1, int(n * eval_fraction))
  return (
    Subset(dataset, perm[n_eval:].tolist()),
    Subset(dataset, perm[:n_eval].tolist()),
  )


def _rollout_batch(
  cache_dir: Path, num_frames: int, batch_size: int, tokens_per_frame: int
) -> tuple[Tensor, Tensor, Tensor]:
  """One fixed (seed, actions, ground-truth) batch, shared by every arm so the
  drift comparison is apples-to-apples. Mirrors the drift CLI's loader."""
  seeds, actions, truths = [], [], []
  with h5py.File(cache_dir / "tokens.h5", "r") as f:
    for episode_id in f.keys():
      tokens = f[episode_id]["tokens"][:]  # type: ignore
      acts = f[episode_id]["actions"][:]  # type: ignore
      if tokens.shape[0] < num_frames + 1 or acts.shape[0] < num_frames:  # type: ignore
        continue
      seeds.append(torch.tensor(tokens[0].reshape(-1), dtype=torch.long))  # type: ignore
      actions.append(torch.tensor(acts[:num_frames], dtype=torch.long))  # type: ignore
      truths.append(
        torch.tensor(
          tokens[1 : num_frames + 1].reshape(num_frames, tokens_per_frame),  # type: ignore
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


def _run_arm(
  arm: Arm,
  *,
  train_set,
  eval_set,
  manifest: dict,
  frozen: FrozenTokenizer,
  drift_batch: tuple[Tensor, Tensor, Tensor],
  base_train: TrainConfig,
  model_kw: dict,
  data_kw: dict,
  device: torch.device,
  num_frames: int,
  psnr_threshold: float,
  seed: int,
) -> ArmResult:
  torch.manual_seed(seed)  # same init seed for every arm
  cfg = WorldModelConfig(
    vocab_size=manifest["vocab_size"],
    num_actions=data_kw["num_actions"],
    tokenizer_version=manifest["tokenizer_version"],
    mixer=arm.mixer,
    action_conditioning=arm.action_conditioning,
    **model_kw,
  )
  model = WorldModel(cfg)
  params = sum(p.numel() for p in model.parameters())

  # Fixed generator => identical batch ORDER across arms, independent of how
  # much RNG each architecture consumes at init.
  gen = torch.Generator().manual_seed(seed)
  train_loader = DataLoader(
    train_set,
    batch_size=data_kw["batch_size"],
    shuffle=True,
    num_workers=data_kw["num_workers"],
    drop_last=True,
    generator=gen,
  )
  train_cfg = dataclasses.replace(base_train, context_noise_prob=arm.context_noise_prob)

  logger.info(
    "[%s] training %d steps (%.2fM params)...", arm.name, train_cfg.max_steps, params / 1e6
  )
  train_world_model(model, train_loader, train_cfg)

  eval_loader = DataLoader(
    eval_set, batch_size=data_kw["batch_size"], shuffle=False, num_workers=data_kw["num_workers"]
  )
  token_report = evaluate_world_model(model, eval_loader, device=base_train.device)

  model.to(device)
  seeds, act, truths = (t.to(device) for t in drift_batch)
  drift = evaluate_drift(
    model, frozen, seeds, act, truths,
    num_frames=num_frames, psnr_threshold=psnr_threshold, temperature=0.0,
  )

  return ArmResult(
    name=arm.name,
    params=params,
    ce_loss=token_report.ce_loss,
    next_token_accuracy=token_report.next_token_accuracy,
    drift_horizon=drift.drift_horizon,
    drift_summary=drift.summary(),
  )


def render_markdown(results: list[ArmResult], num_frames: int, psnr_threshold: float) -> str:
  lines = [
    f"| arm | params | ce loss ↓ | next-token acc ↑ | "
    f"drift horizon ↑ (PSNR>{psnr_threshold:.0f}dB, /{num_frames}) |",
    "| --- | ---: | ---: | ---: | ---: |",
  ]
  for r in results:
    lines.append(
      f"| {r.name} | {r.params / 1e6:.2f}M | {r.ce_loss:.3f} | "
      f"{r.next_token_accuracy:.1%} | {r.drift_horizon} |"
    )
  if len(results) >= 2:
    base, full = results[0], results[-1]
    lines += [
      "",
      f"Full stack vs baseline: drift horizon {base.drift_horizon} → "
      f"{full.drift_horizon} frames, next-token accuracy "
      f"{base.next_token_accuracy:.1%} → {full.next_token_accuracy:.1%}.",
    ]
  return "\n".join(lines)


def run_ablation(
  cache_dir: str,
  tokenizer_checkpoint: str,
  *,
  arms: tuple[Arm, ...] = DEFAULT_ARMS,
  steps: int = 4000,
  lr: float = 3e-4,
  device: str = "auto",
  d_model: int = 128,
  d_state: int = 16,
  d_conv: int = 4,
  n_layers: int = 4,
  ffn_mult: int = 4,
  num_actions: int = 5,
  context_frames: int = 4,
  batch_size: int = 32,
  num_workers: int = 0,
  eval_fraction: float = 0.1,
  split_seed: int = 0,
  seed: int = 0,
  num_frames: int = 8,
  drift_batch_size: int = 8,
  psnr_threshold: float = 20.0,
  out: str | None = None,
) -> list[ArmResult]:
  cache = Path(cache_dir)
  if not (cache / "tokens.h5").exists():
    raise FileNotFoundError(f"no token cache at {cache} — run build_token_cache first")
  manifest = load_cache_manifest(cache)
  dev = resolve_device(device)

  dataset = SequenceDataset(
    cache, context_frames=context_frames, num_actions=num_actions,
    tokenizer_version=manifest["tokenizer_version"],
  )
  train_set, eval_set = _split(dataset, eval_fraction, split_seed)

  frozen = FrozenTokenizer.from_checkpoint(tokenizer_checkpoint, device=str(dev))
  spec = frozen.token_spec
  tpf = spec.grid_height * spec.grid_width
  drift_batch = _rollout_batch(cache, num_frames, drift_batch_size, tpf)

  base_train = TrainConfig(
    max_steps=steps, lr=lr, device=device, amp=(dev.type == "cuda"),
    log_every=max(1, steps // 10), checkpoint_every=steps + 1,  # skip mid-run ckpts
    checkpoint_dir="checkpoints/ablation",
  )
  model_kw = dict(
    d_model=d_model, d_state=d_state, d_conv=d_conv, n_layers=n_layers, ffn_mult=ffn_mult
  )
  data_kw = dict(num_actions=num_actions, batch_size=batch_size, num_workers=num_workers)

  results: list[ArmResult] = []
  for arm in arms:
    results.append(
      _run_arm(
        arm, train_set=train_set, eval_set=eval_set, manifest=manifest, frozen=frozen,
        drift_batch=drift_batch, base_train=base_train, model_kw=model_kw, data_kw=data_kw,
        device=dev, num_frames=num_frames, psnr_threshold=psnr_threshold, seed=seed,
      )
    )

  table = render_markdown(results, num_frames, psnr_threshold)
  print("\n" + table + "\n")
  if out:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(table + "\n")
    logger.info("Wrote %s", out)
  return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Additive ablation of the world-model improvements.")
  p.add_argument("--cache", required=True, help="Token cache directory.")
  p.add_argument("--tokenizer", required=True, help="Frozen tokenizer checkpoint.")
  p.add_argument("--steps", type=int, default=4000)
  p.add_argument("--device", default="auto")
  p.add_argument("--d-model", type=int, default=128)
  p.add_argument("--n-layers", type=int, default=4)
  p.add_argument("--frames", type=int, default=8, help="Rollout depth for drift.")
  p.add_argument("--drift-batch", type=int, default=8)
  p.add_argument("--psnr-threshold", type=float, default=20.0)
  p.add_argument("--out", default="results/ablation.md", help="Markdown table output.")
  p.add_argument("--log-level", default="INFO")
  return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  torch.backends.cudnn.enabled = False
  args = _parse_args(argv)
  logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(levelname)s %(name)s: %(message)s",
  )
  try:
    run_ablation(
      args.cache, args.tokenizer, steps=args.steps, device=args.device,
      d_model=args.d_model, n_layers=args.n_layers, num_frames=args.frames,
      drift_batch_size=args.drift_batch, psnr_threshold=args.psnr_threshold, out=args.out,
    )
  except (FileNotFoundError, ValueError) as exc:
    logging.getLogger("ablation").error("Ablation failed: %s", exc)
    return 1
  return 0