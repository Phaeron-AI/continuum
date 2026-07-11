from __future__ import annotations

import json
from pathlib import Path

import h5py
import torch
import numpy as np

from engine.data.storage import iterate_episodes
from engine.data.transforms import frame_to_tensor
from engine.models.tokenizer.frozen import FrozenTokenizer

__CACHE_MANIFEST__ = "token_cache_manifest.json"

def build_token_cache(
  dataset_dir: Path | str,
  tokenizer: FrozenTokenizer,
  output_dir: Path | str,
  device: str = "cpu",
  batch_size: int = 64
)-> Path:
  dataset_dir = Path(dataset_dir)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  spec = tokenizer.token_spec

  shard_paths = sorted(dataset_dir.glob("shard_*.h5"))
  if not shard_paths:
    raise FileNotFoundError(f"No shard_*.h5 files found in {dataset_dir}")
  
  cache_path = output_dir / "tokens.h5"
  num_episodes = 0
  with h5py.File(cache_path, "w") as out:
    for shard_path in shard_paths:
      for episode_id, frames, actions, _attrs in iterate_episodes(shard_path):
        # frames: (T+1, H, W, C) uint8 -> token grid (T+1, h, w)
        token_grids = _encode_frames(
          frames, tokenizer, spec.input_height, device, batch_size
        )
        grp = out.create_group(episode_id)
        grp.create_dataset("tokens", data=token_grids.astype(np.int32))
        grp.create_dataset("actions", data=np.asarray(actions, dtype=np.int64))
        num_episodes += 1

  manifest = {
    "tokenizer_version": spec.tokenizer_version,
    "grid_height": spec.grid_height,
    "grid_width": spec.grid_width,
    "vocab_size": spec.vocab_size,
    "num_episodes": num_episodes,
    "source_dataset": str(dataset_dir),
  }
  (output_dir / __CACHE_MANIFEST__).write_text(json.dumps(manifest, indent=2))
  return output_dir

def _encode_frames(
  frames: np.ndarray,
  tokenizer: FrozenTokenizer,
  input_size: int,
  device: str,
  batch_size: int
)-> np.ndarray:
  grids: list[np.ndarray] = []
  n = frames.shape[0]
  for start in range(0, n, batch_size):
    chunk = frames[start : start + batch_size]
    tensors = torch.stack(
      [frame_to_tensor(f, target_size=(input_size, input_size)) for f in chunk]
    ).to(device)
    idx = tokenizer.encode(tensors)  # (b, h, w)
    grids.append(idx.cpu().numpy())
  return np.concatenate(grids, axis=0)

def load_cache_manifest(cache_dir: Path | str) -> dict:
  return json.loads((Path(cache_dir) / __CACHE_MANIFEST__).read_text())