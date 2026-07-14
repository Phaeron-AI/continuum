from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py

from engine.data.storage.storage import __MANIFEST_FILENAME__

@dataclass(frozen=True)
class FrameLocator:
  shard_path: str
  episode_id: str
  frame_idx: int


def build_frame_index(dataset_dir: Path | str) -> list[FrameLocator]:
  """Scans every shard's metadata and returns one FrameLocator per frame.

  Reads only dataset shapes (h5py exposes these without loading data), so
  the cost is proportional to episode count, not total pixel volume.
  """
  dataset_dir = Path(dataset_dir)
  shard_paths = sorted(dataset_dir.glob("shard_*.h5"))
  if not shard_paths:
      raise FileNotFoundError(f"No shard_*.h5 files found in {dataset_dir}")

  index: list[FrameLocator] = []
  for shard_path in shard_paths:
    with h5py.File(shard_path, "r") as f:
      for episode_id in f.keys():
        group = f[episode_id]
        if not isinstance(group, h5py.Group):
          raise TypeError(
            f"Expected {episode_id!r} in {shard_path} to be a Group."
          )
        frames = group["frames"]
        if not isinstance(frames, h5py.Dataset):
          raise TypeError(
            f"Expected 'frames' in {episode_id!r} to be a Dataset."
          )
        num_frames = frames.shape[0]  # shape read, pixels not loaded
        for frame_idx in range(num_frames):
          index.append(
            FrameLocator(str(shard_path), episode_id, frame_idx)
          )
  return index

def has_manifest(dataset_dir: Path | str) -> bool:
    return (Path(dataset_dir) / __MANIFEST_FILENAME__).exists()