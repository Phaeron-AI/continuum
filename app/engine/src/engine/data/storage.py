from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .schema import DatasetManifest, EpisodeMetadata

__MANIFEST_FILENAME__ = "manifest.json"

class ShardWriter:
  def __init__(self, output_dir: Path | str, episodes_per_shard: int = 100)-> None:
    self._output_dir = Path(output_dir)
    self._output_dir.mkdir(parents=True, exist_ok=True)
    self._episodes_per_shard = episodes_per_shard
    self._shard_index = 0
    self._episodes_in_current_shard = 0
    self._current_file: Optional[h5py.File] = None
    self._shard_paths: list[Path] = []
  
  def add_episode(self, frames: np.ndarray, actions: np.ndarray, metadata: EpisodeMetadata)-> None:
    if (
      self._current_file is None
      or self._episodes_in_current_shard >= self._episodes_per_shard
    ):
      self._roll_shard()

    assert self._current_file is not None
    group = self._current_file.create_group(metadata.episode_id)
    group.create_dataset(
        "frames", data=frames, compression="gzip", compression_opts=4
    )
    group.create_dataset("actions", data=actions)
    group.attrs["seed"] = metadata.seed
    group.attrs["num_steps"] = metadata.num_steps
    group.attrs["env_name"] = metadata.env_name
    group.attrs["env_version"] = metadata.env_version
    group.attrs["action_space_version"] = metadata.action_space_version
    group.attrs["policy_name"] = metadata.policy_name

    self._episodes_in_current_shard += 1
  
  def _roll_shard(self) -> None:
    if self._current_file is not None:
      self._current_file.close()
    shard_path = self._output_dir / f"shard_{self._shard_index:05d}.h5"
    self._current_file = h5py.File(shard_path, "w")
    self._shard_paths.append(shard_path)
    self._shard_index += 1
    self._episodes_in_current_shard = 0
  
  def close(self) -> list[Path]:
    if self._current_file is not None:
      self._current_file.close()
      self._current_file = None
    return self._shard_paths
  
def iterate_episodes(
    shard_path: Path | str,
) -> Iterator[tuple[str, np.ndarray, np.ndarray, dict]]:
  with h5py.File(shard_path, "r") as f:
    for episode_id in f.keys():
      item = f[episode_id]
      if not isinstance(item, h5py.Group):
        raise TypeError(
          f"Expected {episode_id!r} in {shard_path} to be a Group, "
          f"got {type(item).__name__} — shard file may be corrupt."
        )
      group = item

      frames_ds = group["frames"]
      actions_ds = group["actions"]
      if not isinstance(frames_ds, h5py.Dataset) or not isinstance(
        actions_ds, h5py.Dataset
      ):
        raise TypeError(
          f"Expected 'frames'/'actions' in episode {episode_id!r} "
          f"of {shard_path} to be Datasets — shard file may be corrupt."
        )

      frames: np.ndarray = frames_ds[:]
      actions: np.ndarray = actions_ds[:]
      attrs = dict(group.attrs)
      yield episode_id, frames, actions, attrs

def write_manifest(output_dir: Path | str, manifest: DatasetManifest) -> Path:
  path = Path(output_dir) / __MANIFEST_FILENAME__
  path.write_text(json.dumps(manifest.to_dict(), indent=2))
  return path


def load_manifest(output_dir: Path | str) -> DatasetManifest:
  path = Path(output_dir) / __MANIFEST_FILENAME__
  data = json.loads(path.read_text())
  return DatasetManifest.from_dict(data)