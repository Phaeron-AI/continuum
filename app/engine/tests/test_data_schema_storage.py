"""Sub-phase 0.3 verification: shard round-trip and manifest round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.engine.data.schema import DatasetManifest, EpisodeMetadata
from src.engine.data.storage import (
  ShardWriter,
  iterate_episodes,
  load_manifest,
  write_manifest,
)


def _make_fake_episode(episode_id: str, num_steps: int = 5):
  frames = np.random.randint(0, 255, size=(num_steps + 1, 8, 8, 3), dtype=np.uint8)
  actions = np.random.randint(0, 5, size=(num_steps,), dtype=np.int64)
  metadata = EpisodeMetadata(
    episode_id=episode_id,
    seed=0,
    num_steps=num_steps,
    env_name="grid2d",
    env_version="v1",
    action_space_version="v1",
    policy_name="random",
  )
  return frames, actions, metadata


def test_shard_round_trip(tmp_path: Path) -> None:
  writer = ShardWriter(tmp_path, episodes_per_shard=2)
  written = {}
  for i in range(3):
    episode_id = f"ep_{i:04d}"
    frames, actions, metadata = _make_fake_episode(episode_id)
    writer.add_episode(frames, actions, metadata)
    written[episode_id] = (frames, actions)
  shard_paths = writer.close()

  # 3 episodes at 2 per shard -> 2 shard files
  assert len(shard_paths) == 2

  read_back = {}
  for shard_path in shard_paths:
    for episode_id, frames, actions, attrs in iterate_episodes(shard_path):
      read_back[episode_id] = (frames, actions)
      assert attrs["env_name"] == "grid2d"
      assert attrs["action_space_version"] == "v1"

  assert set(read_back.keys()) == set(written.keys())
  for episode_id, (frames, actions) in written.items():
    got_frames, got_actions = read_back[episode_id]
    np.testing.assert_array_equal(frames, got_frames)
    np.testing.assert_array_equal(actions, got_actions)


def test_manifest_round_trip(tmp_path: Path) -> None:
  manifest = DatasetManifest(
    dataset_name="phase0_grid2d",
    env_name="grid2d",
    env_version="v1",
    action_space_version="v1",
    policy_name="random",
    observation_height=84,
    observation_width=84,
    observation_channels=3,
    observation_dtype="uint8",
    num_episodes=10,
    total_steps=2000,
    seed_start=0,
    seed_end=9,
  )
  write_manifest(tmp_path, manifest)
  loaded = load_manifest(tmp_path)
  assert loaded.dataset_name == manifest.dataset_name
  assert loaded.num_episodes == manifest.num_episodes
  assert loaded.seed_start == manifest.seed_start
  assert loaded.seed_end == manifest.seed_end