"""Headless, parallel episode generation.

Wires together the seams built in sub-phases 0.1-0.3: an Env produces
observations, a Policy picks action ids, and a ShardWriter persists the
result. Environments and policies are resolved by name through their
registries — this module never imports a concrete env or policy class,
which is what keeps the future 2D -> 3D swap out of this file entirely.

Parallelism model: episodes are independent given their seeds, so workers
each generate complete episodes in their own process (pygame/HDF5 state is
per-process, never shared). Workers return episodes to the parent, and the
parent alone writes shards — single-writer, so shard contents stay
deterministic regardless of worker scheduling. Serial and parallel modes
share one consumption loop; the only thing the worker count changes is
which iterator yields the episodes.

Determinism contract: dataset content is fully determined by
(GenerationConfig, env code version). Episode i always uses seed
seed_start + i for both the env and its policy, so any single episode can
be regenerated in isolation for debugging.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from engine.data.storage.schema import DatasetManifest, EpisodeMetadata
from engine.data.storage.storage import ShardWriter, write_manifest
from engine.envs.base import Action, Env
from engine.envs.grid2d.config import Grid2DConfig
from engine.envs.policies.registry import make_policy
from engine.envs.registry import make_env

Episode = tuple[np.ndarray, np.ndarray, EpisodeMetadata]


@dataclass(frozen=True)
class GenerationConfig:
  """Everything that determines a generation run's output."""

  dataset_name: str
  output_dir: str
  num_episodes: int
  seed_start: int = 0
  episodes_per_shard: int = 100
  num_workers: int = 1
  env_name: str = "grid2d"
  policy_name: str = "random"
  env: Any = field(default_factory=Grid2DConfig)

  def __post_init__(self) -> None:
    if self.num_episodes <= 0:
      raise ValueError(f"num_episodes must be positive, got {self.num_episodes}")
    if self.num_workers <= 0:
      raise ValueError(f"num_workers must be positive, got {self.num_workers}")


def _generate_one_episode(config: GenerationConfig, episode_index: int) -> Episode:
  """Runs one complete episode. Executed inside a worker process; must not
  touch any shared state."""
  seed = config.seed_start + episode_index
  env: Env = make_env(config.env_name, config.env)
  policy = make_policy(
    config.policy_name,
    num_actions=len(Action),
    rng=np.random.default_rng(seed),
  )

  frames = [env.reset(seed=seed)]
  actions: list[int] = []
  done = False
  while not done:
    action_id = policy.act(frames[-1])
    result = env.step(Action(action_id))
    frames.append(result.observation)
    actions.append(action_id)
    done = result.done

  metadata = EpisodeMetadata(
    episode_id=f"ep_{episode_index:06d}",
    seed=seed,
    num_steps=len(actions),
    env_name=config.env_name,
    env_version=env.env_version,  
    action_space_version=env.action_space_version,
    policy_name=config.policy_name,
  )
  return np.stack(frames), np.asarray(actions, dtype=np.int64), metadata


@dataclass
class CoverageReport:
  """Cheap post-generation sanity signals.

  Exists to catch degenerate data the moment it's generated instead of
  three phases later when the world model mysteriously won't learn. Random
  policies in particular tend to produce boring, low-coverage trajectories
  — this report is how that shows up as a number instead of a hunch.
  """

  action_counts: dict[int, int]
  unique_frames_ratio: float  # mean over episodes: unique frames / frames
  num_episodes: int
  total_steps: int

  def summary(self) -> str:
    total_actions = sum(self.action_counts.values()) or 1
    dist = ", ".join(
      f"{Action(a).name}: {c / total_actions:.1%}"
      for a, c in sorted(self.action_counts.items())
    )
    return (
      f"episodes={self.num_episodes} steps={self.total_steps}\n"
      f"action distribution: {dist}\n"
      f"mean unique-frame ratio: {self.unique_frames_ratio:.1%} "
      f"(low values mean the agent barely moves — consider a "
      f"scripted policy)"
    )


class _CoverageAccumulator:
  """Folds per-episode stats into a final CoverageReport. Extracted so the
  consumption loop stays a plain loop and the accounting lives (and is
  testable) in one place."""

  def __init__(self) -> None:
    self._action_counts: dict[int, int] = {}
    self._unique_ratios: list[float] = []
    self._total_steps = 0
    self._num_episodes = 0

  def add(self, frames: np.ndarray, actions: np.ndarray) -> None:
    for a in actions.tolist():
      self._action_counts[a] = self._action_counts.get(a, 0) + 1
    # Hash frames cheaply to count distinct visual states visited.
    hashes = {hash(frame.tobytes()) for frame in frames}
    self._unique_ratios.append(len(hashes) / len(frames))
    self._total_steps += len(actions)
    self._num_episodes += 1

  def report(self) -> CoverageReport:
    return CoverageReport(
      action_counts=dict(self._action_counts),
      unique_frames_ratio=(
        float(np.mean(self._unique_ratios)) if self._unique_ratios else 0.0
      ),
      num_episodes=self._num_episodes,
      total_steps=self._total_steps,
    )


def _episode_iterator(config: GenerationConfig) -> Iterator[Episode]:
  """Yields episodes in index order. The worker count decides only where
  generation happens — the consumer downstream is identical either way."""
  indices = range(config.num_episodes)
  if config.num_workers == 1:
    for i in indices:
      yield _generate_one_episode(config, i)
  else:
    with ProcessPoolExecutor(max_workers=config.num_workers) as pool:
      # map() preserves submission order, so shard contents are
      # identical regardless of which worker finishes first.
      yield from pool.map(
        _generate_one_episode, [config] * config.num_episodes, indices
      )


def _current_git_commit() -> str | None:
  try:
    return (
      subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
      ).stdout.strip()
      or None
    )
  except (subprocess.CalledProcessError, FileNotFoundError):
    return None


def _build_manifest(config: GenerationConfig, total_steps: int) -> DatasetManifest:
  env = make_env(config.env_name, config.env)
  spec = env.observation_spec
  return DatasetManifest(
    dataset_name=config.dataset_name,
    env_name=config.env_name,
    env_version=env.env_version,
    action_space_version=env.action_space_version,
    policy_name=config.policy_name,
    observation_height=spec.height,
    observation_width=spec.width,
    observation_channels=spec.channels,
    observation_dtype=spec.dtype,
    num_episodes=config.num_episodes,
    total_steps=total_steps,
    seed_start=config.seed_start,
    seed_end=config.seed_start + config.num_episodes - 1,
    git_commit=_current_git_commit(),
  )


def generate_dataset(config: GenerationConfig) -> tuple[Path, CoverageReport]:
  """Generates the full dataset described by `config`.

  Returns (output_dir, coverage_report). Writes shards + manifest to
  config.output_dir. The parent process is the only writer; workers only
  generate.
  """
  output_dir = Path(config.output_dir)
  writer = ShardWriter(output_dir, episodes_per_shard=config.episodes_per_shard)
  coverage = _CoverageAccumulator()

  try:
    for frames, actions, metadata in _episode_iterator(config):
      writer.add_episode(frames, actions, metadata)
      coverage.add(frames, actions)
  finally:
    writer.close()

  report = coverage.report()
  write_manifest(output_dir, _build_manifest(config, total_steps=report.total_steps))
  return output_dir, report