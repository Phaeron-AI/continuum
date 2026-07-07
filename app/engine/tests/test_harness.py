"""Sub-phase 0.4 verification: the harness generates valid, deterministic,
well-formed datasets -- serially and in parallel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.data.harness import CoverageReport, GenerationConfig, generate_dataset
from engine.data.storage import iterate_episodes, load_manifest
from engine.envs.grid2d.config import Grid2DConfig


def _small_config(output_dir: Path, **overrides) -> GenerationConfig:
    defaults = dict(
        dataset_name="test_run",
        output_dir=str(output_dir),
        num_episodes=4,
        seed_start=0,
        episodes_per_shard=3,
        num_workers=1,
        env=Grid2DConfig(canvas_size=32, episode_length=10, num_obstacles=2),
    )
    defaults.update(overrides)
    return GenerationConfig(**defaults) # type: ignore


def _load_all_episodes(output_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    episodes = {}
    for shard in sorted(output_dir.glob("shard_*.h5")):
        for episode_id, frames, actions, _attrs in iterate_episodes(shard):
            episodes[episode_id] = (frames, actions)
    return episodes


def test_generation_end_to_end(tmp_path: Path) -> None:
    output_dir, report = generate_dataset(_small_config(tmp_path / "ds"))

    # 4 episodes at 3 per shard -> 2 shards
    shards = sorted(output_dir.glob("shard_*.h5"))
    assert len(shards) == 2

    episodes = _load_all_episodes(output_dir)
    assert len(episodes) == 4
    for frames, actions in episodes.values():
        # frames = T+1, actions = T, with episode_length = 10
        assert frames.shape == (11, 32, 32, 3)
        assert actions.shape == (10,)
        assert frames.dtype == np.uint8

    manifest = load_manifest(output_dir)
    assert manifest.num_episodes == 4
    assert manifest.total_steps == 40
    assert manifest.seed_end == 3
    assert manifest.observation_height == 32

    assert isinstance(report, CoverageReport)
    assert report.total_steps == 40
    assert sum(report.action_counts.values()) == 40
    assert "action distribution" in report.summary()


def test_generation_is_deterministic(tmp_path: Path) -> None:
    _, _ = generate_dataset(_small_config(tmp_path / "a"))
    _, _ = generate_dataset(_small_config(tmp_path / "b"))
    eps_a = _load_all_episodes(tmp_path / "a")
    eps_b = _load_all_episodes(tmp_path / "b")
    assert eps_a.keys() == eps_b.keys()
    for episode_id in eps_a:
        np.testing.assert_array_equal(eps_a[episode_id][0], eps_b[episode_id][0])
        np.testing.assert_array_equal(eps_a[episode_id][1], eps_b[episode_id][1])


def test_parallel_matches_serial(tmp_path: Path) -> None:
    """The scalability seam: adding workers must never change the data."""
    _, _ = generate_dataset(_small_config(tmp_path / "serial", num_workers=1))
    _, _ = generate_dataset(_small_config(tmp_path / "parallel", num_workers=2))
    serial = _load_all_episodes(tmp_path / "serial")
    parallel = _load_all_episodes(tmp_path / "parallel")
    assert serial.keys() == parallel.keys()
    for episode_id in serial:
        np.testing.assert_array_equal(serial[episode_id][0], parallel[episode_id][0])
        np.testing.assert_array_equal(serial[episode_id][1], parallel[episode_id][1])


def test_invalid_generation_config_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _small_config(tmp_path, num_episodes=0)
    with pytest.raises(ValueError):
        _small_config(tmp_path, num_workers=0)
    with pytest.raises(ValueError):
        generate_dataset(_small_config(tmp_path, policy_name="does_not_exist"))

def test_unknown_env_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_dataset(_small_config(tmp_path, env_name="does_not_exist"))


def test_registries_reject_duplicate_registration() -> None:
    from engine.envs.registry import register_env
    from engine.envs.policies.registry import register_policy

    with pytest.raises(ValueError):
        register_env("grid2d", lambda cfg: None)  # type: ignore
    with pytest.raises(ValueError):
        register_policy("random", lambda n, rng: None)  # type: ignore