"""Sub-phase 0.5 verification: YAML config loading, unknown-key handling,
validation propagation, and the CLI entrypoint end-to-end."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from engine.data.config_loader import load_generation_config
from engine.data.harness import GenerationConfig
from engine.envs.grid2d.config import Grid2DConfig


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_loads_valid_config(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "c.yaml",
        """
dataset_name: test_ds
output_dir: out/test
num_episodes: 12
num_workers: 2
env_name: grid2d
policy_name: random
env:
  canvas_size: 64
  episode_length: 20
""",
    )
    config = load_generation_config(cfg_path)
    assert isinstance(config, GenerationConfig)
    assert config.num_episodes == 12
    assert config.num_workers == 2
    assert isinstance(config.env, Grid2DConfig)
    assert config.env.canvas_size == 64
    assert config.env.episode_length == 20


def test_unknown_top_level_key_warns_but_continues(tmp_path: Path, caplog) -> None:
    cfg_path = _write(
        tmp_path / "c.yaml",
        """
dataset_name: test_ds
output_dir: out/test
num_episodes: 5
num_epsiodes: 999   # typo — must not silently override the real key
env:
  canvas_size: 32
""",
    )
    with caplog.at_level(logging.WARNING):
        config = load_generation_config(cfg_path)
    # The real key wins; the typo is ignored.
    assert config.num_episodes == 5
    # And the warning names the offending key loudly.
    assert "num_epsiodes" in caplog.text
    assert "Valid keys" in caplog.text


def test_unknown_env_key_warns(tmp_path: Path, caplog) -> None:
    cfg_path = _write(
        tmp_path / "c.yaml",
        """
dataset_name: test_ds
output_dir: out/test
num_episodes: 3
env:
  canvas_size: 32
  cavnas_szie: 999   # typo in the env section
""",
    )
    with caplog.at_level(logging.WARNING):
        config = load_generation_config(cfg_path)
    assert config.env.canvas_size == 32
    assert "cavnas_szie" in caplog.text


def test_invalid_value_propagates_as_error(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "c.yaml",
        """
dataset_name: test_ds
output_dir: out/test
num_episodes: 0   # invalid — dataclass __post_init__ must reject
env:
  canvas_size: 32
""",
    )
    with pytest.raises(ValueError):
        load_generation_config(cfg_path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_generation_config(tmp_path / "does_not_exist.yaml")


def test_unknown_env_name_raises(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "c.yaml",
        """
dataset_name: test_ds
output_dir: out/test
num_episodes: 3
env_name: grid9000
env: {}
""",
    )
    with pytest.raises(ValueError):
        load_generation_config(cfg_path)


def test_cli_end_to_end(tmp_path: Path) -> None:
    """The CLI runs a full generation with overrides and produces a dataset."""
    from engine.data.cli import main

    cfg_path = _write(
        tmp_path / "c.yaml",
        """
dataset_name: cli_ds
output_dir: PLACEHOLDER
num_episodes: 2
num_workers: 1
env:
  canvas_size: 32
  episode_length: 8
""",
    )
    out_dir = tmp_path / "cli_out"
    rc = main(
        [
            "--config",
            str(cfg_path),
            "--output-dir",
            str(out_dir),
            "--episodes",
            "3",
        ]
    )
    assert rc == 0
    assert (out_dir / "manifest.json").exists()
    assert len(list(out_dir.glob("shard_*.h5"))) >= 1


def test_cli_bad_config_returns_error_code(tmp_path: Path) -> None:
    from engine.data.cli import main

    rc = main(["--config", str(tmp_path / "nope.yaml")])
    assert rc == 1