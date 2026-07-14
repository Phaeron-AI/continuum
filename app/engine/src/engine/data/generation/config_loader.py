from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from engine.data.generation.harness import GenerationConfig
from engine.envs.config_registry import dataclass_factory, make_env_config

def load_generation_config(path: Path | str) -> GenerationConfig:
  """Loads and validates a generation config from a YAML file.

  Raises FileNotFoundError if the path is missing, yaml.YAMLError on
  malformed YAML, and ValueError (from dataclass __post_init__) on invalid
  values.
  """
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(f"Config file not found: {path}")

  raw = yaml.safe_load(path.read_text())
  if not isinstance(raw, dict):
    raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")

  env_name = raw.get("env_name", "grid2d")
  env_section = raw.get("env", {}) or {}
  if not isinstance(env_section, dict):
    raise ValueError(f"[env] must be a mapping, got {type(env_section).__name__}")
  env_config = make_env_config(env_name, env_section)

  top_level: dict[str, Any] = {k: v for k, v in raw.items() if k != "env"}
  build = dataclass_factory(GenerationConfig, section="root")
  return build(top_level, overrides={"env": env_config})