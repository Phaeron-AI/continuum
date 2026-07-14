from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Callable[[dict[str, Any]], Any]] = {}

def dataclass_factory(cls: type, section: str) -> Callable[..., Any]:
  """Returns build(data, overrides=None) -> cls instance.

  `data` is the raw dict (e.g. from YAML); unknown keys are warned about
  and dropped. `overrides` supplies already-constructed values (e.g. a
  nested config object built elsewhere) that bypass the dict entirely.
  """

  valid = {f.name for f in dataclasses.fields(cls)}

  def build(data: dict[str, Any], overrides: dict[str, Any] | None = None) -> Any:
    unknown = [k for k in data if k not in valid]
    if unknown:
      logger.warning(
        "Ignoring unknown key(s) in [%s]: %s. Valid keys: %s. "
        "Check for typos — ignored keys fall back to defaults.",
        section,
        ", ".join(sorted(unknown)),
        ", ".join(sorted(valid)),
      )
    kwargs = {k: v for k, v in data.items() if k in valid}
    if overrides:
      kwargs.update(overrides)
    return cls(**kwargs)

  return build
  
def register_env_config(name: str, factory: Callable[[dict[str, Any]], Any]) -> None:
  if name in _REGISTRY:
    raise ValueError(f"Env config {name!r} is already registered")
  _REGISTRY[name] = factory


def make_env_config(name: str, data: dict[str, Any]) -> Any:
  if name not in _REGISTRY:
    raise ValueError(
      f"Unknown env config: {name!r}. Registered: {sorted(_REGISTRY)}"
    )
  return _REGISTRY[name](data)


def _register_builtins() -> None:
  from envs.grid2d.config import Grid2DConfig

  register_env_config("grid2d", dataclass_factory(Grid2DConfig, section="env.grid2d"))


_register_builtins()