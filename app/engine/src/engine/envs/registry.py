"""Environment registry.

Maps environment names (the strings that appear in configs and dataset
manifests) to factory functions. The harness and any future consumer build
environments through this registry rather than importing concrete classes —
which is what makes "add a 3D env" a one-line registration instead of a
hunt for every construction site.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from engine.envs.base import Env

_REGISTRY: dict[str, Callable[[Any], Env]] = {}


def register_env(name: str, factory: Callable[[Any], Env]) -> None:
    """Register a factory: takes an env-specific config object, returns an
    Env. Registering the same name twice is an error — silent replacement
    is how two modules end up fighting over a name."""
    if name in _REGISTRY:
        raise ValueError(f"Environment {name!r} is already registered")
    _REGISTRY[name] = factory


def make_env(name: str, config: Any) -> Env:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown environment: {name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](config)


def _register_builtins() -> None:
    # Imported here, not at module top, so importing the registry doesn't
    # drag in pygame and every concrete env's dependencies.
    from engine.envs.grid2d.config import Grid2DConfig
    from engine.envs.grid2d.env import Grid2DEnv

    def _make_grid2d(config: Any) -> Env:
        if config is None:
            config = Grid2DConfig()
        if not isinstance(config, Grid2DConfig):
            raise TypeError(f"grid2d expects Grid2DConfig, got {type(config).__name__}")
        return Grid2DEnv(config=config)

    register_env("grid2d", _make_grid2d)


_register_builtins()