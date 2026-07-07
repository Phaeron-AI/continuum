from __future__ import annotations

from collections.abc import Callable
import numpy as np

from engine.envs.policies.base import Policy

PolicyFactory = Callable[[int, np.random.Generator], Policy]
_REGISTRY: dict[str, PolicyFactory] = {}

def register_policy(name: str, factory: PolicyFactory)-> None:
  if name in _REGISTRY:
    raise ValueError(f"Policy {name!r} is already registered")
  _REGISTRY[name] = factory

def make_policy(name: str, num_actions: int, rng: np.random.Generator)-> Policy:
  if name not in _REGISTRY:
    raise ValueError(f"Unknown policy: {name!r}. Registered: {sorted(_REGISTRY)}")
  return _REGISTRY[name](num_actions, rng)

def _register_bullitins()-> None:
  from engine.envs.policies.random_policy import RandomPolicy

  register_policy(
    "random", lambda num_actions, rng: RandomPolicy(num_actions=num_actions, rng=rng)
  )

_register_bullitins()