from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

@dataclass(frozen=True)
class StepRecord:
  episode_id: str
  step_idx: int
  frame: np.ndarray
  action: int
  done: bool

@dataclass(frozen=True)
class EpisodeMetaData:
  episode_id: str
  seed: int
  num_steps: int
  env_name: str
  env_version: str
  action_space_version: str
  policy_name: str

@dataclass
class DatasetManifest:
  dataset_name: str
  env_name: str
  env_version: str
  action_space_version: str
  policy_name: str
  observation_height: int
  observation_width: int
  observation_channels: int
  observation_dtype: str
  num_episodes: int
  total_steps: int
  seed_start: int
  seed_end: int
  created_at: str = field(
    default_factory=lambda: datetime.now(timezone.utc).isoformat()
  )
  git_commit: Optional[str] = None
  notes: str = ""

  def to_dict(self)-> dict:
    return {
      "dataset_name": self.dataset_name,
      "env_name": self.env_name,
      "env_version": self.env_version,
      "action_space_version": self.action_space_version,
      "policy_name": self.policy_name,
      "observation_spec": {
        "height": self.observation_height,
        "width": self.observation_width,
        "channels": self.observation_channels,
        "dtype": self.observation_dtype,
      },
      "num_episodes": self.num_episodes,
      "total_steps": self.total_steps,
      "seed_range": [self.seed_start, self.seed_end],
      "created_at": self.created_at,
      "git_commit": self.git_commit,
      "notes": self.notes,
    }
  
  @classmethod
  def from_dict(cls, data: dict)-> "DatasetManifest":
    obs = data["observation_spec"]
    seed_start, seed_end = data["seed_range"]
    return cls(
      dataset_name=data["dataset_name"],
      env_name=data["env_name"],
      env_version=data["env_version"],
      action_space_version=data["action_space_version"],
      policy_name=data["policy_name"],
      observation_height=obs["height"],
      observation_width=obs["width"],
      observation_channels=obs["channels"],
      observation_dtype=obs["dtype"],
      num_episodes=data["num_episodes"],
      total_steps=data["total_steps"],
      seed_start=seed_start,
      seed_end=seed_end,
      created_at=data["created_at"],
      git_commit=data.get("git_commit"),
      notes=data.get("notes", ""),
    )