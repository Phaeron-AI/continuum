from __future__ import annotations

import os
from pathlib import Path

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset

from engine.data.tokens.token_cache import load_cache_manifest


class SequenceDataset(Dataset):
  def __init__(
    self,
    cache_dir: Path | str,
    context_frames: int = 4,
    num_actions: int = 5,
    tokenizer_version: str | None = None,
  ) -> None:
    self._cache_dir = Path(cache_dir)
    self._manifest = load_cache_manifest(self._cache_dir)

    # Version guard: refuse to train on tokens from a different tokenizer.
    if (
      tokenizer_version is not None
      and self._manifest["tokenizer_version"] != tokenizer_version
    ):
      raise ValueError(
        f"token cache built with tokenizer_version "
        f"{self._manifest['tokenizer_version']!r}, but "
        f"{tokenizer_version!r} was requested — rebuild the cache"
      )

    self._h = self._manifest["grid_height"]
    self._w = self._manifest["grid_width"]
    self._vocab_size = self._manifest["vocab_size"]
    self._tokens_per_frame = self._h * self._w
    self._context_frames = context_frames
    self._num_actions = num_actions

    # Build a flat index of (episode_id, start_frame) windows. Metadata
    # only — no token data loaded here.
    self._index: list[tuple[str, int]] = []
    self._handles: dict[int, h5py.File] = {}
    cache_path = self._cache_dir / "tokens.h5"
    with h5py.File(cache_path, "r") as f:
      for episode_id in f.keys():
        num_frames = f[episode_id]["tokens"].shape[0] # type: ignore
        last_start = num_frames - context_frames
        for start in range(last_start + 1):
          self._index.append((episode_id, start))
    self._cache_path = str(cache_path)

  @property
  def vocab_size(self) -> int:
    return self._vocab_size

  @property
  def total_vocab(self) -> int:
    return self._vocab_size + self._num_actions

  @property
  def seq_len(self) -> int:
    return self._context_frames * self._tokens_per_frame + (self._context_frames - 1)

  def __len__(self) -> int:
    return len(self._index)

  def _handle(self) -> h5py.File:
    pid = os.getpid()
    h = self._handles.get(pid)
    if h is None:
      h = h5py.File(self._cache_path, "r")
      self._handles[pid] = h
    return h

  def __getitem__(self, idx: int) -> Tensor:
    episode_id, start = self._index[idx]
    grp = self._handle()[episode_id]
    tokens = grp["tokens"][start : start + self._context_frames]  # (F, h, w) # type: ignore
    actions = grp["actions"][start : start + self._context_frames - 1]  # (F-1,)  # type: ignore

    seq: list[int] = []
    for fi in range(self._context_frames):
      flat = tokens[fi].reshape(-1)  # row-major flatten (stage 01 contract)  # type: ignore
      seq.extend(int(t) for t in flat)
      if fi < self._context_frames - 1:
        # action token offset into the shared embedding space
        seq.append(self._vocab_size + int(actions[fi])) # type: ignore
    return torch.tensor(seq, dtype=torch.long)

  def __getstate__(self) -> dict:
    state = self.__dict__.copy()
    state["_handles"] = {}  # never pickle open handles across processes
    return state