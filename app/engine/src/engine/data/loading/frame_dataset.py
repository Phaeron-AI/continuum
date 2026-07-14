from __future__ import annotations

import os
from pathlib import Path

import h5py
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from engine.data.loading.frame_index import FrameLocator, build_frame_index
from engine.data.storage.transforms import frame_to_tensor

class FrameDataset(Dataset):
  def __init__(self, dataset_dir: Path | str, target_size: tuple[int, int] = (64, 64))-> None:
    self._dataset_dir = dataset_dir
    self._target_size = target_size
    self._index: list[FrameLocator] = build_frame_index(self._dataset_dir)

    self._handles: dict[int, dict[str, h5py.File]] = {}
  
  def __len__(self)-> int:
    return len(self._index)
  
  def _shard_handle(self, shard_path: str)-> h5py.File:
    pid = os.getpid()
    per_proc = self._handles.setdefault(pid, {})
    handle = per_proc.get(shard_path)
    if handle is None:
      handle = h5py.File(shard_path, "r")
      per_proc[shard_path] = handle
    return handle
  
  def __getitem__(self, index: int) -> Tensor:
    loc = self._index[index]
    handle = self._shard_handle(loc.shard_path)
    group = handle[loc.episode_id]

    assert isinstance(group, h5py.Group)
    frames = group["frames"]
    assert isinstance(frames, h5py.Dataset)
    frame = frames[loc.frame_idx]
    return frame_to_tensor(frame, target_size=self._target_size)
  
  def __getstate__(self) -> dict:
    state = self.__dict__.copy()
    state["_handles"] = {}
    return state
  
def make_frame_dataloader(
  dataset_dir: Path | str,
  batch_size: int = 64,
  shuffle: bool = True,
  num_workers: int = 0,
  target_size: tuple[int, int] = (64, 64),
  pin_memory: bool = False,
  drop_last: bool = False,
) -> DataLoader:
  dataset = FrameDataset(dataset_dir, target_size=target_size)
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=drop_last,
  )