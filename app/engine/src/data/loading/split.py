from __future__ import annotations

import numpy as np
from torch.utils.data import Subset

from data.loading.frame_dataset import FrameDataset


def split_dataset(
  dataset: FrameDataset, 
  eval_fraction: float = 0.1, 
  seed: int = 0
) -> tuple[Subset,Subset]:
  if not 0.0 < eval_fraction < 1.0:
    raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")
  
  n = len(dataset)
  rng = np.random.default_rng(seed)
  perm = rng.permutation(n)
  n_eval = max(1, int(n * eval_fraction))
  eval_idx = perm[:n_eval].tolist()
  train_idx = perm[n_eval:].tolist()
  return Subset(dataset, train_idx), Subset(dataset, eval_idx)
