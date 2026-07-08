from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

def frame_to_tensor(frame: np.ndarray, target_size: tuple[int, int] = (64, 64))-> Tensor:
  if frame.ndim != 2:
    raise ValueError(f"Expected HWC frame, got shape: {frame.shape}")
  
  tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

  target_h, target_w = target_size
  if tensor.shape[1:] != (target_h, target_w):
    tensor = F.interpolate(
      tensor.unsqueeze(0),
      size=(target_h, target_w),
      mode="bilinear",
      align_corners=False
    ).squeeze(0)
  
  return tensor * 2.0 - 1.0

def tensor_to_frame(tensor: Tensor)-> np.ndarray:
  tensor = (tensor.detach().clamp(-1.0, 1.0) + 1.0) / 2.0
  hwc = tensor.permute(1, 2, 0).cpu().numpy()
  return (hwc * 255.0).round().astype(np.uint8)