from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

def associative_scan(a: Tensor, u: Tensor)-> Tensor:
  if a.shape != u.shape:
    raise ValueError(f"a and u must match; got: {a.shape} & {u.shape}")
  
  length = a.shape[1]
  a_acc = a
  b_acc = u
  shift = 1

  trailing = a.dim() - 2  # dims after the length axis
  while shift < length:
    pad = [0, 0] * trailing + [shift, 0]  # pad `shift` on the left of dim=1
    a_prev = F.pad(a_acc[:, : length - shift], pad, value=1.0)  # identity coeff
    b_prev = F.pad(b_acc[:, : length - shift], pad, value=0.0)  # identity bias
    b_acc = a_acc * b_prev + b_acc
    a_acc = a_acc * a_prev
    shift *= 2

  return b_acc