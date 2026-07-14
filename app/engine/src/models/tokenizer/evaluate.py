from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.device import resolve_device
from models.tokenizer.tokenizer import Tokenizer

@dataclass
class EvalReport:
  recon_mse: float
  psnr_db: float
  codebook_usage: float
  unique_tokens: int
  vocab_size: int
  num_frames: int
  bits_per_frame: float

  def summary(self) -> str:
    return (
      f"frames={self.num_frames} | recon_mse={self.recon_mse:.6f} | "
      f"psnr={self.psnr_db:.2f} dB | "
      f"codebook_usage={self.codebook_usage:.1%} "
      f"({self.unique_tokens}/{self.vocab_size}) | "
      f"capacity={self.bits_per_frame:.0f} bits/frame"
    )
  

def _psnr_from_mse(mse: float, data_range: float = 2.0) -> float:
  """PSNR for data in [-1, 1] (range 2.0). Infinite for a perfect match."""
  if mse <= 0:
      return float("inf")
  return 10.0 * math.log10((data_range**2) / mse)

@torch.no_grad()
def evaluate_tokenizer(
  tokenizer: Tokenizer,
  loader: DataLoader,
  device: str = "auto",
  max_batches: int | None = None
)-> EvalReport:
  resolved = resolve_device(device)
  tokenizer.to(resolved)
  tokenizer.eval()

  spec = tokenizer.token_spec
  total_sq_err = 0.0
  total_elems = 0
  total_frames = 0
  seen_tokens: set[int] = set()

  for i, batch in enumerate(loader):
    if max_batches is not None and i >= max_batches:
      break
    x = batch.to(resolved)
    recon, indices = tokenizer(x)

    total_sq_err += float(F.mse_loss(recon, x, reduction="sum").cpu())
    total_elems += x.numel()
    total_frames += x.shape[0]
    seen_tokens.update(indices.flatten().cpu().tolist())
  
  mse = total_sq_err / max(total_elems, 1)
  unique = len(seen_tokens)
  return EvalReport(
    recon_mse=mse,
    psnr_db=_psnr_from_mse(mse),
    codebook_usage=unique / spec.vocab_size,
    unique_tokens=unique,
    vocab_size=spec.vocab_size,
    num_frames=total_frames,
    bits_per_frame=spec.bits_per_frame,
  )