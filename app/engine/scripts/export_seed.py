"""Export a short seed clip from a Phase 0 dataset as PNG frames.

The interactive server opens a session from one or more seed PNGs (earliest
first) and warms its recurrent state on them. This dumps a few consecutive
frames from one episode so you have something to start the player with.

    python scripts/export_seed.py --dataset data/raw/phase0_grid2d_v2 --count 4

Frames are written full-resolution; the server resizes them to the tokenizer's
input size, so no normalisation is needed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from data.loading.frame_index import build_frame_index


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description="Export seed PNG frames from a Phase 0 dataset.")
  p.add_argument("--dataset", required=True, help="Phase 0 dataset directory (shards).")
  p.add_argument("--count", type=int, default=4, help="Number of consecutive frames.")
  p.add_argument("--start", type=int, default=0, help="Frame offset within the episode.")
  p.add_argument("--episode", default=None, help="Episode id (default: the first one).")
  p.add_argument("--out", default="seeds", help="Output directory.")
  args = p.parse_args(argv)

  locators = build_frame_index(args.dataset)
  episode = args.episode or locators[0].episode_id
  frames = sorted(
    (loc for loc in locators if loc.episode_id == episode),
    key=lambda loc: loc.frame_idx,
  )
  chosen = frames[args.start : args.start + args.count]
  if not chosen:
    raise SystemExit(f"no frames for episode {episode!r} at offset {args.start}")

  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  for i, loc in enumerate(chosen):
    with h5py.File(loc.shard_path, "r") as f:
      group = f[loc.episode_id]
      assert isinstance(group, h5py.Group)
      dataset = group["frames"]
      assert isinstance(dataset, h5py.Dataset)
      frame = np.asarray(dataset[loc.frame_idx], dtype=np.uint8)
    Image.fromarray(frame, mode="RGB").save(out_dir / f"seed_{i:02d}.png")

  print(f"wrote {len(chosen)} seed frame(s) to {out_dir}/ (episode {episode})")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())