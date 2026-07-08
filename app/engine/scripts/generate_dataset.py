"""Executable entrypoint for dataset generation.

Thin shim — all logic lives in engine.data.cli so it stays importable and
testable. Run:
    python scripts/generate_dataset.py --config configs/phase0_grid2d.yaml
"""

import sys

from engine.data.cli import main

if __name__ == "__main__":
  sys.exit(main())