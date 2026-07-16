# engine

Mamba-based world model engine: tokenized visual states, action-conditioned
sequence prediction, staged toward real-time inference. The Python backend
tier of the continuum project.

## Status

**Phase 0 — data engine.** Environment interface, 2D environment, policies,
data schema, and sharded storage are in place. Generation harness is next.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate    # Git Bash on Windows
pip install -e ".[dev]"
```

## Sanity check

```bash
python -c "import engine; print(engine.__file__)"
pytest
```

## Project layout

- `src/engine/envs/` — the environment contract (`base.py`) and concrete
  implementations. `grid2d/` is the only one that exists; it's the only
  package allowed to know it's 2D.
- `src/engine/envs/policies/` — action-selection policies used to generate
  training data (random for now, scripted/heuristic later).
- `src/engine/data/` — versioned episode schema and sharded HDF5 storage.
- `configs/` — YAML configs for each phase's runs.
- `docs/decisions/` — short dated notes on real forks in the design.
- `tests/` — includes the pipeline's load-bearing smoke tests.
