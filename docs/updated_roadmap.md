# Continuum: Updated Roadmap

```
continuum/
├── README.md                                    ★ NEW  (real storefront: pitch, architecture, roadmap)
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                               ★ NEW  (ruff on src + fast pytest, Python 3.11)
├── docs/
│   └── phase_02_roadmap.md
└── app/
    ├── .gitignore
    └── engine/
        ├── README.md
        ├── pyproject.toml
        ├── .gitignore
        ├── configs/
        │   ├── phase0_grid2d.yaml
        │   └── phase2_world_model.yaml          ← flip on mamba / film / context_noise here
        ├── scripts/
        │   ├── generate_dataset.py
        │   ├── build_token_cache.py
        │   ├── train_world_model.py
        │   └── evalute_drift.py
        ├── src/
        │   ├── envs/                             (Phase 0 — data engine)
        │   │   ├── base.py, registry.py, config_registry.py
        │   │   ├── grid2d/  (env.py, renderer.py, config.py)
        │   │   └── policies/  (base.py, random_policy.py, registry.py)
        │   ├── data/                             (Phase 0 — storage & sequences)
        │   │   ├── generation/  (harness.py, config_loader.py)
        │   │   ├── storage/     (schema.py, storage.py, transforms.py)
        │   │   ├── loading/      (frame_dataset.py, frame_index.py, split.py)
        │   │   ├── tokens/       (sequence_dataset.py, token_cache.py)
        │   │   └── cli/          (generation.py, token_cache.py)
        │   └── models/
        │       ├── device.py
        │       ├── tokenizer/                    (Phase 1 — frozen FSQ tokenizer)
        │       │   ├── tokenizer.py, frozen.py, spec.py, config.py
        │       │   ├── train.py, evaluate.py, checkpoint.py, cli.py
        │       │   └── modules/  (encoder.py, decoder.py, quantizer.py)
        │       └── world_model/                  (Phase 2 — the Mamba world model)
        │           ├── layers/
        │           │   ├── mixer.py              ← updated: registers "mamba"
        │           │   ├── scan.py               ★ NEW  (#3 parallel associative scan)
        │           │   ├── ssm.py                ← updated: forward uses parallel scan
        │           │   ├── mamba.py              ★ NEW  (#4 causal conv + gated Mamba block)
        │           │   ├── conditioning.py       ★ NEW  (#2 FiLM action conditioning)
        │           │   ├── block.py
        │           │   ├── embedding.py
        │           │   └── __init__.py           ← updated: exports new symbols
        │           ├── model/
        │           │   ├── world_model.py        ← updated: FiLM wiring, context-noise loss
        │           │   ├── config.py             ← updated: d_conv, action_conditioning
        │           │   └── checkpoint.py
        │           ├── training/
        │           │   ├── train.py              ← updated: #1 context_noise_prob
        │           │   └── evaluate.py
        │           ├── inference/
        │           │   ├── rollout.py            ← updated: top-k / top-p sampling
        │           │   └── drift.py
        │           ├── server/  (app.py, schemas.py, registry.py)
        │           ├── cli/     (train_cli.py, drift_cli.py)
        │           └── session.py
        └── tests/
            ├── test_scan.py                      ★ NEW  (#3 scan parity + causality)
            ├── test_mamba.py                     ★ NEW  (#4 conv causality, step/forward parity)
            ├── test_conditioning.py              ★ NEW  (#2 FiLM identity-at-init, forward-fill)
            ├── test_noise_aug.py                 ★ NEW  (#1 targets never corrupted)
            └── … (existing: test_ssm, test_world_model, test_rollout, test_wm_train, …)
```