"""Sub-phase 2.0 verification: the SequenceMixer interface, the token cache,
and the SequenceDataset interleaving contract.

The load-bearing test is the interleaving: the flattened window must be
exactly [frame tokens][action][frame tokens]... with visual tokens in
[0, vocab) and action tokens offset by vocab. A bug here silently corrupts
every training example.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.data.harness import GenerationConfig, generate_dataset
from engine.data.sequence_dataset import SequenceDataset
from engine.data.token_cache import build_token_cache, load_cache_manifest
from engine.envs.grid2d.config import Grid2DConfig
from engine.models.tokenizer.config import TokenizerConfig
from engine.models.tokenizer.frozen import FrozenTokenizer
from engine.models.tokenizer.tokenizer import Tokenizer
from engine.models.world_model.mixer import IdentityMixer, make_mixer


# ---- mixer interface ----

def test_identity_mixer_shape_and_passthrough() -> None:
    mix = IdentityMixer(d_model=16)
    x = torch.randn(2, 10, 16)
    out = mix(x)
    assert out.shape == x.shape
    assert torch.equal(out, x)


def test_mixer_rejects_wrong_shape() -> None:
    mix = IdentityMixer(d_model=16)
    with pytest.raises(ValueError):
        mix(torch.randn(2, 10))  # not 3D
    with pytest.raises(ValueError):
        mix(torch.randn(2, 10, 8))  # wrong d_model


def test_make_mixer_factory() -> None:
    assert isinstance(make_mixer("identity", 32), IdentityMixer)
    with pytest.raises(ValueError):
        make_mixer("mamba", 32)  # not available yet


# ---- fixtures: real dataset + frozen tokenizer + cache ----

@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    ds, _ = generate_dataset(
        GenerationConfig(
            dataset_name="seq_ds",
            output_dir=str(tmp_path / "ds"),
            num_episodes=3,
            episodes_per_shard=2,
            num_workers=1,
            env=Grid2DConfig(canvas_size=32, episode_length=6, num_obstacles=2),
        )
    )
    tok = Tokenizer(TokenizerConfig(levels=(4, 4, 4), hidden=16))
    frozen = FrozenTokenizer(tok)
    return build_token_cache(ds, frozen, tmp_path / "cache", device="cpu")


def test_cache_manifest_records_version(cache_dir: Path) -> None:
    m = load_cache_manifest(cache_dir)
    assert m["tokenizer_version"] == "v1"
    assert m["vocab_size"] == 4 * 4 * 4
    assert m["grid_height"] == 8 and m["grid_width"] == 8
    assert m["num_episodes"] == 3


def test_sequence_dataset_window_length(cache_dir: Path) -> None:
    ds = SequenceDataset(cache_dir, context_frames=4, num_actions=5)
    # 4 frames * 64 tokens + 3 action tokens = 259
    assert ds.seq_len == 4 * 64 + 3
    sample = ds[0]
    assert sample.shape == (259,)
    assert sample.dtype == torch.long


def test_interleaving_contract(cache_dir: Path) -> None:
    """THE load-bearing test: verify the exact interleaved structure — visual
    tokens in [0, vocab), action tokens at vocab + a, in the right positions."""
    ds = SequenceDataset(cache_dir, context_frames=3, num_actions=5)
    vocab = ds.vocab_size
    seq = ds[0].tolist()
    tpf = 64  # tokens per frame

    # positions 0..63 = frame0 visual, pos 64 = action, 65..128 = frame1, etc.
    for fi in range(3):
        base = fi * (tpf + 1)
        frame_tokens = seq[base : base + tpf]
        assert all(0 <= t < vocab for t in frame_tokens), f"frame {fi} tokens out of range"
        if fi < 2:
            action_tok = seq[base + tpf]
            assert vocab <= action_tok < vocab + 5, "action token not in action range"


def test_total_vocab_spans_both(cache_dir: Path) -> None:
    ds = SequenceDataset(cache_dir, context_frames=2, num_actions=5)
    assert ds.total_vocab == ds.vocab_size + 5


def test_version_mismatch_rejected(cache_dir: Path) -> None:
    with pytest.raises(ValueError):
        SequenceDataset(cache_dir, tokenizer_version="v999")


def test_dataloader_multiworker(cache_dir: Path) -> None:
    from torch.utils.data import DataLoader

    ds = SequenceDataset(cache_dir, context_frames=3)
    loader = DataLoader(ds, batch_size=4, num_workers=2, shuffle=False)
    total = sum(b.shape[0] for b in loader)
    assert total == len(ds)