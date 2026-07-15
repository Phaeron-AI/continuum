"""Sub-phase 2.2 verification: the shared token embedding.

Two things matter here. First, that visual and action tokens genuinely share
one embedding space (stage 06's whole point — it is what lets the action ride
the same recurrence). Second, `is_action_token`, which the training loss uses
to score only visual positions (stage 08: the model predicts frames, not
actions). A wrong mask would train the model to predict actions, silently
changing the objective.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from data.generation.harness import GenerationConfig, generate_dataset
from data.tokens.sequence_dataset import SequenceDataset
from data.tokens.token_cache import build_token_cache
from envs.grid2d.config import Grid2DConfig
from models.tokenizer.config import TokenizerConfig
from models.tokenizer.frozen import FrozenTokenizer
from models.tokenizer.tokenizer import Tokenizer
from models.world_model.layers.embedding import TokenEmbedding
from models.world_model.layers.ssm import SelectiveSSM


def test_embedding_shape_and_total_vocab() -> None:
    emb = TokenEmbedding(vocab_size=100, num_actions=5, d_model=16)
    assert emb.total_vocab == 105
    ids = torch.randint(0, 105, (2, 7))
    out = emb(ids)
    assert out.shape == (2, 7, 16)
    assert out.dtype == torch.float32


def test_rejects_out_of_range_ids() -> None:
    emb = TokenEmbedding(vocab_size=10, num_actions=3, d_model=8)
    with pytest.raises(ValueError):
        emb(torch.tensor([[0, 13]]))  # 13 >= total_vocab (13)
    with pytest.raises(ValueError):
        emb(torch.tensor([[0, -1]]))


def test_rejects_bad_dims() -> None:
    emb = TokenEmbedding(vocab_size=10, num_actions=3, d_model=8)
    with pytest.raises(ValueError):
        emb(torch.zeros(2, 3, 4, dtype=torch.long))  # not (B, L)


def test_visual_and_action_share_one_table() -> None:
    """A visual id and an action id must both index the SAME table — that is
    the shared space stage 06 requires."""
    emb = TokenEmbedding(vocab_size=10, num_actions=3, d_model=8)
    assert emb.embed.num_embeddings == 13
    # every id in range embeds without error, visual and action alike
    ids = torch.arange(13).unsqueeze(0)
    assert emb(ids).shape == (1, 13, 8)


def test_action_mask() -> None:
    """ids >= vocab_size are actions; below are visual."""
    emb = TokenEmbedding(vocab_size=10, num_actions=3, d_model=8)
    ids = torch.tensor([[0, 5, 9, 10, 11, 12]])
    mask = emb.is_action_token(ids)
    expected = torch.tensor([[False, False, False, True, True, True]])
    assert torch.equal(mask, expected)


# ---- integration: real sequence -> embedding -> SSM ----

@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    ds, _ = generate_dataset(
        GenerationConfig(
            dataset_name="emb_ds",
            output_dir=str(tmp_path / "ds"),
            num_episodes=2,
            episodes_per_shard=2,
            num_workers=1,
            env=Grid2DConfig(canvas_size=32, episode_length=5, num_obstacles=2),
        )
    )
    tok = Tokenizer(TokenizerConfig(levels=(4, 4, 4), hidden=16))
    return build_token_cache(ds, FrozenTokenizer(tok), tmp_path / "cache", device="cpu")


def test_real_sequence_embeds_and_mixes(cache_dir: Path) -> None:
    """End-to-end 2.0 -> 2.2 -> 2.1: a real interleaved window embeds cleanly
    and flows through the SSM with shapes intact."""
    ds = SequenceDataset(cache_dir, context_frames=3, num_actions=5)
    seq = ds[0].unsqueeze(0)  # (1, L)

    emb = TokenEmbedding(ds.vocab_size, num_actions=5, d_model=16)
    x = emb(seq)  # (1, L, 16)
    assert x.shape == (1, ds.seq_len, 16)

    mixer = SelectiveSSM(d_model=16, d_state=4)
    y = mixer(x)
    assert y.shape == x.shape


def test_action_mask_finds_actions_in_real_sequence(cache_dir: Path) -> None:
    """On a real interleaved window, the mask must flag exactly the action
    positions — every (tokens_per_frame)th slot, and no others."""
    ds = SequenceDataset(cache_dir, context_frames=3, num_actions=5)
    seq = ds[0].unsqueeze(0)
    emb = TokenEmbedding(ds.vocab_size, num_actions=5, d_model=16)
    mask = emb.is_action_token(seq)[0]

    # 3 frames -> 2 action tokens, at positions 64 and 129
    assert int(mask.sum()) == 2
    action_positions = mask.nonzero().flatten().tolist()
    assert action_positions == [64, 129]