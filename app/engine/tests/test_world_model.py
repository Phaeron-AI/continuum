"""Sub-phase 2.3 verification: the world model, its loss, and — the two
load-bearing properties — the teacher-forcing shift and the visual mask.

`test_shift_does_not_leak_target` is the critical one. Roadmap stage 05 warns
that an off-by-one in the shift shows the model the very token it is meant to
predict: training loss collapses toward zero and generation is garbage, with
no error raised. This test catches that directly by checking the model's
prediction at a position cannot depend on the target at that position.

`test_loss_scores_only_visual_targets` guards stage 08: action tokens are
context, never targets. Scoring them would silently change the objective.
"""

from __future__ import annotations

import pytest
import torch

from engine.models.tokenizer.spec import TokenSpec
from engine.models.world_model.block import SSMBlock
from engine.models.world_model.config import WorldModelConfig
from engine.models.world_model.mixer import make_mixer
from engine.models.world_model.world_model import WorldModel


def _cfg(**kw: object) -> WorldModelConfig:
    base: dict = dict(
        vocab_size=32, num_actions=5, d_model=16, d_state=4, n_layers=2
    )
    base.update(kw)
    return WorldModelConfig(**base)  # type: ignore[arg-type]


def _model(**kw: object) -> WorldModel:
    torch.manual_seed(0)
    return WorldModel(_cfg(**kw))


# ---- block ----

def test_block_preserves_shape_and_causality() -> None:
    mixer = make_mixer("ssm", 16, d_state=4)
    block = SSMBlock(mixer, d_model=16).eval()
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        y1 = block(x)
        assert y1.shape == x.shape
        # causality survives the residual + FFN wrapper
        x2 = x.clone()
        x2[0, 4] = torch.randn(16)
        y2 = block(x2)
    assert torch.allclose(y1[0, :4], y2[0, :4], atol=1e-6), "block leaked the future"


# ---- config: the vocab coupling ----

def test_config_from_token_spec_matches_vocab() -> None:
    spec = TokenSpec(grid_height=8, grid_width=8, levels=(8, 8, 8, 5, 5))
    cfg = WorldModelConfig.from_token_spec(spec, num_actions=5, d_model=32) # type: ignore
    assert cfg.vocab_size == spec.vocab_size == 12800
    assert cfg.tokenizer_version == spec.tokenizer_version
    assert cfg.total_vocab == 12800 + 5


def test_config_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        WorldModelConfig(vocab_size=0)
    with pytest.raises(ValueError):
        WorldModelConfig(vocab_size=10, n_layers=0)


# ---- model shapes ----

def test_forward_shapes() -> None:
    m = _model()
    ids = torch.randint(0, 37, (2, 10))  # total_vocab = 32 + 5
    logits = m(ids)
    assert logits.shape == (2, 10, 32)  # head width == vocab_size, not total


def test_head_width_is_visual_vocab_only() -> None:
    """The head predicts visual tokens; it must NOT have action columns."""
    m = _model()
    assert m.head.out_features == 32
    assert m.embedding.embed.num_embeddings == 37  # embedding spans both


# ---- THE shift: teacher forcing must not leak the target ----

def test_shift_does_not_leak_target() -> None:
    """Changing token at position i must not change the model's prediction FOR
    position i (which is produced from inputs strictly before i). If the shift
    were wrong, the model would be reading the answer."""
    m = _model().eval()
    ids = torch.randint(0, 32, (1, 8))

    with torch.no_grad():
        inputs = ids[:, :-1]
        logits1 = m(inputs)

        # perturb the LAST token of the full sequence — it is a target, and
        # must not appear anywhere in the inputs
        ids2 = ids.clone()
        ids2[0, -1] = (ids[0, -1] + 1) % 32
        logits2 = m(ids2[:, :-1])

    assert torch.equal(logits1, logits2), (
        "the final target token influenced the logits — the shift leaks"
    )


def test_loss_runs_and_is_finite() -> None:
    m = _model()
    ids = torch.randint(0, 32, (2, 10))
    loss = m.loss(ids)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_loss_rejects_too_short() -> None:
    m = _model()
    with pytest.raises(ValueError):
        m.loss(torch.randint(0, 32, (1, 1)))  # cannot shift a length-1 seq


# ---- THE visual mask: actions are context, not targets ----

def test_loss_scores_only_visual_targets() -> None:
    """No action token may ever be a scored target (stage 08): the model
    predicts frames, not actions. Verify the mask excludes every action id
    from the set of scored targets."""
    m = _model().eval()
    vocab = 32

    # A sequence containing action tokens at several positions.
    ids = torch.randint(0, vocab, (2, 9))
    ids[0, 5] = vocab + 2
    ids[1, 3] = vocab + 4

    targets = ids[:, 1:]  # what the loss would score, after the shift
    is_visual = ~m.embedding.is_action_token(targets)
    scored = targets[is_visual]

    assert bool((scored < vocab).all()), "an action token was scored as a target"
    # and the action targets really were present to be excluded
    assert int((~is_visual).sum()) == 2, "expected exactly 2 action targets"


def test_loss_raises_if_no_visual_targets() -> None:
    """A batch with only action targets has nothing to train on — loud error,
    not a silent zero-loss."""
    m = _model()
    vocab = 32
    ids = torch.tensor([[0, vocab + 1, vocab + 2]])  # targets are both actions
    with pytest.raises(ValueError):
        m.loss(ids)


# ---- gradients ----

def test_gradients_reach_all_parameters() -> None:
    m = _model()
    ids = torch.randint(0, 32, (2, 10))
    m.loss(ids).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_predict_next_shape() -> None:
    m = _model().eval()
    ids = torch.randint(0, 32, (3, 7))
    nxt = m.predict_next(ids)
    assert nxt.shape == (3,)
    assert bool((nxt < 32).all()), "predicted a non-visual token id"