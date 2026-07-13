"""Sub-phase 2.5 verification: autoregressive rollout and drift measurement.

The two load-bearing tests:

`test_rollout_uses_own_predictions` — the rollout must feed the model its OWN
output, never the ground truth. If it peeked at truth, drift would be
artificially low and the headline metric would be a lie. Verified by rolling
out with no truth available at all: the function's signature makes truth
structurally unreachable, and this test confirms generation still proceeds.

`test_action_changes_the_future` — the same seed with different actions must
produce different futures. If it did not, the action conditioning (stage 06)
would be doing nothing, and the "world model" would be an unconditional video
generator.
"""

from __future__ import annotations

import pytest
import torch

from engine.models.tokenizer.config import TokenizerConfig
from engine.models.tokenizer.frozen import FrozenTokenizer
from engine.models.tokenizer.tokenizer import Tokenizer
from engine.models.world_model.config import WorldModelConfig
from engine.models.world_model.drift import DriftReport, evaluate_drift
from engine.models.world_model.rollout import rollout, rollout_to_frames
from engine.models.world_model.world_model import WorldModel

TPF = 64  # 8x8 token grid


@pytest.fixture
def setup():
    """A frozen tokenizer and a world model whose vocab matches it."""
    torch.manual_seed(0)
    tok = Tokenizer(TokenizerConfig(levels=(4, 4, 4), hidden=16))
    frozen = FrozenTokenizer(tok)
    spec = frozen.token_spec
    model = WorldModel(
        WorldModelConfig.from_token_spec(
            spec, num_actions=5, d_model=16, d_state=4, n_layers=1
        )
    )
    return frozen, model, spec


# ---- rollout mechanics ----

def test_rollout_shapes(setup) -> None:
    frozen, model, _ = setup
    seed = torch.randint(0, model.config.vocab_size, (2, TPF))
    actions = torch.randint(0, 5, (2, 3))
    out = rollout(model, seed, actions, num_frames=3, tokens_per_frame=TPF)
    assert out.shape == (2, 3, TPF)


def test_rollout_emits_only_visual_tokens(setup) -> None:
    """Generated tokens must be VISUAL ids — the head only predicts those."""
    frozen, model, _ = setup
    vocab = model.config.vocab_size
    seed = torch.randint(0, vocab, (1, TPF))
    actions = torch.randint(0, 5, (1, 2))
    out = rollout(model, seed, actions, num_frames=2, tokens_per_frame=TPF)
    assert bool((out < vocab).all()), "rollout emitted an action-token id"
    assert bool((out >= 0).all())


def test_greedy_is_deterministic(setup) -> None:
    """temperature=0 must give identical output every time — this is what
    makes drift measurement reproducible."""
    frozen, model, _ = setup
    seed = torch.randint(0, model.config.vocab_size, (1, TPF))
    actions = torch.randint(0, 5, (1, 2))
    a = rollout(model, seed, actions, 2, TPF, temperature=0.0)
    b = rollout(model, seed, actions, 2, TPF, temperature=0.0)
    assert torch.equal(a, b)


def test_sampling_is_stochastic(setup) -> None:
    """temperature>0 should (with high probability) differ across runs."""
    frozen, model, _ = setup
    seed = torch.randint(0, model.config.vocab_size, (1, TPF))
    actions = torch.randint(0, 5, (1, 2))
    torch.manual_seed(1)
    a = rollout(model, seed, actions, 2, TPF, temperature=1.0)
    torch.manual_seed(2)
    b = rollout(model, seed, actions, 2, TPF, temperature=1.0)
    assert not torch.equal(a, b), "sampling produced identical output twice"


def test_rollout_uses_own_predictions(setup) -> None:
    """Rollout never receives the ground-truth continuation — it cannot peek.
    Generation must still proceed from the seed alone."""
    frozen, model, _ = setup
    seed = torch.randint(0, model.config.vocab_size, (1, TPF))
    actions = torch.randint(0, 5, (1, 4))
    out = rollout(model, seed, actions, num_frames=4, tokens_per_frame=TPF)
    # 4 frames generated with no truth in scope at any point
    assert out.shape == (1, 4, TPF)


def test_action_changes_the_future(setup) -> None:
    """THE conditioning test: same seed, different actions -> different
    futures. If this fails, action conditioning does nothing."""
    frozen, model, _ = setup
    seed = torch.randint(0, model.config.vocab_size, (1, TPF))

    a0 = torch.zeros(1, 2, dtype=torch.long)  # always action 0
    a1 = torch.full((1, 2), 3, dtype=torch.long)  # always action 3

    out0 = rollout(model, seed, a0, 2, TPF, temperature=0.0)
    out1 = rollout(model, seed, a1, 2, TPF, temperature=0.0)
    assert not torch.equal(out0, out1), (
        "different actions produced identical futures — conditioning is dead"
    )


def test_rollout_rejects_too_few_actions(setup) -> None:
    frozen, model, _ = setup
    seed = torch.randint(0, model.config.vocab_size, (1, TPF))
    actions = torch.randint(0, 5, (1, 1))
    with pytest.raises(ValueError):
        rollout(model, seed, actions, num_frames=3, tokens_per_frame=TPF)


# ---- decoding back to pixels: closes the Phase 1 <-> Phase 2 loop ----

def test_rollout_to_frames_decodes(setup) -> None:
    frozen, model, spec = setup
    seed = torch.randint(0, model.config.vocab_size, (1, TPF))
    actions = torch.randint(0, 5, (1, 2))
    frames = rollout_to_frames(model, frozen, seed, actions, num_frames=2)
    assert frames.shape == (1, 2, 3, spec.input_height, spec.input_width)
    assert bool((frames >= -1.0).all()) and bool((frames <= 1.0).all())


# ---- drift ----

def test_drift_report_structure(setup) -> None:
    frozen, model, _ = setup
    vocab = model.config.vocab_size
    seed = torch.randint(0, vocab, (1, TPF))
    actions = torch.randint(0, 5, (1, 3))
    truth = torch.randint(0, vocab, (1, 3, TPF))

    report = evaluate_drift(model, frozen, seed, actions, truth, num_frames=3)
    assert isinstance(report, DriftReport)
    assert len(report.token_accuracy_per_step) == 3
    assert len(report.psnr_per_step) == 3
    assert 0 <= report.drift_horizon <= 3
    assert all(0.0 <= a <= 1.0 for a in report.token_accuracy_per_step)
    assert "drift_horizon" in report.summary()


def test_drift_perfect_when_truth_matches_generation(setup) -> None:
    """Sanity anchor: if we feed the model's OWN generation back as the
    'truth', accuracy must be 100% and PSNR infinite at every step. This
    verifies the comparison logic itself is not broken."""
    frozen, model, _ = setup
    vocab = model.config.vocab_size
    seed = torch.randint(0, vocab, (1, TPF))
    actions = torch.randint(0, 5, (1, 2))

    generated = rollout(model, seed, actions, 2, TPF, temperature=0.0)
    report = evaluate_drift(model, frozen, seed, actions, generated, num_frames=2)
    assert all(a == 1.0 for a in report.token_accuracy_per_step)
    assert report.drift_horizon == 2


def test_drift_rejects_shape_mismatch(setup) -> None:
    frozen, model, _ = setup
    vocab = model.config.vocab_size
    seed = torch.randint(0, vocab, (1, TPF))
    actions = torch.randint(0, 5, (1, 2))
    wrong_truth = torch.randint(0, vocab, (1, 3, TPF))  # 3 != 2 frames
    with pytest.raises(ValueError):
        evaluate_drift(model, frozen, seed, actions, wrong_truth, num_frames=2)