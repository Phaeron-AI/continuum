"""Sub-phase 2.4 verification: the world-model training loop, checkpointing,
the tokenizer version guard, and evaluation.

The Phase-2-specific load-bearing test is `test_version_guard_rejects_mismatch`:
a world model's token ids only mean anything relative to the tokenizer it was
trained against. Loading it against a different tokenizer must be a loud error,
never a silent one — the same integer would decode to a different image patch.

`test_loss_decreases` proves the loop actually trains: cross-entropy must fall
from ~ln(vocab) (random guessing) as the model learns.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from models.world_model import (
    build_world_model_from_checkpoint,
    load_checkpoint,
    WorldModelConfig,
    WMEvalReport,
    evaluate_world_model,
    TrainConfig, train_world_model,
    WorldModel
)

VOCAB = 32
ACTIONS = 5


def _model(version: str = "v1") -> WorldModel:
    torch.manual_seed(0)
    return WorldModel(
        WorldModelConfig(
            vocab_size=VOCAB,
            num_actions=ACTIONS,
            d_model=16,
            d_state=4,
            n_layers=2,
            tokenizer_version=version,
        )
    )


def _loader(n_batches: int = 4, batch: int = 2, length: int = 12):
    """Tiny in-memory loader of a FIXED interleaved-ish batch, so the loss has
    something learnable and tests are deterministic."""
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (batch, length))
    ids[:, 5] = VOCAB + 1  # an action token in the middle
    return [ids for _ in range(n_batches)]


# ---- training loop ----

def test_short_training_runs_and_checkpoints(tmp_path: Path) -> None:
    m = _model()
    cfg = TrainConfig(
        max_steps=5, device="cpu", amp=False,
        log_every=100, checkpoint_every=100,
        checkpoint_dir=str(tmp_path / "ckpts"),
    )
    state = train_world_model(m, _loader(), cfg)  # type: ignore
    assert state.step == 5
    assert len(list((tmp_path / "ckpts").glob("*.pt"))) == 1


def test_checkpoint_captures_full_state(tmp_path: Path) -> None:
    m = _model()
    cfg = TrainConfig(
        max_steps=3, device="cpu", amp=False, checkpoint_dir=str(tmp_path / "c")
    )
    train_world_model(m, _loader(), cfg)  # type: ignore
    ckpt = load_checkpoint(next((tmp_path / "c").glob("*.pt")))
    for key in (
        "step", "model_state", "optimizer_state", "scaler_state",
        "config", "tokenizer_version",
    ):
        assert key in ckpt
    assert ckpt["step"] == 3
    assert ckpt["tokenizer_version"] == "v1"


def test_resume_continues_from_saved_step(tmp_path: Path) -> None:
    m = _model()
    cfg1 = TrainConfig(
        max_steps=3, device="cpu", amp=False, checkpoint_dir=str(tmp_path / "c")
    )
    train_world_model(m, _loader(), cfg1)  # type: ignore 

    ckpt_path = next((tmp_path / "c").glob("*.pt"))
    resumed, ckpt = build_world_model_from_checkpoint(ckpt_path)
    for p_new, p_old in zip(resumed.parameters(), m.parameters(), strict=True):
        assert torch.allclose(p_new, p_old)

    cfg2 = TrainConfig(
        max_steps=5, device="cpu", amp=False, checkpoint_dir=str(tmp_path / "c2")
    )
    state = train_world_model(resumed, _loader(), cfg2, resume_state=ckpt)  # type: ignore
    assert state.step == 5


# ---- THE version guard ----

def test_version_guard_rejects_mismatch(tmp_path: Path) -> None:
    """A world model trained against tokenizer v1 must refuse to load against
    v2 — the token ids do not mean the same thing."""
    m = _model(version="v1")
    cfg = TrainConfig(
        max_steps=2, device="cpu", amp=False, checkpoint_dir=str(tmp_path / "c")
    )
    train_world_model(m, _loader(), cfg)  # type: ignore
    ckpt_path = next((tmp_path / "c").glob("*.pt"))

    # matching version: fine
    build_world_model_from_checkpoint(ckpt_path, expect_tokenizer_version="v1")

    # mismatched version: loud error
    with pytest.raises(ValueError, match="tokenizer_version"):
        build_world_model_from_checkpoint(ckpt_path, expect_tokenizer_version="v2")


# ---- evaluation ----

def test_eval_report_metrics() -> None:
    m = _model()
    report = evaluate_world_model(m, _loader(n_batches=2), device="cpu")  # type: ignore
    assert isinstance(report, WMEvalReport)
    assert report.vocab_size == VOCAB
    assert report.num_visual_targets > 0
    assert 0.0 <= report.next_token_accuracy <= 1.0
    assert report.perplexity > 1.0
    assert "perplexity" in report.summary()


def test_untrained_perplexity_near_vocab() -> None:
    """An untrained model should be near random: ce ~ ln(vocab), so
    perplexity ~ vocab. This is the sanity anchor that the metric is right."""
    m = _model()
    report = evaluate_world_model(m, _loader(n_batches=2), device="cpu")  # type: ignore
    assert abs(report.ce_loss - math.log(VOCAB)) < 0.7, (
        f"untrained ce_loss {report.ce_loss:.3f} far from ln(vocab)={math.log(VOCAB):.3f}"
    )


# ---- THE learning test ----

@pytest.mark.slow
def test_loss_decreases(tmp_path: Path) -> None:
    """Cross-entropy must fall well below ln(vocab) as the model overfits a
    fixed batch. If the shift leaked, it would crash to ~0 instantly; if the
    loop were broken, it would not move at all."""
    m = _model()
    loader = _loader(n_batches=1)

    start = evaluate_world_model(m, loader, device="cpu").ce_loss # type: ignore
    cfg = TrainConfig(
        max_steps=150, lr=3e-3, device="cpu", amp=False,
        log_every=1000, checkpoint_every=1000,
        checkpoint_dir=str(tmp_path / "c"),
    )
    train_world_model(m, loader, cfg) # type: ignore
    end = evaluate_world_model(m, loader, device="cpu").ce_loss # type: ignore

    assert start > math.log(VOCAB) - 0.7, "did not start near random guessing"
    assert end < start * 0.5, f"loss did not decrease: {start:.3f} -> {end:.3f}"