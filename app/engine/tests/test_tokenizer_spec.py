"""Sub-phase 1.3 verification: tokenizer assembly, training loop, and — the
load-bearing ones — checkpoint/resume integrity and that the loss actually
decreases (STE + loop wired correctly end to end).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.tokenizer import (
    build_tokenizer_from_checkpoint,
    load_checkpoint,
    TokenizerConfig,
    Tokenizer,
    TrainConfig,
    train_tokenizer
)


def _tiny_tokenizer() -> Tokenizer:
    # small hidden to keep CPU tests fast
    return Tokenizer(TokenizerConfig(levels=(4, 4, 4), hidden=16))


def _fixed_loader(n_batches: int = 4, batch: int = 4):
    """A tiny in-memory loader of a FIXED repeating frame batch, so the loss
    has something learnable and tests are deterministic."""
    torch.manual_seed(0)
    data = torch.rand(batch, 3, 64, 64) * 2 - 1  # in [-1, 1]
    return [data for _ in range(n_batches)]


def test_tokenizer_forward_shapes() -> None:
    tok = _tiny_tokenizer()
    x = torch.randn(2, 3, 64, 64)
    recon, indices = tok(x)
    assert recon.shape == (2, 3, 64, 64)
    assert indices.shape == (2, 8, 8)


def test_config_produces_consistent_spec() -> None:
    tok = _tiny_tokenizer()
    spec = tok.token_spec
    assert spec.grid_height == 8 and spec.grid_width == 8
    assert spec.vocab_size == 4 * 4 * 4
    assert spec.latent_dim == 3  # == len(levels)


def test_config_rejects_bad_downsample() -> None:
    # input/grid must be exactly 8x; 64/16 = 4 is invalid
    with pytest.raises(ValueError):
        TokenizerConfig(input_size=64, grid_size=16)


def test_short_training_runs_and_checkpoints(tmp_path: Path) -> None:
    tok = _tiny_tokenizer()
    loader = _fixed_loader()
    cfg = TrainConfig(
        max_steps=5,
        log_every=10,
        checkpoint_every=100,  # only the final checkpoint fires
        device="cpu",
        amp=False,
        checkpoint_dir=str(tmp_path / "ckpts"),
    )
    state = train_tokenizer(tok, loader, cfg) # type: ignore
    assert state.step == 5
    # final checkpoint written
    ckpts = list((tmp_path / "ckpts").glob("*.pt"))
    assert len(ckpts) == 1


def test_checkpoint_captures_full_state(tmp_path: Path) -> None:
    tok = _tiny_tokenizer()
    cfg = TrainConfig(
        max_steps=3, device="cpu", amp=False, checkpoint_dir=str(tmp_path / "c")
    )
    train_tokenizer(tok, _fixed_loader(), cfg)  # type: ignore
    ckpt_path = next((tmp_path / "c").glob("*.pt"))
    ckpt = load_checkpoint(ckpt_path)
    # everything needed for a clean resume is present
    for key in ("step", "model_state", "optimizer_state", "scaler_state", "config", "token_spec"):
        assert key in ckpt
    assert ckpt["step"] == 3


def test_resume_continues_from_saved_step(tmp_path: Path) -> None:
    """Train 3 steps, checkpoint, rebuild, resume 2 more -> ends at step 5,
    weights carried over."""
    tok = _tiny_tokenizer()
    cfg1 = TrainConfig(max_steps=3, device="cpu", amp=False, checkpoint_dir=str(tmp_path / "c"))
    train_tokenizer(tok, _fixed_loader(), cfg1) # type: ignore

    ckpt_path = next((tmp_path / "c").glob("*.pt"))
    resumed_model, ckpt = build_tokenizer_from_checkpoint(ckpt_path)

    # weights match what was saved
    for p_new, p_old in zip(
        resumed_model.parameters(), tok.parameters(), strict=True
    ):
        assert torch.allclose(p_new, p_old)

    cfg2 = TrainConfig(max_steps=5, device="cpu", amp=False, checkpoint_dir=str(tmp_path / "c2"))
    state = train_tokenizer(resumed_model, _fixed_loader(), cfg2, resume_state=ckpt)  # type: ignore
    assert state.step == 5  


def test_loss_decreases_on_fixed_batch(tmp_path: Path) -> None:
    """THE learning test: on a fixed batch the tokenizer must overfit —
    reconstruction MSE at step 200 well below step 1. If this fails, the STE
    or loop is broken (roadmap stages 04, 07)."""
    torch.manual_seed(0)
    tok = _tiny_tokenizer()
    data = torch.rand(4, 3, 64, 64) * 2 - 1
    loader = [data]  # single fixed batch, repeated

    # measure initial loss
    tok.eval()
    with torch.no_grad():
        recon0, _ = tok(data)
        loss_start = torch.nn.functional.mse_loss(recon0, data).item()

    cfg = TrainConfig(
        max_steps=200, lr=1e-3, device="cpu", amp=False,
        log_every=1000, checkpoint_every=10000,
        checkpoint_dir=str(tmp_path / "c"),
    )
    train_tokenizer(tok, loader, cfg) # type: ignore

    tok.eval()
    with torch.no_grad():
        recon1, _ = tok(data)
        loss_end = torch.nn.functional.mse_loss(recon1, data).item()

    # A clear decrease proves the STE + loop learn; the exact ratio depends
    # on the tiny test model and step budget, so assert meaningful (not
    # magic-number) improvement.
    assert loss_end < loss_start * 0.7, f"loss did not decrease: {loss_start} -> {loss_end}"