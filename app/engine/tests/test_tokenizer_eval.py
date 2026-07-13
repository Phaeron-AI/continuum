"""Sub-phase 1.4 verification: the decode path, evaluation metrics, and the
frozen artifact. The two load-bearing correctness properties for
decode_indices — index bit-exactness and encode/decode interface identity —
are both tested, per the design decision.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.models.tokenizer.checkpoint import save_checkpoint
from engine.models.tokenizer.config import TokenizerConfig
from engine.models.tokenizer.evaluate import EvalReport, evaluate_tokenizer
from engine.models.tokenizer.frozen import FrozenTokenizer
from engine.models.tokenizer.modules.quantizer import FSQ
from engine.models.tokenizer.tokenizer import Tokenizer


def _tiny() -> Tokenizer:
    return Tokenizer(TokenizerConfig(levels=(4, 4, 4), hidden=16))


# ---- correctness property 1: bit-exact index round trip (the mixed-radix math)

@pytest.mark.slow
def test_indices_codes_indices_is_lossless() -> None:
    """indices -> codes -> indices must be the identity for every valid id.
    This tests the mixed-radix encode/decode pair in isolation."""
    fsq = FSQ(levels=(8, 8, 8, 5, 5))
    all_ids = torch.arange(fsq.codebook_size)
    codes = fsq.indices_to_codes(all_ids) # type: ignore
    recovered = fsq.codes_to_indices(codes)
    assert torch.equal(recovered, all_ids)


# ---- correctness property 2: encode/decode interface identity

def test_encode_decode_matches_forward_reconstruction() -> None:
    """decode_indices(encode_indices(x)) must equal the model's own forward
    reconstruction — the discrete interface and the continuous forward path
    agree."""
    tok = _tiny().eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        recon_forward, _ = tok(x)
        indices = tok.encode_indices(x)
        recon_via_indices = tok.decode_indices(indices)
    assert torch.allclose(recon_forward, recon_via_indices, atol=1e-5)


def test_decode_indices_shapes() -> None:
    tok = _tiny().eval()
    indices = torch.randint(0, tok.token_spec.vocab_size, (3, 8, 8))
    frames = tok.decode_indices(indices)
    assert frames.shape == (3, 3, 64, 64)


# ---- evaluation

def test_eval_report_metrics() -> None:
    tok = _tiny()
    data = torch.rand(4, 3, 64, 64) * 2 - 1
    loader = [data, data]
    report = evaluate_tokenizer(tok, loader, device="cpu")  # type: ignore
    assert isinstance(report, EvalReport)
    assert report.num_frames == 8
    assert report.recon_mse >= 0.0
    assert 0.0 <= report.codebook_usage <= 1.0
    assert report.vocab_size == 4 * 4 * 4
    assert "psnr" in report.summary()


def test_perfect_reconstruction_gives_high_psnr() -> None:
    """PSNR sanity: near-zero MSE -> large dB."""
    from engine.models.tokenizer.evaluate import _psnr_from_mse

    assert _psnr_from_mse(0.0) == float("inf")
    assert _psnr_from_mse(1e-6) > 40.0


# ---- frozen artifact

def test_frozen_tokenizer_is_immutable_and_usable(tmp_path: Path) -> None:
    tok = _tiny()
    opt = torch.optim.AdamW(tok.parameters(), lr=1e-3)
    ckpt = tmp_path / "tok.pt"
    save_checkpoint(ckpt, tok, opt, None, step=10)

    frozen = FrozenTokenizer.from_checkpoint(ckpt, device="cpu")
    # spec preserved
    assert frozen.token_spec.vocab_size == 4 * 4 * 4
    # parameters frozen
    assert all(not p.requires_grad for p in frozen._tokenizer.parameters())
    # interface works and round-trips through indices
    x = torch.randn(2, 3, 64, 64)
    idx = frozen.encode(x)
    assert idx.shape == (2, 8, 8)
    frames = frozen.decode(idx)
    assert frames.shape == (2, 3, 64, 64)