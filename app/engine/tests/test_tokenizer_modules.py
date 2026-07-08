"""Sub-phase 1.2 verification: tokenizer modules in isolation.

The two load-bearing tests are (a) FSQ matches the published reference
output/index for a known input, proving our quantization math is the real
FSQ and not an approximation of it, and (b) the straight-through estimator
delivers an all-ones gradient through round(), proving the encoder can learn.
Everything else is shape/contract checking.
"""

from __future__ import annotations

import torch

from engine.models.tokenizer.modules.decoder import Decoder
from engine.models.tokenizer.modules.encoder import Encoder
from engine.models.tokenizer.modules.quantizer import FSQ, round_ste


def test_encoder_downsamples_to_grid() -> None:
    enc = Encoder(in_channels=3, hidden=32, latent_dim=5)
    x = torch.randn(4, 3, 64, 64)
    z = enc(x)
    assert z.shape == (4, 5, 8, 8)


def test_decoder_upsamples_to_frame() -> None:
    dec = Decoder(out_channels=3, hidden=32, latent_dim=5)
    z = torch.randn(4, 5, 8, 8)
    x = dec(z)
    assert x.shape == (4, 3, 64, 64)
    # tanh output must be in [-1, 1]
    assert x.min() >= -1.0 and x.max() <= 1.0


def test_encoder_decoder_round_trip_shape() -> None:
    """The classic tokenizer bug is a 63x63 or 65x65 decoder output. Assert
    the shape survives a full encode->decode round trip exactly."""
    enc = Encoder(hidden=32, latent_dim=5)
    dec = Decoder(hidden=32, latent_dim=5)
    x = torch.randn(2, 3, 64, 64)
    out = dec(enc(x))
    assert out.shape == x.shape


def test_fsq_matches_reference_example() -> None:
    """Reference (Mentzer et al. official impl): FSQ(levels=[3,5,4]) maps
    [0.25, 0.6, -7] -> [0.0, 0.5, -1.0], index 10. We test our quantize()
    reproduces the same code (normalized) and index."""
    fsq = FSQ(levels=(3, 5, 4))
    z = torch.tensor([[0.25, 0.6, -7.0]])  # (1, D)
    code = fsq.quantize(z)
    idx = fsq.codes_to_indices(code)
    # Reference normalized code is [0.0, 0.5, -1.0]; index 10.
    assert torch.allclose(code, torch.tensor([[0.0, 0.5, -1.0]]), atol=1e-4)
    assert idx.item() == 10


def test_fsq_output_is_on_finite_grid() -> None:
    """Every quantized value must be one of exactly L_i distinct levels."""
    fsq = FSQ(levels=(8, 8, 8, 5, 5))
    z = torch.randn(100, 5) * 3
    code = fsq.quantize(z)
    for dim, level in enumerate(fsq.levels):  # type: ignore
        distinct = torch.unique(code[:, dim])
        assert distinct.numel() <= level


def test_fsq_indices_in_range() -> None:
    fsq = FSQ(levels=(8, 8, 8, 5, 5))
    z = torch.randn(4, 5, 8, 8)
    _, indices = fsq(z)
    assert indices.shape == (4, 8, 8)
    assert indices.min() >= 0
    assert indices.max() < fsq.codebook_size


def test_round_ste_gradient_is_identity() -> None:
    """THE load-bearing test: forward is round(), backward is all-ones. A
    broken detach here silently kills encoder learning."""
    z = torch.tensor([0.2, 0.7, -1.3, 2.9], requires_grad=True)
    out = round_ste(z)
    # forward equals true rounding
    assert torch.allclose(out, torch.round(z))
    # backward is identity: d out / d z == 1 everywhere
    out.sum().backward()
    assert torch.allclose(z.grad, torch.ones_like(z)) # type: ignore


def test_fsq_forward_grad_flows_to_input() -> None:
    """End-to-end: gradient must reach the quantizer input, or the encoder
    upstream would never update."""
    fsq = FSQ(levels=(8, 8, 8, 5, 5))
    z = torch.randn(2, 5, 8, 8, requires_grad=True)
    quantized, _ = fsq(z)
    quantized.sum().backward()
    assert z.grad is not None
    assert not torch.allclose(z.grad, torch.zeros_like(z.grad))