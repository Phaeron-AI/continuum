"""Sub-phase 1.0 verification: the TokenSpec contract and device resolution.

Pure-contract tests — no model, no training. These pin the interface Phase 2
depends on before any tokenizer implementation exists.
"""

from __future__ import annotations

import math

import pytest

from engine.models.device import resolve_device
from engine.models.tokenizer.spec import TokenSpec


def test_vocab_size_is_product_of_levels() -> None:
    spec = TokenSpec(grid_height=8, grid_width=8, levels=(8, 8, 8, 5, 5))
    assert spec.vocab_size == 8 * 8 * 8 * 5 * 5 == 12800


def test_latent_dim_derived_from_levels() -> None:
    spec = TokenSpec(grid_height=8, grid_width=8, levels=(8, 6, 5))
    assert spec.latent_dim == 3


def test_latent_dim_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        TokenSpec(grid_height=8, grid_width=8, levels=(8, 8), latent_dim=3)


def test_bits_math_matches_roadmap() -> None:
    spec = TokenSpec(grid_height=8, grid_width=8, levels=(8, 8, 8, 5, 5))
    assert spec.tokens_per_frame == 64
    assert spec.bits_per_token == pytest.approx(math.log2(12800))
    assert spec.bits_per_frame == pytest.approx(64 * math.log2(12800))


def test_invalid_specs_rejected() -> None:
    with pytest.raises(ValueError):
        TokenSpec(grid_height=0, grid_width=8, levels=(8, 8))
    with pytest.raises(ValueError):
        TokenSpec(grid_height=8, grid_width=8, levels=())
    with pytest.raises(ValueError):
        TokenSpec(grid_height=8, grid_width=8, levels=(8, 1))  # level < 2


def test_tokenspec_round_trip() -> None:
    spec = TokenSpec(
        grid_height=8,
        grid_width=8,
        levels=(8, 8, 8, 5, 5),
        tokenizer_version="v3",
        input_height=64,
    )
    restored = TokenSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.vocab_size == spec.vocab_size


def test_resolve_device_explicit_and_auto() -> None:
    assert resolve_device("cpu").type == "cpu"
    # auto resolves to a valid device type regardless of hardware
    assert resolve_device("auto").type in {"cpu", "cuda"}