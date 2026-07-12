"""Sub-phase 2.1 verification: the selective SSM reference implementation.

The load-bearing test is `test_scan_matches_naive_reference` — it asserts the
vectorised scan computes exactly what the fully-explicit recurrence computes.
With no compiled mamba-ssm kernel available yet as a numerical oracle, this
self-check is what catches an implementation slip (a wrong broadcast axis, a
swapped index) that would otherwise train silently and wrongly.

`test_causality` is the second critical one: position i must never depend on
positions > i, or teacher forcing (roadmap stage 05) is invalid and training
loss collapses to a meaningless near-zero.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from engine.models.world_model.mixer import SequenceMixer
from engine.models.world_model.ssm import SelectiveSSM


def _ssm(d_model: int = 4, d_state: int = 3) -> SelectiveSSM:
    torch.manual_seed(0)
    return SelectiveSSM(d_model=d_model, d_state=d_state)


# ---- interface conformance ----

def test_is_a_sequence_mixer() -> None:
    m = _ssm()
    assert isinstance(m, SequenceMixer)


def test_shape_preserved() -> None:
    m = _ssm(d_model=8, d_state=4)
    x = torch.randn(3, 12, 8)
    assert m(x).shape == (3, 12, 8)


def test_rejects_bad_shape() -> None:
    m = _ssm(d_model=8)
    with pytest.raises(ValueError):
        m(torch.randn(3, 12))  # not 3D
    with pytest.raises(ValueError):
        m(torch.randn(3, 12, 5))  # wrong d_model


# ---- THE self-check: scan == naive recurrence ----

def test_scan_matches_naive_reference() -> None:
    """The vectorised scan must compute exactly the recurrence the naive,
    fully-explicit loop computes. This is the correctness oracle available to
    us before the compiled kernel exists."""
    m = _ssm(d_model=4, d_state=3).eval()
    x = torch.randn(2, 6, 4)
    with torch.no_grad():
        fast = m(x)
        slow = m.naive_reference(x)
    assert torch.allclose(fast, slow, atol=1e-5), (
        f"scan diverges from naive recurrence: max diff "
        f"{(fast - slow).abs().max().item()}"
    )


@pytest.mark.slow
def test_scan_matches_naive_across_shapes() -> None:
    """Same equivalence over several shapes — guards broadcasting bugs that
    only appear at particular dimensions."""
    for d_model, d_state, length, batch in [(2, 2, 3, 1), (5, 4, 7, 2), (3, 6, 4, 3)]:
        torch.manual_seed(1)
        m = SelectiveSSM(d_model=d_model, d_state=d_state).eval()
        x = torch.randn(batch, length, d_model)
        with torch.no_grad():
            assert torch.allclose(m(x), m.naive_reference(x), atol=1e-5), (
                f"mismatch at d_model={d_model} d_state={d_state} L={length}"
            )


# ---- causality: the teacher-forcing prerequisite ----

def test_causality() -> None:
    """Perturbing token t must not change any output before t. Without this,
    the model sees the future and teacher forcing is invalid."""
    m = _ssm(d_model=4, d_state=3).eval()
    x = torch.randn(1, 6, 4)
    with torch.no_grad():
        y1 = m(x)
        x2 = x.clone()
        x2[0, 4] = torch.randn(4)  # perturb position 4
        y2 = m(x2)

    # positions before the perturbation are untouched
    assert torch.equal(y1[0, :4], y2[0, :4]), "future token leaked into the past"
    # and the perturbation does take effect from that position on
    assert not torch.allclose(y1[0, 4:], y2[0, 4:]), "perturbation had no effect"


# ---- stability: A negative => A_bar in (0, 1) ----

def test_state_transition_is_stable() -> None:
    """A is parameterised as -exp(A_log), so it is always negative and
    A_bar = exp(delta * A) lies in (0, 1): state decays, never explodes."""
    m = _ssm(d_model=4, d_state=3)
    A = -torch.exp(m.A_log)
    assert bool((A < 0).all()), "A must be negative for a stable recurrence"

    x = torch.randn(2, 5, 4)
    with torch.no_grad():
        delta = F.softplus(m.delta_proj(x))
        A_bar = torch.exp(delta.unsqueeze(-1) * A)
    assert bool((A_bar > 0).all()) and bool((A_bar < 1).all()), (
        "A_bar must lie strictly within (0, 1)"
    )


# ---- selectivity: the thing that makes it Mamba, not S4 ----

def test_selectivity_is_input_dependent() -> None:
    """delta (and hence retention) must VARY across positions because it is a
    function of the input. A non-selective SSM would produce an identical
    value at every position."""
    m = _ssm(d_model=4, d_state=3).eval()
    x = torch.randn(1, 6, 4)
    with torch.no_grad():
        delta = F.softplus(m.delta_proj(x))  # (1, 6, D)
    per_position = delta[0].mean(dim=-1)
    assert per_position.std() > 1e-4, (
        "delta is constant across positions — the SSM is not selective"
    )


def test_identical_inputs_give_identical_deltas() -> None:
    """Selectivity is a function of the input, so the same token in the same
    state must produce the same delta — determinism, not noise."""
    m = _ssm(d_model=4, d_state=3).eval()
    tok = torch.randn(1, 1, 4)
    x = torch.cat([tok, tok], dim=1)  # same token twice
    with torch.no_grad():
        delta = F.softplus(m.delta_proj(x))
    assert torch.allclose(delta[0, 0], delta[0, 1]), "delta must be a function of input"


# ---- gradients reach the parameters ----

def test_gradients_flow_to_all_parameters() -> None:
    m = _ssm(d_model=4, d_state=3)
    x = torch.randn(2, 5, 4)
    m(x).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert not torch.allclose(p.grad, torch.zeros_like(p.grad)), (
            f"zero gradient at {name}"
        )