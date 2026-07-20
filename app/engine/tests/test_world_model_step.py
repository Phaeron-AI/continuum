"""Sub-phase 3.1 verification: WorldModel.step() and whole-model equivalence.

THE load-bearing test is `test_model_step_matches_forward`: generating
token-by-token through the full stack (embedding -> every block -> head),
carrying per-layer state, must produce the identical logits as the parallel
forward. This lifts 3.0's mixer-level equivalence up to the whole model — a
state-threading bug in any layer shows up here, and (as at the mixer level) it
would otherwise be an invisible generation error, not a crash.
"""

from __future__ import annotations

import pytest
import torch

from models.world_model.model.config import WorldModelConfig
from models.world_model.model.world_model import WorldModel


def _model(n_layers: int = 3, **kw) -> WorldModel:
  torch.manual_seed(0)
  base = dict(vocab_size=32, num_actions=5, d_model=16, d_state=4, n_layers=n_layers)
  base.update(kw)
  return WorldModel(WorldModelConfig(**base)) # type: ignore


def _run_recurrent(m: WorldModel, ids: torch.Tensor) -> torch.Tensor:
  """Step the model through a sequence one token at a time, carrying state."""
  state = None
  outs = []
  for t in range(ids.shape[1]):
    logits, state = m.step(ids[:, t], state)
    outs.append(logits)
  return torch.stack(outs, dim=1)


# ---- THE whole-model equivalence test ----

def test_model_step_matches_forward() -> None:
  """Recurrent generation through the full stack must equal parallel forward."""
  m = _model(n_layers=3).eval()
  ids = torch.randint(0, 37, (2, 15))  # total_vocab = 32 + 5
  with torch.no_grad():
    parallel = m(ids)
    recurrent = _run_recurrent(m, ids)
  assert torch.allclose(parallel, recurrent, atol=1e-4), (
    f"recurrent diverges from parallel: max diff "
    f"{(parallel - recurrent).abs().max().item()}"
  )


@pytest.mark.slow
def test_model_step_matches_forward_across_configs() -> None:
  """Equivalence across layer counts and dims — guards per-layer state
  threading bugs that only surface at particular depths."""
  for n_layers, d_model, d_state, length in [(1, 8, 2, 6), (4, 16, 8, 12), (2, 24, 4, 9)]:
    torch.manual_seed(1)
    m = WorldModel(
      WorldModelConfig(
        vocab_size=20, num_actions=4, d_model=d_model,
        d_state=d_state, n_layers=n_layers,
      )
    ).eval()
    ids = torch.randint(0, 24, (2, length))
    with torch.no_grad():
      assert torch.allclose(m(ids), _run_recurrent(m, ids), atol=1e-4), (
        f"mismatch at n_layers={n_layers} d_model={d_model} L={length}"
      )


# ---- state structure ----

def test_generation_state_is_per_layer() -> None:
  """The state is a list with one entry per block."""
  m = _model(n_layers=3)
  state = m.init_generation_state(2, torch.device("cpu"), torch.float32)
  assert isinstance(state, list)
  assert len(state) == 3
  for s in state:
    assert s.shape == (2, 16, 4)  # (B, D, N) per layer # type: ignore


def test_step_shapes() -> None:
  m = _model(n_layers=2)
  token = torch.randint(0, 37, (3,))
  logits, state = m.step(token, None)
  assert logits.shape == (3, 32)  # (B, vocab_size)
  assert len(state) == 2


def test_step_rejects_bad_token_shape() -> None:
  m = _model()
  with pytest.raises(ValueError):
    m.step(torch.randint(0, 32, (3, 1)), None)  # (B,1) not (B,)


# ---- state carries across steps ----

def test_state_carries_across_steps() -> None:
  """Same token twice gives different logits — the second sees accumulated
  state through every layer. Proof the state threads, not resets."""
  m = _model(n_layers=2).eval()
  token = torch.randint(0, 32, (1,))
  with torch.no_grad():
    l1, s1 = m.step(token, None)
    l2, _ = m.step(token, s1)
  assert not torch.allclose(l1, l2), "state not carried across steps"


# ---- action tokens step through identically ----

def test_action_token_updates_all_layers() -> None:
  """An action token, stepped in, must advance every layer's state — its
  effect has to be present before the next frame's tokens are generated
  (stage 02 pitfall: skipping this silently drops conditioning)."""
  m = _model(n_layers=3).eval()
  vocab = m.config.vocab_size
  action_tok = torch.tensor([vocab + 2])  # a valid action id

  with torch.no_grad():
    state0 = m.init_generation_state(1, torch.device("cpu"), torch.float32)
    before = [s.clone() for s in state0]  # type: ignore
    _, state1 = m.step(action_tok, state0)

  # every layer's state changed after stepping the action
  for lstate_before, lstate_after in zip(before, state1, strict=True):
    assert not torch.allclose(lstate_before, lstate_after), ( # type: ignore
      "an action token left some layer's state unchanged"
    )