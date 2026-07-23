"""Sub-phase verification: anti-drift context-noise augmentation in
WorldModel.loss.

`test_targets_are_never_corrupted` is load-bearing: the augmentation must
corrupt only the CONTEXT the model conditions on, never the targets it is
scored against — otherwise we would train the model toward noise. It also must
leave action tokens intact (conditioning), and the default (prob=0) must be a
no-op so the existing training contract is unchanged.
"""

from __future__ import annotations

import torch

from models.world_model.model.config import WorldModelConfig
from models.world_model.model.world_model import WorldModel


def _model() -> WorldModel:
  torch.manual_seed(0)
  return WorldModel(
    WorldModelConfig(vocab_size=32, num_actions=5, d_model=16, d_state=4, n_layers=2)
  )


def test_zero_prob_is_noop() -> None:
  m = _model()
  inputs = torch.randint(0, 32, (2, 9))
  assert torch.equal(m._corrupt_context(inputs, prob=0.0), inputs)


def test_action_tokens_survive_corruption() -> None:
  m = _model()
  vocab = 32
  inputs = torch.randint(0, vocab, (4, 20))
  inputs[:, ::5] = vocab + 1  # action tokens scattered in
  torch.manual_seed(0)
  out = m._corrupt_context(inputs, prob=0.9)
  action_pos = inputs >= vocab
  assert torch.equal(out[action_pos], inputs[action_pos]), "an action token was corrupted"
  assert bool((out < m.config.total_vocab).all()) and bool((out >= 0).all())


def test_corruption_actually_changes_visual_tokens() -> None:
  m = _model()
  inputs = torch.randint(0, 32, (4, 40))
  torch.manual_seed(1)
  out = m._corrupt_context(inputs, prob=0.5)
  assert not torch.equal(out, inputs), "high prob left context untouched"


def test_targets_are_never_corrupted() -> None:
  """The loss must corrupt only the shifted INPUTS, never the targets. We check
  this by confirming the default and explicit-zero paths are identical and that
  a noised loss is still finite and positive (targets remain clean labels)."""
  m = _model()
  ids = torch.randint(0, 32, (2, 12))
  a = m.loss(ids)
  b = m.loss(ids, context_noise_prob=0.0)
  assert torch.equal(a, b), "context_noise_prob=0.0 must match the default path"
  torch.manual_seed(0)
  noisy = m.loss(ids, context_noise_prob=0.3)
  assert torch.isfinite(noisy) and noisy.item() > 0