"""Sub-phase 3.4 verification: the FastAPI server surface.

Structural tests only — no trained checkpoints exist yet, so these verify
that routing, session lifecycle, error mapping, and pixel/token
(de)serialization all work correctly, not that generation quality is any
good. A tiny randomly-initialized WorldModel and a fake tokenizer are
injected via app.dependency_overrides, so the real checkpoint loader
(_load_model_and_tokenizer) is never invoked.
"""

from __future__ import annotations

import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import Tensor

from models.tokenizer.spec import TokenSpec
from models.world_model.model.config import WorldModelConfig
from models.world_model.model.world_model import WorldModel
from models.world_model.server.app import app, get_model, get_registry, get_tokenizer
from models.world_model.server.registry import SessionRegistry


def _model(n_layers: int = 2, **kw) -> WorldModel:
  torch.manual_seed(0)
  base = dict(vocab_size=32, num_actions=5, d_model=16, d_state=4, n_layers=n_layers)
  base.update(kw)
  return WorldModel(WorldModelConfig(**base)).eval()  # type: ignore


class _FakeTokenizer:
  """Extends the fake used in test_session*.py with encode(), needed here
  since the server does real PNG -> token encoding on open()."""

  def __init__(self, spec: TokenSpec) -> None:
    self._spec = spec

  @property
  def token_spec(self) -> TokenSpec:
    return self._spec

  def encode(self, frames: Tensor) -> Tensor:
    # frames: (T, C, H, W) -> (T, h, w). Content-independent (structural
    # test only) but shape- and range-correct: valid token ids in [0, vocab).
    t = frames.shape[0]
    return torch.zeros(t, self._spec.grid_height, self._spec.grid_width, dtype=torch.long)

  def decode(self, indices: Tensor) -> Tensor:
    batch = indices.shape[0]
    spec = self._spec
    return torch.zeros(batch, spec.input_channels, spec.input_height, spec.input_width)


def _tokenizer(grid_height: int = 2, grid_width: int = 2) -> _FakeTokenizer:
  spec = TokenSpec(grid_height=grid_height, grid_width=grid_width, levels=(2, 2, 2, 2, 2))
  return _FakeTokenizer(spec)


def _png_bytes(size: int = 8) -> bytes:
  img = Image.new("RGB", (size, size), color=(128, 64, 32))
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  return buf.getvalue()


@pytest.fixture()
def client():
  model = _model()
  tokenizer = _tokenizer()

  app.dependency_overrides[get_model] = lambda: model
  app.dependency_overrides[get_tokenizer] = lambda: tokenizer

  with TestClient(app) as c:
    yield c

  app.dependency_overrides.clear()


def _open_session(client: TestClient, n_frames: int = 2) -> str:
  files = [("frames", (f"seed{i}.png", _png_bytes(), "image/png")) for i in range(n_frames)]
  resp = client.post("/sessions", files=files)
  assert resp.status_code == 201, resp.text
  return resp.json()["session_id"]


# ---- open_session ----

def test_open_session_returns_session_id(client: TestClient) -> None:
  session_id = _open_session(client)
  assert isinstance(session_id, str)
  assert len(session_id) > 0


def test_open_session_no_frames_is_422(client: TestClient) -> None:
  resp = client.post("/sessions", files=[])
  assert resp.status_code == 422


# ---- advance ----

def test_advance_returns_png_by_default(client: TestClient) -> None:
  session_id = _open_session(client)
  resp = client.post(f"/sessions/{session_id}/advance", json={"action": 0})
  assert resp.status_code == 200
  assert resp.headers["content-type"] == "image/png"
  # valid PNG signature
  assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_advance_return_pixels_false_returns_grid(client: TestClient) -> None:
  session_id = _open_session(client)
  resp = client.post(
    f"/sessions/{session_id}/advance",
    json={"action": 0, "return_pixels": False},
  )
  assert resp.status_code == 200
  body = resp.json()
  grid = body["grid"]
  assert len(grid) == 2  # grid_height
  assert len(grid[0]) == 2  # grid_width


def test_advance_unknown_session_is_404(client: TestClient) -> None:
  resp = client.post("/sessions/does-not-exist/advance", json={"action": 0})
  assert resp.status_code == 404


def test_advance_after_close_is_404(client: TestClient) -> None:
  session_id = _open_session(client)
  client.delete(f"/sessions/{session_id}")
  resp = client.post(f"/sessions/{session_id}/advance", json={"action": 0})
  assert resp.status_code == 404


# ---- metrics ----

def test_metrics_before_advance_is_404(client: TestClient) -> None:
  session_id = _open_session(client)
  resp = client.get(f"/sessions/{session_id}/metrics")
  assert resp.status_code == 404


def test_metrics_after_advance_has_expected_fields(client: TestClient) -> None:
  session_id = _open_session(client)
  client.post(f"/sessions/{session_id}/advance", json={"action": 0, "return_pixels": False})

  resp = client.get(f"/sessions/{session_id}/metrics")
  assert resp.status_code == 200
  body = resp.json()
  assert body["count"] == 1
  assert body["last"]["total_ms"] >= 0.0
  assert body["fps"] > 0.0


def test_metrics_unknown_session_is_404(client: TestClient) -> None:
  resp = client.get("/sessions/does-not-exist/metrics")
  assert resp.status_code == 404


# ---- close ----

def test_close_then_metrics_is_404(client: TestClient) -> None:
  session_id = _open_session(client)
  client.post(f"/sessions/{session_id}/advance", json={"action": 0, "return_pixels": False})

  resp = client.delete(f"/sessions/{session_id}")
  assert resp.status_code == 204

  resp = client.get(f"/sessions/{session_id}/metrics")
  assert resp.status_code == 404


def test_close_unknown_session_is_404(client: TestClient) -> None:
  resp = client.delete("/sessions/does-not-exist")
  assert resp.status_code == 404


# ---- registry wiring ----

def test_registry_dependency_is_shared_across_requests(client: TestClient) -> None:
  """Sanity check that get_registry resolves to the same app.state
  instance across requests, not a fresh one each time (which would make
  sessions vanish between calls)."""
  session_id = _open_session(client)
  # If the registry weren't shared/stable, this second call would 404.
  resp = client.post(f"/sessions/{session_id}/advance", json={"action": 0, "return_pixels": False})
  assert resp.status_code == 200