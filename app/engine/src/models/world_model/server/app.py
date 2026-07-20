from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from functools import lru_cache

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image

from models.tokenizer.frozen import FrozenTokenizer
from models.world_model.model.checkpoint import build_world_model_from_checkpoint
from models.world_model.model.world_model import WorldModel
from models.world_model.server.registry import SessionRegistry
from models.world_model.server.schemas import (
  AdvanceRequest,
  FrameLatencyResponse,
  LatencyReportResponse,
  OpenSessionResponse,
  TokenGridResponse,
)
from models.world_model.session import GenerationSession

MODEL_CHECKPOINT = os.environ.get("CONTINUUM_MODEL_CHECKPOINT", "")
TOKENIZER_CHECKPOINT = os.environ.get("CONTINUUM_TOKENIZER_CHECKPOINT", "")
DEVICE = os.environ.get("CONTINUUM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SESSION_TTL_SECONDS = float(os.environ.get("CONTINUUM_SESSION_TTL_SECONDS", "300"))
SWEEP_INTERVAL_SECONDS = float(os.environ.get("CONTINUUM_SWEEP_INTERVAL_SECONDS", "30"))


@lru_cache(maxsize=1)
def _load_model_and_tokenizer()-> tuple[WorldModel, FrozenTokenizer]:
  """Loads real checkpoints from disk, once, on first use. Not called
  during tests — get_model/get_tokenizer are overridden via
  app.dependency_overrides before any request reaches here.
  """
  if not MODEL_CHECKPOINT or not TOKENIZER_CHECKPOINT:
    raise RuntimeError(
      "CONTINUUM_MODEL_CHECKPOINT / CONTINUUM_TOKENIZER_CHECKPOINT are not set — "
      "set them, or override get_model/get_tokenizer (e.g. in tests)"
    )

  tokenizer = FrozenTokenizer.from_checkpoint(TOKENIZER_CHECKPOINT, device=DEVICE)

  # expect_tokenizer_version catches a mismatched model/tokenizer pair
  # loudly, on first use — instead of silently-wrong token ids that only
  # surface as garbage generations later.
  model, _ckpt = build_world_model_from_checkpoint(
    MODEL_CHECKPOINT,
    map_location=DEVICE,
    expect_tokenizer_version=tokenizer.token_spec.tokenizer_version,
  )
  model.to(DEVICE)
  model.eval()
  return model, tokenizer


def get_model()-> WorldModel:
  model, _tokenizer = _load_model_and_tokenizer()
  return model


def get_tokenizer()-> FrozenTokenizer:
  _model, tokenizer = _load_model_and_tokenizer()
  return tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
  registry = SessionRegistry(
    ttl_seconds=SESSION_TTL_SECONDS, sweep_interval_seconds=SWEEP_INTERVAL_SECONDS
  )
  registry.start()
  app.state.registry = registry
  try:
    yield
  finally:
    await registry.stop()


app = FastAPI(title="continuum inference server", lifespan=lifespan)


def get_registry(request: Request)-> SessionRegistry:
  return request.app.state.registry


def _decode_png_to_tensor(data: bytes)-> torch.Tensor:
  img = Image.open(io.BytesIO(data)).convert("RGB")
  arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, C)
  return torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)


def _encode_tensor_to_png(frame: torch.Tensor)-> bytes:
  arr = (frame.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
  arr = arr.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
  img = Image.fromarray(arr, mode="RGB")
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  return buf.getvalue()


@app.post("/sessions", response_model=OpenSessionResponse, status_code=201)
async def open_session(
  frames: list[UploadFile] = File(...),
  model: WorldModel = Depends(get_model),
  tokenizer: FrozenTokenizer = Depends(get_tokenizer),
  registry: SessionRegistry = Depends(get_registry),
)-> OpenSessionResponse:
  """Open a session from a short seed clip — one or more PNG frames,
  earliest first. The server tokenizes and warms the recurrent state."""
  if not frames:
    raise HTTPException(status_code=400, detail="at least one seed frame is required")

  device = next(model.parameters()).device
  raw = [await f.read() for f in frames]
  clip = torch.stack([_decode_png_to_tensor(d) for d in raw]).to(device)  # (T, C, H, W)

  def _build_session()-> GenerationSession:
    with torch.no_grad():
      indices = tokenizer.encode(clip)  # (T, h, w)
    # Row-major flatten per frame, concatenated across T — must match the
    # token order used during training data generation. Verify against a
    # known seed clip once a real trained model/tokenizer exist.
    seed_tokens = indices.reshape(1, -1)  # (1, T*h*w); batch=1 per server session
    session = GenerationSession(model, tokenizer)
    session.open(seed_tokens)
    return session

  session = await run_in_threadpool(_build_session)
  session_id = registry.create(session)
  return OpenSessionResponse(session_id=session_id)


@app.post("/sessions/{session_id}/advance")
async def advance_session(
  session_id: str,
  body: AdvanceRequest,
  registry: SessionRegistry = Depends(get_registry),
):
  entry = registry.get(session_id)
  if entry is None:
    raise HTTPException(status_code=404, detail="session not found")

  async with entry.lock:
    def _run():
      action = torch.tensor([body.action], dtype=torch.long)
      return entry.session.advance(
        action, return_pixels=body.return_pixels, temperature=body.temperature
      )

    try:
      result = await run_in_threadpool(_run)
    except RuntimeError as exc:
      # advance() before open() — session exists but isn't started
      raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
      # bad action shape / batch mismatch
      raise HTTPException(status_code=400, detail=str(exc)) from exc

  if body.return_pixels:
    png_bytes = _encode_tensor_to_png(result[0])  # batch=1 -> the one frame
    return Response(content=png_bytes, media_type="image/png")

  return TokenGridResponse(grid=result[0].tolist())


@app.get("/sessions/{session_id}/metrics", response_model=LatencyReportResponse)
async def session_metrics(
  session_id: str,
  registry: SessionRegistry = Depends(get_registry),
)-> LatencyReportResponse:
  entry = registry.get(session_id)
  if entry is None:
    raise HTTPException(status_code=404, detail="session not found")

  report = entry.session.metrics()
  if report is None:
    raise HTTPException(status_code=404, detail="no frames generated yet")

  return LatencyReportResponse(
    last=FrameLatencyResponse(
      step_ms=report.last.step_ms,
      sample_ms=report.last.sample_ms,
      decode_ms=report.last.decode_ms,
      total_ms=report.last.total_ms,
    ),
    count=report.count,
    mean_total_ms=report.mean_total_ms,
    p50_total_ms=report.p50_total_ms,
    p95_total_ms=report.p95_total_ms,
    fps=report.fps,
  )


@app.delete("/sessions/{session_id}", status_code=204)
async def close_session(
  session_id: str,
  registry: SessionRegistry = Depends(get_registry),
)-> Response:
  entry = registry.get(session_id)
  if entry is None:
    raise HTTPException(status_code=404, detail="session not found")
  registry.remove(session_id)
  return Response(status_code=204)