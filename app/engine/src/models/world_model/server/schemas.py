from __future__ import annotations

from pydantic import BaseModel

class OpenSessionResponse(BaseModel):
  session_id: str

class AdvanceRequest(BaseModel):
  action: int
  return_pixels: bool = True
  temperature: float = 0.0

class TokenGridResponse(BaseModel):
  grid: list[list[int]]


class FrameLatencyResponse(BaseModel):
  step_ms: float
  sample_ms: float
  decode_ms: float
  total_ms: float

class LatencyReportResponse(BaseModel):
  last: FrameLatencyResponse
  count: int
  mean_total_ms: float
  p50_total_ms: float
  p95_total_ms: float
  fps: float