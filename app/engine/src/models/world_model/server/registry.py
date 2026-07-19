from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from uuid import uuid4

from models.world_model.session import GenerationSession

@dataclass
class SessionEntry:
  session: GenerationSession
  lock: asyncio.Lock = field(default_factory=asyncio.Lock)
  last_touched: float = field(default_factory=time.monotonic)


class SessionRegistry:
  def __init__(self, ttl_seconds: float = 300.0, sweep_interval_seconds: float = 30.0)-> None:
    self._entries: dict[str, SessionEntry] = {}
    self._ttl = ttl_seconds
    self._sweep_interval = sweep_interval_seconds
    self._sweep_task: asyncio.Task | None = None
  
  def start(self)-> None:
    if self._sweep_task is None:
      self._sweep_task = asyncio.create_task(self._sweep_loop())
  
  async def stop(self)-> None:
    """Cancel the sweep and close every remaining session. Call at shutdown."""
    if self._sweep_task is not None:
      self._sweep_task.cancel()
      try:
        await self._sweep_task
      except asyncio.CancelledError:
        pass
      self._sweep_task = None
 
    for entry in list(self._entries.values()):
      entry.session.close()
    self._entries.clear()
  
  def create(self, session: GenerationSession)-> str:
    session_id = uuid4().hex
    self._entries[session_id] = SessionEntry(session=session)
    return session_id
 
  def get(self, session_id: str)-> SessionEntry | None:
    entry = self._entries.get(session_id)
    if entry is not None:
      entry.last_touched = time.monotonic()
    return entry
 
  def remove(self, session_id: str)-> None:
    entry = self._entries.pop(session_id, None)
    if entry is not None:
      entry.session.close()
 
  async def _sweep_loop(self)-> None:
    while True:
      await asyncio.sleep(self._sweep_interval)
      now = time.monotonic()
      expired = [
        sid for sid, entry in self._entries.items()
        if now - entry.last_touched > self._ttl
      ]
      for sid in expired:
        self.remove(sid)