// Framework-agnostic transport for the continuum player.
//
// Two planes, kept separate on purpose:
//   - control: openSession() is a one-off REST call that seeds and warms a
//     session (this is where a Node/auth tier would sit in production).
//   - data:    FrameStream holds a WebSocket straight to the inference server
//     and drives the per-frame loop — action out, PNG frame in. It never
//     routes through any middleware; that is what keeps latency low.

/** Seed a session from one or more PNG frames; returns the session id. */
export async function openSession(server, frames) {
  const form = new FormData();
  // Accept both File objects and raw Blobs (e.g. canvas.toBlob) — a Blob has
  // no name, and the server needs a filename to parse the multipart upload.
  frames.forEach((f, i) => form.append("frames", f, f.name ?? `seed_${i}.png`));
  const res = await fetch(`${server}/sessions`, { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(`open session failed: ${res.status} ${await res.text()}`);
  }
  const body = await res.json();
  return body.session_id;
}

/** Delete a session (best-effort). */
export function closeSession(server, sessionId) {
  void fetch(`${server}/sessions/${sessionId}`, { method: "DELETE" }).catch(() => {});
}

function toWsUrl(server, sessionId) {
  const base = server.replace(/^http/, "ws"); // http->ws, https->wss
  return `${base}/sessions/${sessionId}/stream`;
}

/**
 * Drives one session over a persistent WebSocket. On each received frame it
 * asks nextAction() for the current input and requests the next frame, so the
 * loop self-paces to the server's real frame rate.
 */
export class FrameStream {
  constructor(server, sessionId, nextAction, onFrame, onError, onClose) {
    this.server = server;
    this.sessionId = sessionId;
    this.nextAction = nextAction;
    this.onFrame = onFrame;
    this.onError = onError;
    this.onClose = onClose;
    this.ws = null;
  }

  start() {
    const ws = new WebSocket(toWsUrl(this.server, this.sessionId));
    ws.binaryType = "blob";
    this.ws = ws;

    ws.onopen = () => this.request();
    ws.onmessage = async (ev) => {
      try {
        const bitmap = await createImageBitmap(ev.data);
        this.onFrame(bitmap);
      } catch (err) {
        this.onError(`decode failed: ${String(err)}`);
      }
      this.request(); // pull the next frame
    };
    ws.onerror = () => this.onError("websocket error");
    ws.onclose = () => this.onClose();
  }

  request() {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify(this.nextAction()));
  }

  stop() {
    this.ws?.close();
    this.ws = null;
  }
}