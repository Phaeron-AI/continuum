import { useEffect, useRef, useState } from "react";

import { Grid2D } from "./grid2d";
import { closeSession, FrameStream, openSession } from "./player";

const SERVER = import.meta.env.VITE_SERVER ?? "http://localhost:8000";

// Grid2D action ids (engine envs/base.py): UP=0 DOWN=1 LEFT=2 RIGHT=3 INTERACT=4.
const KEY_TO_ACTION = {
  ArrowUp: 0, KeyW: 0,
  ArrowDown: 1, KeyS: 1,
  ArrowLeft: 2, KeyA: 2,
  ArrowRight: 3, KeyD: 3,
};
const NOOP = 4;

export default function App() {
  const realRef = useRef(null); // left canvas: the real game engine
  const neuralRef = useRef(null); // right canvas: the neural model
  const envRef = useRef(null);
  const actionRef = useRef(NOOP);
  const heldRef = useRef(new Set());
  const tempRef = useRef(0.9);
  const streamRef = useRef(null);
  const sessionIdRef = useRef(null);
  const fpsMeter = useRef({ frames: 0, t0: performance.now() });

  const [playing, setPlaying] = useState(false);
  const [status, setStatus] = useState("press start");
  const [fps, setFps] = useState(0);
  const [temp, setTemp] = useState(0.9);

  useEffect(() => {
    const down = (e) => {
      if (e.code in KEY_TO_ACTION) {
        heldRef.current.add(e.code);
        actionRef.current = KEY_TO_ACTION[e.code];
        e.preventDefault();
      }
    };
    const up = (e) => {
      if (e.code in KEY_TO_ACTION) {
        heldRef.current.delete(e.code);
        const last = [...heldRef.current].pop();
        actionRef.current = last === undefined ? NOOP : KEY_TO_ACTION[last];
      }
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  const drawNeural = (bitmap) => {
    const c = neuralRef.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    ctx.drawImage(bitmap, 0, 0, c.width, c.height);
    bitmap.close();
    const m = fpsMeter.current;
    m.frames += 1;
    const dt = performance.now() - m.t0;
    if (dt >= 500) {
      setFps((1000 * m.frames) / dt);
      m.frames = 0;
      m.t0 = performance.now();
    }
  };

  const captureSeed = () =>
    new Promise((resolve, reject) => {
      const c = realRef.current;
      if (!c) return reject(new Error("no canvas"));
      c.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob failed"))), "image/png");
    });

  const start = async () => {
    setStatus("opening…");
    try {
      // The real game generates the starting frame; that same frame seeds the
      // neural model, so both begin pixel-identical.
      const env = new Grid2D();
      envRef.current = env;
      env.render(realRef.current.getContext("2d"));

      const seed = await captureSeed();
      const id = await openSession(SERVER, [seed]);
      sessionIdRef.current = id;

      const stream = new FrameStream(
        SERVER,
        id,
        () => {
          // Advance the real engine with the same action, in lockstep.
          const a = actionRef.current;
          env.step(a);
          env.render(realRef.current.getContext("2d"));
          return { action: a, temperature: tempRef.current };
        },
        drawNeural,
        (msg) => setStatus(msg),
        () => setPlaying(false),
      );
      streamRef.current = stream;
      stream.start();
      setPlaying(true);
      setStatus("playing — click the page, then use WASD");
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err));
    }
  };

  const stop = () => {
    streamRef.current?.stop();
    streamRef.current = null;
    if (sessionIdRef.current) {
      closeSession(SERVER, sessionIdRef.current);
      sessionIdRef.current = null;
    }
    setPlaying(false);
    setStatus("stopped");
  };

  const newWorld = () => {
    stop();
    setTimeout(start, 60);
  };

  const onTemp = (e) => {
    const v = Number(e.target.value);
    setTemp(v);
    tempRef.current = v;
  };

  return (
    <main>
      <header>
        <h1>continuum</h1>
        <p className="tagline">A neural network dreaming a video game, in real time.</p>
      </header>

      <div className="stage">
        <figure>
          <canvas ref={realRef} width={84} height={84} className="screen" />
          <figcaption>
            <b>real game engine</b>
            <span>deterministic code</span>
          </figcaption>
        </figure>

        <div className="vs">vs</div>

        <figure>
          <canvas ref={neuralRef} width={64} height={64} className="screen neural" />
          <figcaption>
            <b>neural network</b>
            <span>no game code — every pixel generated</span>
          </figcaption>
        </figure>
      </div>

      <div className="controls">
        {!playing ? (
          <button onClick={start}>start</button>
        ) : (
          <>
            <button onClick={stop}>stop</button>
            <button className="ghost" onClick={newWorld}>
              new world
            </button>
          </>
        )}
        <label>
          temperature{" "}
          <input type="range" min={0} max={1.5} step={0.05} value={temp} onChange={onTemp} />
        </label>
        <span className="stat">{fps.toFixed(0)} fps</span>
        <span className="stat">{status}</span>
      </div>

      <p className="explainer">
        Both panels start from the <b>same frame</b> and obey the <b>same keys</b> (WASD). The
        left is the real game, running actual code. The right is a neural network that learned to{" "}
        <b>predict each next frame from scratch</b> — there is no game engine behind it. Watch it
        track the real game, then slowly dream its own version.
      </p>
    </main>
  );
}