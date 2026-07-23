// Browser reimplementation of the grid2d environment — the "real game engine"
// side of the side-by-side. Faithful to engine/src/envs/grid2d: dark
// background, blue agent circle, red obstacle circles; move 3px/step, clamp to
// bounds, blocked when within (agent_radius + obstacle_radius) of an obstacle.
// This is deterministic game code — no neural network involved.

const BG = "rgb(20,20,20)";
const AGENT = "rgb(66,135,245)";
const OBSTACLE = "rgb(200,60,60)";

// Action ids match engine envs/base.py: UP=0 DOWN=1 LEFT=2 RIGHT=3 INTERACT=4.
const MOVES = { 0: [0, -1], 1: [0, 1], 2: [-1, 0], 3: [1, 0] };

export class Grid2D {
  constructor({
    canvasSize = 84,
    agentRadius = 3,
    obstacleRadius = 3,
    moveStep = 3,
    numObstacles = 5,
  } = {}) {
    this.size = canvasSize;
    this.ar = agentRadius;
    this.or = obstacleRadius;
    this.moveStep = moveStep;
    this.n = numObstacles;
    this.reset();
  }

  _rand(lo, hi) {
    return lo + Math.random() * (hi - lo);
  }

  reset() {
    const s = this.size;
    this.agent = [this._rand(this.ar, s - this.ar), this._rand(this.ar, s - this.ar)];
    this.obstacles = [];
    for (let i = 0; i < this.n; i++) {
      this.obstacles.push([
        this._rand(this.or, s - this.or),
        this._rand(this.or, s - this.or),
      ]);
    }
  }

  _collides(p) {
    const threshold = this.ar + this.or;
    return this.obstacles.some((o) => Math.hypot(p[0] - o[0], p[1] - o[1]) < threshold);
  }

  step(action) {
    const mv = MOVES[action];
    if (!mv) return; // INTERACT / unknown -> stay put
    const s = this.size;
    const r = this.ar;
    let p = [this.agent[0] + mv[0] * this.moveStep, this.agent[1] + mv[1] * this.moveStep];
    p = [Math.min(Math.max(p[0], r), s - r), Math.min(Math.max(p[1], r), s - r)];
    if (!this._collides(p)) this.agent = p;
  }

  render(ctx) {
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, this.size, this.size);
    ctx.fillStyle = OBSTACLE;
    for (const o of this.obstacles) {
      ctx.beginPath();
      ctx.arc(o[0], o[1], this.or, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.fillStyle = AGENT;
    ctx.beginPath();
    ctx.arc(this.agent[0], this.agent[1], this.ar, 0, 2 * Math.PI);
    ctx.fill();
  }
}