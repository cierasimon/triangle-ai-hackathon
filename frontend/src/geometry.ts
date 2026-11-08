// Ramp geometry in meters (y up), shared by the canvas and the drag preview.
import type { Frame, OtherBall, Point, Scene } from "./types";

export const ANGLE_MIN = 5;
export const ANGLE_MAX = 75;
export const RAMP_MIN_M = 0.2; // same limits as backend/app/scene/builder.py
export const RAMP_MAX_M = 5;

export interface Ramp {
  bottom: Point;
  side: number; // +1: the top is right of the bottom, -1: left
  fullLength: number; // whole drawn ramp
  travel: number; // ball contact point → bottom
  radius: number;
}

export function rampOf(scene: Scene): Ramp {
  const [tx, ty] = scene.ramp_top_m;
  const [bx, by] = scene.ramp_bottom_m;
  return {
    bottom: [bx, by],
    side: tx >= bx ? 1 : -1,
    fullLength: Math.hypot(tx - bx, ty - by),
    travel: scene.ramp_length_m,
    radius: scene.ball_radius_m,
  };
}

const rad = (deg: number) => (deg * Math.PI) / 180;

export function topAt(r: Ramp, angleDeg: number): Point {
  const a = rad(angleDeg);
  return [r.bottom[0] + r.side * r.fullLength * Math.cos(a), r.bottom[1] + r.fullLength * Math.sin(a)];
}

/** The same ramp stretched to a new whole length; the ball keeps its place on it. */
export function withLength(r: Ramp, fullLength: number): Ramp {
  return { ...r, fullLength, travel: (r.travel * fullLength) / r.fullLength };
}

/** Ball centre resting on the ramp, `travel` meters up from the bottom (default: its start). */
export function ballStartAt(r: Ramp, angleDeg: number, travel = r.travel, radius = r.radius): Point {
  const a = rad(angleDeg);
  const cx = r.bottom[0] + r.side * travel * Math.cos(a);
  const cy = r.bottom[1] + travel * Math.sin(a);
  // outward normal: up and away from the triangle under the ramp
  return [cx - r.side * Math.sin(a) * radius, cy + Math.cos(a) * radius];
}

/** Where another ball rests for this ramp (mirrors other_center in scene/builder.py). */
export function otherRestAt(r: Ramp, angleDeg: number, o: OtherBall): Point {
  if (o.surface === "ground") return [o.at, r.bottom[1] + o.radius_m];
  return ballStartAt(r, angleDeg, o.at * r.fullLength, o.radius_m);
}

/** Whole ramp length from the fixed bottom to point p, clamped to the allowed range. */
export function lengthTo(r: Ramp, p: Point): number {
  const len = Math.hypot(p[0] - r.bottom[0], p[1] - r.bottom[1]);
  return Math.min(RAMP_MAX_M, Math.max(RAMP_MIN_M, len));
}

/** Angle (deg) of a ramp from the fixed bottom to point p, clamped to the allowed range. */
export function angleTo(r: Ramp, p: Point, snap = false): number {
  const dx = r.side * (p[0] - r.bottom[0]);
  const dy = p[1] - r.bottom[1];
  let deg = (Math.atan2(dy, Math.max(dx, 1e-6)) * 180) / Math.PI;
  if (snap) deg = Math.round(deg / 5) * 5;
  return Math.min(ANGLE_MAX, Math.max(ANGLE_MIN, deg));
}

/** Where on the ramp (0 = bottom, 1 = top) point p projects to, for a given angle. */
export function fractionAt(r: Ramp, angleDeg: number, p: Point): number {
  const a = rad(angleDeg);
  const along = r.side * (p[0] - r.bottom[0]) * Math.cos(a) + (p[1] - r.bottom[1]) * Math.sin(a);
  return Math.min(1, Math.max(0, along / r.fullLength));
}

/** Fixed view in meters covering every angle the ramp can be dragged to. */
export function viewOf(r: Ramp, groundY: number, others: OtherBall[] = [], groundEnd: number | null = null) {
  const pad = 0.12 * r.fullLength + 2 * r.radius;
  // the whole flat ground up to its stop, so the balls never leave the view
  const xs = [r.bottom[0], r.bottom[0] + r.side * r.fullLength, ...others.map((o) => o.center_m[0])];
  if (groundEnd != null) xs.push(groundEnd);
  return {
    minX: Math.min(...xs) - pad,
    maxX: Math.max(...xs) + pad,
    minY: Math.min(groundY, r.bottom[1]) - pad - 0.2 * r.fullLength, // room for the weight arrow
    maxY: r.bottom[1] + r.fullLength * Math.sin(rad(ANGLE_MAX)) + pad,
  };
}

/** The ball's state at time t, linearly interpolated between recorded frames (smooth in slow motion). */
export function frameAt(frames: Frame[], t: number): Frame | undefined {
  if (!frames.length) return undefined;
  const last = frames[frames.length - 1];
  if (t >= last.t) return last;
  if (t <= frames[0].t) return frames[0];
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t < t) lo = mid + 1;
    else hi = mid;
  }
  const a = frames[lo - 1];
  const b = frames[lo];
  const k = (t - a.t) / (b.t - a.t || 1);
  const mix = (p: number, q: number) => p + (q - p) * k;
  return {
    t,
    x: mix(a.x, b.x),
    y: mix(a.y, b.y),
    angle: mix(a.angle, b.angle),
    others: (b.others ?? []).map((o, i) => {
      const prev = a.others?.[i] ?? o;
      return [mix(prev[0], o[0]), mix(prev[1], o[1]), mix(prev[2], o[2])] as [number, number, number];
    }),
  };
}

export const duration = (frames: Frame[]) => (frames.length ? frames[frames.length - 1].t : 0);

const posOf = (f: Frame, i: number): number[] | undefined => (i < 0 ? [f.x, f.y] : f.others?.[i]);

/** Velocity (m/s) at recorded frame k: centred difference of its neighbours. */
function velocityAtFrame(frames: Frame[], k: number, i: number): Point {
  const a = frames[Math.max(0, k - 1)];
  const b = frames[Math.min(frames.length - 1, k + 1)];
  const pa = posOf(a, i);
  const pb = posOf(b, i);
  const dt = b.t - a.t;
  return pa && pb && dt > 0 ? [(pb[0] - pa[0]) / dt, (pb[1] - pa[1]) / dt] : [0, 0];
}

/** Velocity (m/s, y up) at time t of the main ball, or other ball `i`: blended between the
 * frames around t like the position, so the arrows change smoothly in slow motion. */
export function velocityAt(frames: Frame[], t: number, i = -1): Point {
  if (frames.length < 2) return [0, 0];
  let j = frames.findIndex((f) => f.t >= t);
  if (j <= 0) j = j < 0 ? frames.length - 1 : 1;
  const a = frames[j - 1];
  const b = frames[j];
  const k = Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t || 1)));
  const va = velocityAtFrame(frames, j - 1, i);
  const vb = velocityAtFrame(frames, j, i);
  return [va[0] + (vb[0] - va[0]) * k, va[1] + (vb[1] - va[1]) * k];
}
