// Ramp geometry in meters (y up), shared by the canvas and the drag preview.
import type { Frame, Point, Scene } from "./types";

export const ANGLE_MIN = 5;
export const ANGLE_MAX = 75;

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

/** Ball centre resting on the ramp at its start position, for a given angle. */
export function ballStartAt(r: Ramp, angleDeg: number): Point {
  const a = rad(angleDeg);
  const cx = r.bottom[0] + r.side * r.travel * Math.cos(a);
  const cy = r.bottom[1] + r.travel * Math.sin(a);
  // outward normal: up and away from the triangle under the ramp
  return [cx - r.side * Math.sin(a) * r.radius, cy + Math.cos(a) * r.radius];
}

/** Angle (deg) of a ramp from the fixed bottom to point p, clamped to the allowed range. */
export function angleTo(r: Ramp, p: Point, snap = false): number {
  const dx = r.side * (p[0] - r.bottom[0]);
  const dy = p[1] - r.bottom[1];
  let deg = (Math.atan2(dy, Math.max(dx, 1e-6)) * 180) / Math.PI;
  if (snap) deg = Math.round(deg / 5) * 5;
  return Math.min(ANGLE_MAX, Math.max(ANGLE_MIN, deg));
}

/** Fixed view in meters covering every angle the ramp can be dragged to. */
export function viewOf(r: Ramp, groundY: number) {
  const pad = 0.12 * r.fullLength + 2 * r.radius;
  const xs = [r.bottom[0], r.bottom[0] + r.side * r.fullLength];
  return {
    minX: Math.min(...xs) - pad,
    maxX: Math.max(...xs) + pad,
    minY: Math.min(groundY, r.bottom[1]) - pad,
    maxY: r.bottom[1] + r.fullLength * Math.sin(rad(ANGLE_MAX)) + pad,
  };
}

export function frameAt(frames: Frame[], t: number): Frame | undefined {
  if (!frames.length) return undefined;
  let lo = 0;
  let hi = frames.length - 1;
  if (t >= frames[hi].t) return frames[hi];
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t < t) lo = mid + 1;
    else hi = mid;
  }
  return frames[lo];
}

export const duration = (frames: Frame[]) => (frames.length ? frames[frames.length - 1].t : 0);
