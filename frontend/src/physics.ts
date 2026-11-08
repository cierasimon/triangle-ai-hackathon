// Browser port of backend/app/physics/formulas.py `predict`, for live updates while dragging.
// Cross-checked against the Python version through predict_golden.json (pytest + vitest).
import type { Motion } from "./types";

export interface Prediction {
  time_s: number | null;
  speed_m_s: number | null;
  will_move: boolean;
  note: string | null;
}

const none = (note: string): Prediction => ({ time_s: null, speed_m_s: null, will_move: false, note });

/** v0: initial speed along the ramp (+ down, - up). roomUp: ramp left above the start (m). */
export function predict(
  angleDeg: number,
  lengthM: number,
  gravity = 9.81,
  friction = 0,
  motion: Motion = "slide",
  v0 = 0,
  roomUp: number | null = null,
): Prediction {
  const th = (angleDeg * Math.PI) / 180;
  if (v0 !== 0) return predictMoving(th, lengthM, gravity, friction, motion, v0, roomUp);
  if (motion === "slide" && friction >= Math.tan(th)) {
    return none("ball will not slide: μ ≥ tan θ");
  }
  const a =
    motion === "roll"
      ? (5 / 7) * gravity * Math.sin(th)
      : gravity * (Math.sin(th) - friction * Math.cos(th));
  if (a <= 0 || lengthM <= 0) return none("ball does not move");
  const t = Math.sqrt((2 * lengthM) / a);
  return { time_s: t, speed_m_s: a * t, will_move: true, note: null };
}

function predictMoving(
  th: number, lengthM: number, gravity: number, friction: number, motion: Motion, v0: number,
  roomUp: number | null,
): Prediction {
  if (lengthM <= 0) return none("ball does not move");
  const s = Math.sin(th);
  const c = Math.cos(th);
  const aDown = motion === "roll" ? (5 / 7) * gravity * s : gravity * (s - friction * c);
  const aUp = motion === "roll" ? (5 / 7) * gravity * s : gravity * (s + friction * c);
  if (v0 > 0) {
    const disc = v0 * v0 + 2 * aDown * lengthM;
    if (disc <= 0) return none("ball stops before reaching the bottom");
    const v = Math.sqrt(disc);
    const t = Math.abs(aDown) > 1e-12 ? (v - v0) / aDown : lengthM / v0;
    return { time_s: t, speed_m_s: v, will_move: true, note: null };
  }
  if (aUp <= 1e-12) return none("ball keeps moving away up the ramp");
  const d = (v0 * v0) / (2 * aUp);
  if (roomUp !== null && d > roomUp) return none("ball runs off the top of the ramp");
  if (aDown <= 1e-12) return none("ball stops on the ramp");
  const tDown = Math.sqrt((2 * (d + lengthM)) / aDown);
  return { time_s: -v0 / aUp + tDown, speed_m_s: aDown * tDown, will_move: true, note: null };
}
