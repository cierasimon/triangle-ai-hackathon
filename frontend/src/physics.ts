// Browser port of backend/app/physics/formulas.py `predict`, for live updates while dragging.
// Cross-checked against the Python version through predict_golden.json (pytest + vitest).
import type { Motion } from "./types";

export interface Prediction {
  time_s: number | null;
  speed_m_s: number | null;
  will_move: boolean;
  note: string | null;
}

export function predict(
  angleDeg: number,
  lengthM: number,
  gravity = 9.81,
  friction = 0,
  motion: Motion = "slide",
): Prediction {
  const th = (angleDeg * Math.PI) / 180;
  if (motion === "slide" && friction >= Math.tan(th)) {
    return { time_s: null, speed_m_s: null, will_move: false, note: "ball will not slide: μ ≥ tan θ" };
  }
  const a =
    motion === "roll"
      ? (5 / 7) * gravity * Math.sin(th)
      : gravity * (Math.sin(th) - friction * Math.cos(th));
  if (a <= 0 || lengthM <= 0) {
    return { time_s: null, speed_m_s: null, will_move: false, note: "ball does not move" };
  }
  const t = Math.sqrt((2 * lengthM) / a);
  return { time_s: t, speed_m_s: a * t, will_move: true, note: null };
}
