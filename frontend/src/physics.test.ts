import { describe, expect, it } from "vitest";
import golden from "./predict_golden.json";
import { predict } from "./physics";
import type { Motion } from "./types";

describe("predict() matches the Python formulas", () => {
  for (const c of golden) {
    it(`${c.motion} θ=${c.angle_deg} μ=${c.friction} g=${c.gravity} L=${c.length_m} v0=${c.v0}`, () => {
      const p = predict(c.angle_deg, c.length_m, c.gravity, c.friction, c.motion as Motion, c.v0, 1.0);
      expect(p.will_move).toBe(c.will_move);
      if (c.time_s === null) {
        expect(p.time_s).toBeNull();
      } else {
        expect(p.time_s).toBeCloseTo(c.time_s, 9);
        expect(p.speed_m_s).toBeCloseTo(c.speed_m_s as number, 9);
      }
    });
  }
});
