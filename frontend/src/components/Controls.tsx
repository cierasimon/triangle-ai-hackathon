import { useEffect, useState } from "react";
import { ANGLE_MAX, ANGLE_MIN } from "../geometry";
import type { Params, ParamsOverride } from "../types";

interface Props {
  params: Params;
  busy: boolean;
  onChange: (patch: ParamsOverride) => void;
  onResetAngle: () => void;
  onResetAll: () => void;
  angleOverridden: boolean;
}

const GRAVITY = [
  { name: "Earth", g: 9.81 },
  { name: "Moon", g: 1.62 },
  { name: "Mars", g: 3.71 },
];
const FRICTION = [0, 0.1, 0.3, 0.6];
const ANGLES = [15, 30, 45];

const near = (a: number, b: number) => Math.abs(a - b) < 1e-6;

export function Controls({ params, busy, onChange, onResetAngle, onResetAll, angleOverridden }: Props) {
  const [angleText, setAngleText] = useState(String(Math.round(params.angle_deg)));
  useEffect(() => setAngleText(String(Math.round(params.angle_deg))), [params.angle_deg]);

  function submitAngle() {
    const v = Number(angleText);
    if (Number.isFinite(v)) onChange({ angle_deg: Math.min(ANGLE_MAX, Math.max(ANGLE_MIN, v)) });
  }

  const btn = (active: boolean) => (active ? "chip active" : "chip");

  return (
    <div className="controls">
      <div className="control-row">
        <span className="control-label">Angle</span>
        {ANGLES.map((a) => (
          <button key={a} className={btn(near(params.angle_deg, a))} disabled={busy}
                  onClick={() => onChange({ angle_deg: a })}>{a}°</button>
        ))}
        <form className="angle-box" onSubmit={(e) => { e.preventDefault(); submitAngle(); }}>
          <input type="number" inputMode="decimal" min={ANGLE_MIN} max={ANGLE_MAX} value={angleText}
                 aria-label="angle in degrees" onChange={(e) => setAngleText(e.target.value)} />
          <button type="submit" className="chip" disabled={busy}>Set</button>
        </form>
        {angleOverridden && (
          <button className="chip" disabled={busy} onClick={onResetAngle}>Reset angle</button>
        )}
      </div>
      <div className="control-row">
        <span className="control-label">Mass</span>
        <button className="chip" disabled={busy} onClick={() => onChange({ mass_kg: params.mass_kg * 2 })}>
          ×2 mass
        </button>
        <span className="muted">{+params.mass_kg.toFixed(2)} kg</span>
      </div>
      <div className="control-row">
        <span className="control-label">Gravity</span>
        {GRAVITY.map(({ name, g }) => (
          <button key={name} className={btn(near(params.gravity, g))} disabled={busy}
                  onClick={() => onChange({ gravity: g })}>{name}</button>
        ))}
      </div>
      <div className="control-row">
        <span className="control-label">Friction μ</span>
        {FRICTION.map((mu) => (
          <button key={mu} className={btn(near(params.friction, mu))} disabled={busy || params.motion === "roll"}
                  onClick={() => onChange({ friction: mu })}>{mu}</button>
        ))}
      </div>
      <div className="control-row">
        <span className="control-label">Motion</span>
        {(["slide", "roll"] as const).map((m) => (
          <button key={m} className={btn(params.motion === m)} disabled={busy}
                  onClick={() => onChange({ motion: m })}>{m}</button>
        ))}
        <button className="chip ghost" disabled={busy} onClick={onResetAll}>Back to drawing</button>
      </div>
    </div>
  );
}
