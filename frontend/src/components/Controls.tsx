import { ANGLE_MAX, ANGLE_MIN, RAMP_MAX_M, RAMP_MIN_M } from "../geometry";

export const MASS_MIN = 0.1;
export const MASS_MAX = 10;
export const MAX_SPEED = 10;

interface Props {
  angle: number;
  length: number;
  mass: number;
  otherMass: number | null; // null: there is no second ball
  friction: number;
  gravity: number;
  bodies: BodyControl[]; // every object in the picture: A, B, C, ...
  tags: Record<string, string>; // where each value came from: "from drawing" | "default" | "changed"
  disabled: boolean;
  onAngle: (deg: number) => void; // each fires while sliding
  onLength: (m: number) => void;
  onMass: (kg: number) => void;
  onOtherMass: (kg: number) => void;
  onFriction: (mu: number) => void;
  onGravity: (g: number) => void;
  onSpeed: (body: number, v: number) => void; // body 0 is A
}

/** One object: its letter, initial speed (signed) and the surface that gives the sign meaning. */
export interface BodyControl {
  name: string;
  speed: number;
  surface: "ramp" | "ground";
  tag: string;
}

export const letter = (i: number) => String.fromCharCode(65 + i);

function speedText(b: BodyControl) {
  if (b.speed === 0) return "0 m/s · at rest";
  const v = `${Math.abs(b.speed).toFixed(1)} m/s`;
  const dir = b.surface === "ramp"
    ? (b.speed > 0 ? "↓ down the ramp" : "↑ up the ramp")
    : (b.speed > 0 ? "→ right" : "← left");
  return `${v} ${dir}`;
}

function Slider({ label, value, text, min, max, step, disabled, onChange, tag }: {
  label: string; value: number; text: string; min: number; max: number; step: number;
  disabled: boolean; onChange: (v: number) => void; tag?: string;
}) {
  return (
    <label className="slider">
      <span>{label}{tag && <small className={`tag ${tag.replace(" ", "-")}`}>{tag}</small>}</span>
      <output>{text}</output>
      <input type="range" min={min} max={max} step={step} value={value}
             disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

/** Sliders for the ramp (angle, length) and the balls' masses. */
export function Controls({
  angle, length, mass, otherMass, friction, gravity, bodies, tags, disabled,
  onAngle, onLength, onMass, onOtherMass, onFriction, onGravity, onSpeed,
}: Props) {
  const two = otherMass != null;

  return (
    <div className="controls">
      <Slider label="Angle θ" value={Math.round(angle)} text={`${Math.round(angle)} °`}
              min={ANGLE_MIN} max={ANGLE_MAX} step={1} disabled={disabled} onChange={onAngle} tag={tags.angle_deg} />
      <Slider label="Length L" value={length} text={`${length.toFixed(2)} m`}
              min={RAMP_MIN_M} max={RAMP_MAX_M} step={0.05} disabled={disabled} onChange={onLength} />
      <Slider label={two ? `Mass ${letter(0)}` : "Mass"} value={mass} text={`${mass.toFixed(1)} kg`}
              min={MASS_MIN} max={MASS_MAX} step={0.1} disabled={disabled} onChange={onMass} tag={tags.mass_kg} />
      {bodies.map((b, i) => (
        <Slider key={b.name} label={`Speed ${b.name}`} value={b.speed} text={speedText(b)}
                min={-MAX_SPEED} max={MAX_SPEED} step={0.1} disabled={disabled}
                onChange={(v) => onSpeed(i, v)} tag={b.tag} />
      ))}
      <Slider label="Friction μ" value={friction} text={friction.toFixed(2)}
              min={0} max={1.5} step={0.01} disabled={disabled} onChange={onFriction} tag={tags.friction} />
      <Slider label="Gravity g" value={gravity} text={`${gravity.toFixed(2)} m/s²`}
              min={0.5} max={25} step={0.1} disabled={disabled} onChange={onGravity} tag={tags.gravity} />
      {two && (
        <Slider label={`Mass ${bodies.slice(1).map((b) => b.name).join(", ")}`} value={otherMass} text={`${otherMass.toFixed(1)} kg`}
                min={MASS_MIN} max={MASS_MAX} step={0.1} disabled={disabled} onChange={onOtherMass} />
      )}
    </div>
  );
}
