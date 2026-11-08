import { ANGLE_MAX, ANGLE_MIN, RAMP_MAX_M, RAMP_MIN_M } from "../geometry";

export const MASS_MIN = 0.1;
export const MASS_MAX = 10;

interface Props {
  angle: number;
  length: number;
  mass: number;
  otherMass: number | null; // null: there is no second ball
  disabled: boolean;
  onAngle: (deg: number) => void; // each fires while sliding
  onLength: (m: number) => void;
  onMass: (kg: number) => void;
  onOtherMass: (kg: number) => void;
}

function Slider({ label, value, text, min, max, step, disabled, onChange }: {
  label: string; value: number; text: string; min: number; max: number; step: number;
  disabled: boolean; onChange: (v: number) => void;
}) {
  return (
    <label className="slider">
      <span>{label}</span>
      <output>{text}</output>
      <input type="range" min={min} max={max} step={step} value={value}
             disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

/** Sliders for the ramp (angle, length) and the balls' masses. */
export function Controls({ angle, length, mass, otherMass, disabled, onAngle, onLength, onMass, onOtherMass }: Props) {
  const two = otherMass != null;
  return (
    <div className="controls">
      <Slider label="Angle θ" value={Math.round(angle)} text={`${Math.round(angle)} °`}
              min={ANGLE_MIN} max={ANGLE_MAX} step={1} disabled={disabled} onChange={onAngle} />
      <Slider label="Length L" value={length} text={`${length.toFixed(2)} m`}
              min={RAMP_MIN_M} max={RAMP_MAX_M} step={0.05} disabled={disabled} onChange={onLength} />
      <Slider label={two ? "Mass m₁" : "Mass m"} value={mass} text={`${mass.toFixed(1)} kg`}
              min={MASS_MIN} max={MASS_MAX} step={0.1} disabled={disabled} onChange={onMass} />
      {two && (
        <Slider label="Mass m₂" value={otherMass} text={`${otherMass.toFixed(1)} kg`}
                min={MASS_MIN} max={MASS_MAX} step={0.1} disabled={disabled} onChange={onOtherMass} />
      )}
    </div>
  );
}
