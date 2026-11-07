"""Textbook formulas (SymPy) for a ball released from rest on a ramp, and verification.

Slide:        a = g (sin θ − μ cos θ)      (moves only if tan θ > μ)
Roll (sphere): a = (5/7) g sin θ            (see physics/simulate.py for why sphere)
Roll (disk):   a = (2/3) g sin θ            (kept for reference; pymunk's default circle)
t = √(2L / a),  v = a t.  Mass m appears nowhere: ∂t/∂m = 0.
"""

import math
from dataclasses import dataclass
from typing import Literal

import sympy as sp

from app.contracts import Scene, Verification

theta, mu, g, m, L = sp.symbols("theta mu g m L", positive=True)

ACCEL = {
    "slide": g * (sp.sin(theta) - mu * sp.cos(theta)),
    "roll": sp.Rational(5, 7) * g * sp.sin(theta),
    "roll_disk": sp.Rational(2, 3) * g * sp.sin(theta),
}
TIME = {k: sp.sqrt(2 * L / a) for k, a in ACCEL.items()}
SPEED = {k: sp.sqrt(2 * L * a) for k, a in ACCEL.items()}  # = a·t

LATEX = {
    "slide": r"t = \sqrt{\dfrac{2L}{g(\sin\theta - \mu\cos\theta)}}",
    "roll": r"t = \sqrt{\dfrac{2L}{\tfrac{5}{7}\, g\sin\theta}}",
    "roll_disk": r"t = \sqrt{\dfrac{2L}{\tfrac{2}{3}\, g\sin\theta}}",
}

# plain-float versions for /predict (math module, no numpy: fastest for scalars)
_accel_fn = {k: sp.lambdify((theta, mu, g, m, L), a, "math") for k, a in ACCEL.items()}

ERROR_TOLERANCE_PCT = 3.0
SIM_TIME_LIMIT_S = 10.0


@dataclass
class Prediction:
    time_s: float | None
    speed_m_s: float | None
    will_move: bool
    note: str | None


def predict(angle_deg: float, length_m: float, gravity: float = 9.81, friction: float = 0.0,
            motion: Literal["slide", "roll"] = "slide", mass_kg: float = 1.0) -> Prediction:
    th = math.radians(angle_deg)
    if motion == "slide" and friction >= math.tan(th):
        return Prediction(None, None, False, "ball will not slide: μ ≥ tan θ")
    a = _accel_fn[motion](th, friction, gravity, mass_kg, length_m)
    if a <= 0 or length_m <= 0:
        return Prediction(None, None, False, "ball does not move")
    t = math.sqrt(2 * length_m / a)
    return Prediction(t, a * t, True, None)


def verify(scene: Scene, simulated_time_s: float | None,
           simulated_speed_m_s: float | None) -> Verification:
    p = scene.params
    pred = predict(p.angle_deg, scene.ramp_length_m, p.gravity, p.friction, p.motion, p.mass_kg)
    note = pred.note
    error = None
    if pred.will_move and simulated_time_s is not None:
        error = abs(simulated_time_s - pred.time_s) / pred.time_s * 100
        ok = error <= ERROR_TOLERANCE_PCT
    elif not pred.will_move:
        ok = simulated_time_s is None  # both agree it stays put
    else:
        ok = pred.time_s > SIM_TIME_LIMIT_S  # simulation stopped at its time limit
        note = f"takes longer than {SIM_TIME_LIMIT_S:g} s; simulation stopped"
    return Verification(
        predicted_time_s=pred.time_s,
        simulated_time_s=simulated_time_s,
        predicted_speed_m_s=pred.speed_m_s,
        simulated_speed_m_s=simulated_speed_m_s,
        error_pct=error,
        ok=ok,
        formula_latex=LATEX[p.motion],
        note=note,
    )
