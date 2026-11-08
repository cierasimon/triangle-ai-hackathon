"""Textbook formulas (SymPy) for a ball released from rest on a ramp, and verification.

Slide:        a = g (sin θ − μ cos θ)      (moves only if tan θ > μ)
Roll (sphere): a = (5/7) g sin θ            (see physics/simulate.py for why sphere)
Roll (disk):   a = (2/3) g sin θ            (kept for reference; pymunk's default circle)
t = √(2L / a),  v = a t.  Mass m appears nowhere: ∂t/∂m = 0.

With an initial speed v0 (+ down the ramp): s = v0 t + a t² / 2, so t = (√(v0² + 2aL) − v0) / a and
v = √(v0² + 2aL). Thrown up the ramp (v0 < 0) it first slows with a_up = g(sin θ + μ cos θ) over
d = v0² / (2 a_up), then comes back down from rest over the extra distance d + L.
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
            motion: Literal["slide", "roll"] = "slide", mass_kg: float = 1.0, v0: float = 0.0,
            room_up_m: float | None = None) -> Prediction:
    th = math.radians(angle_deg)
    if v0 != 0:
        return _predict_moving(th, length_m, gravity, friction, motion, v0, room_up_m)
    if motion == "slide" and friction >= math.tan(th):
        return Prediction(None, None, False, "ball will not slide: μ ≥ tan θ")
    a = _accel_fn[motion](th, friction, gravity, mass_kg, length_m)
    if a <= 0 or length_m <= 0:
        return Prediction(None, None, False, "ball does not move")
    t = math.sqrt(2 * length_m / a)
    return Prediction(t, a * t, True, None)


def _predict_moving(th: float, length_m: float, gravity: float, friction: float, motion: str,
                    v0: float, room_up_m: float | None) -> Prediction:
    """Released with an initial speed along the ramp (v0 > 0 down, v0 < 0 up). A rolling ball is
    assumed to start rolling without slipping at v0."""
    if length_m <= 0:
        return Prediction(None, None, False, "ball does not move")
    s, c = math.sin(th), math.cos(th)
    if motion == "roll":
        a_down = a_up = 5 / 7 * gravity * s
    else:
        a_down, a_up = gravity * (s - friction * c), gravity * (s + friction * c)
    if v0 > 0:
        disc = v0 * v0 + 2 * a_down * length_m
        if disc <= 0:
            return Prediction(None, None, False, "ball stops before reaching the bottom")
        v = math.sqrt(disc)
        t = (v - v0) / a_down if abs(a_down) > 1e-12 else length_m / v0
        return Prediction(t, v, True, None)
    if a_up <= 1e-12:
        return Prediction(None, None, False, "ball keeps moving away up the ramp")
    d = v0 * v0 / (2 * a_up)
    if room_up_m is not None and d > room_up_m:
        return Prediction(None, None, False, "ball runs off the top of the ramp")
    if a_down <= 1e-12:
        return Prediction(None, None, False, "ball stops on the ramp")
    t_down = math.sqrt(2 * (d + length_m) / a_down)
    return Prediction(-v0 / a_up + t_down, a_down * t_down, True, None)


def latex_for(motion: str, v0: float) -> str:
    if v0 == 0:
        return LATEX[motion]
    a = r"\tfrac{5}{7}\, g\sin\theta" if motion == "roll" else r"g(\sin\theta - \mu\cos\theta)"
    if v0 > 0:
        return r"t = \dfrac{\sqrt{v_0^{2} + 2aL} - v_0}{a},\quad a = " + a
    return r"t = \dfrac{|v_0|}{a_\uparrow} + \sqrt{\dfrac{2(d + L)}{a_\downarrow}},\quad d = \dfrac{v_0^{2}}{2a_\uparrow}"


def verify(scene: Scene, simulated_time_s: float | None,
           simulated_speed_m_s: float | None) -> Verification:
    p = scene.params
    room_up = math.dist(scene.ramp_top_m, scene.ramp_bottom_m) - scene.ramp_length_m
    pred = predict(p.angle_deg, scene.ramp_length_m, p.gravity, p.friction, p.motion, p.mass_kg,
                   p.initial_speed_m_s, room_up)
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
        formula_latex=latex_for(p.motion, p.initial_speed_m_s),
        note=note,
    )
