"""What each setup can report, with LaTeX and the engine's numbers (P3.7).

Fed to the TA context so it can name formulas and quote values without computing anything.
"""

from app.contracts import Run

FORMULAS = {
    "slide": {
        "acceleration": r"a = g(\sin\theta - \mu\cos\theta)",
        "time": r"t = \sqrt{\dfrac{2L}{a}}",
        "final_speed": r"v = a\,t = \sqrt{2aL}",
        "moves_if": r"\tan\theta > \mu",
    },
    "roll": {
        "acceleration": r"a = \tfrac{5}{7}\, g\sin\theta",
        "time": r"t = \sqrt{\dfrac{2L}{a}}",
        "final_speed": r"v = a\,t = \sqrt{2aL}",
        "moves_if": r"\theta > 0",
    },
}
NOTES = {
    "mass": "Mass does not appear in any formula: it changes nothing about the motion.",
    "ratio": "Changing one input scales t by the ratio of the formula factor that holds it, "
             "e.g. t ∝ 1/√g, so Moon/Earth time = √(9.81/1.62).",
    "roll": "Rolling (solid sphere) puts 2/7 of the energy into spin, so a = 5/7 g sin θ.",
}


def _r(v: float | None, digits: int = 3) -> float | None:
    return None if v is None else round(v, digits)


def table_for(run: Run) -> dict:
    """Formulas for the run's motion model and the engine's values for this run."""
    v = run.verification
    t = v.simulated_time_s if v.simulated_time_s is not None else v.predicted_time_s
    speed = v.simulated_speed_m_s if v.simulated_speed_m_s is not None else v.predicted_speed_m_s
    accel = (v.predicted_speed_m_s / v.predicted_time_s
             if v.predicted_time_s and v.predicted_speed_m_s else None)
    return {
        "formulas": FORMULAS[run.scene.params.motion],
        "values": {
            "acceleration_m_s2": _r(accel),
            "time_s": _r(t),
            "final_speed_m_s": _r(speed),
            "distance_L_m": _r(run.scene.ramp_length_m),
        },
    }
