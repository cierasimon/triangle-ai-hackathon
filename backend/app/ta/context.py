"""Compact JSON context for the TA (P3.3): the only facts and numbers it may use.

Every number comes from the engine (runs, verification, formula table) or the drawing; the
number check (ta/numcheck.py) accepts an answer's numbers only if they match these.
"""

import itertools
import json
import re

from app import store
from app.contracts import Level, Run
from app.physics.formula_table import NOTES, table_for
from app.physics.whycard import why_card

AVAILABLE_CHANGES = {
    "angle_deg": [15, 30, 45],
    "angle_range_deg": [0, 90],
    "mass": "×2",
    "gravity_m_s2": {"Earth": 9.81, "Moon": 1.62, "Mars": 3.71},
    "friction": [0, 0.1, 0.3, 0.6],
    "initial_speed_m_s": "-10 to 10 (positive = down the ramp, negative = up the ramp)",
    "motion": ["slide", "roll"],
}
OUT_OF_SCOPE = [
    "Blocks, springs, collisions, pendulums: only a ball on a ramp is simulated.",
    "Static friction and air resistance are not modelled.",
]


def _r(v: float | None, digits: int = 3) -> float | None:
    return None if v is None else round(v, digits)


def run_summary(run: Run) -> dict:
    p, v = run.scene.params, run.verification
    return {
        "label": f"Run {run.index}",
        "id": run.run_id,
        "angle_deg": _r(p.angle_deg, 1),
        "mass_kg": _r(p.mass_kg),
        "gravity_m_s2": _r(p.gravity),
        "friction": _r(p.friction),
        "initial_speed_m_s": _r(p.initial_speed_m_s),
        "motion": p.motion,
        "sources": p.sources,
        "distance_L_m": _r(run.scene.ramp_length_m),
        "predicted_time_s": _r(v.predicted_time_s),
        "simulated_time_s": _r(v.simulated_time_s),
        "predicted_speed_m_s": _r(v.predicted_speed_m_s),
        "simulated_speed_m_s": _r(v.simulated_speed_m_s),
        "error_pct": _r(v.error_pct, 2),
        "verified": v.ok,
        "moves": v.predicted_time_s is not None,
        "note": v.note,
        "changed_vs_previous": {k: [_r(a), _r(b)] for k, (a, b) in run.changed.items()},
    }


def comparisons(runs: list[Run]) -> list[dict]:
    out = []
    for a, b in zip(runs, runs[1:]):
        card = why_card(a, b)
        ta = a.verification.simulated_time_s or a.verification.predicted_time_s
        tb = b.verification.simulated_time_s or b.verification.predicted_time_s
        out.append({
            "runs": [f"Run {a.index}", f"Run {b.index}"],
            "time_ratio": _r(tb / ta) if ta and tb else None,
            "explanation": card.sentence,
            "formula": card.latex,
            "cancels": card.cancels,
        })
    return out


def recognition_info(session: store.Session) -> dict | None:
    stored = session.images.get(session.current_image_id or "")
    rec = stored.recognition if stored else None
    if rec is None:
        return None
    first = session.runs[0] if session.runs else None
    return {
        "paper_found": rec.paper_found,
        "ramp_found": any(line.role == "ramp" for line in rec.lines),
        "ground_found": any(line.role == "ground" for line in rec.lines),
        "ball_found": rec.ball is not None,
        "drawn_angle_deg": _r(first.scene.params.angle_deg, 1) if first else None,
        "defaults_used": first.scene.params.defaults_used if first else [],
        "warnings": rec.warnings,
        "labels_read": [{"key": label.key, "value": label.value} for label in rec.labels],
    }


def build_context(session: store.Session, level: Level) -> dict:
    runs = session.runs
    current = runs[-1] if runs else None
    return {
        "level": level,
        "drawing": recognition_info(session),
        "current_run": f"Run {current.index}" if current else None,
        "runs": [run_summary(r) for r in runs],
        "comparisons": comparisons(runs),
        "formula_table": table_for(current) if current else None,
        "notes": NOTES,
        "learner": session.profile.model_dump(),
        "available_changes": AVAILABLE_CHANGES,
        "out_of_scope": OUT_OF_SCOPE,
    }


def to_json(context: dict) -> str:
    return json.dumps(context, ensure_ascii=False, separators=(",", ":"))


PROSE_KEYS = {"explanation", "note", "warnings"}  # free text whose numbers may be quoted


def _walk(value, key: str = "") -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [x for k, v in value.items() for x in _walk(v, k)]
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in _walk(v, key)]
    if isinstance(value, str) and key in PROSE_KEYS:  # "2.46× (0.40 s → 0.97 s)"
        return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", value)]
    return []  # LaTeX, sources, labels: not quotable numbers


def context_numbers(context: dict, runs: list[Run]) -> list[float]:
    """Every number the TA may quote: context values plus ratios between runs."""
    numbers = _walk({k: v for k, v in context.items() if k not in ("learner", "notes")})
    ratios = []
    for a, b in itertools.permutations(runs, 2):
        for key in ("simulated_time_s", "simulated_speed_m_s"):
            va, vb = getattr(a.verification, key), getattr(b.verification, key)
            if va and vb:
                ratios.append(vb / va)
        if a.scene.params.gravity != b.scene.params.gravity:
            ratios.append(b.scene.params.gravity / a.scene.params.gravity)
    # a ratio may also be quoted as a percentage ("2.46×" = "146% longer")
    numbers += ratios + [abs(r - 1) * 100 for r in ratios]
    return numbers
