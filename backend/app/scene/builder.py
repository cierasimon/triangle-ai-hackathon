"""Recognition (warped-paper pixels, y down) → Scene (meters, y up).

Paper short side = 1 m. Ramp top = higher endpoint. The ball starts where its centre projects
onto the ramp, resting on it; ramp_length_m is from that contact point to the ramp bottom.
Params: angle from the drawing, handwritten labels override it, user overrides win; anything
missing is a default and listed in defaults_used.
"""

import math
from dataclasses import dataclass, field

from app.contracts import Params, ParamsOverride, Recognition, Scene

ANGLE_MIN, ANGLE_MAX = 5.0, 75.0
DEFAULTS = {"mass_kg": 1.0, "gravity": 9.81, "friction": 0.0, "motion": "slide"}
DEFAULT_BALL_RADIUS_M = 0.03
DEFAULT_START_FRACTION = 0.1  # no ball drawn: start 10% down from the ramp top
MIN_TRAVEL_M = 0.05


class SceneError(ValueError):
    """The recognition can't be turned into a scene (e.g. no ramp)."""


@dataclass
class BuildResult:
    scene: Scene
    warnings: list[str] = field(default_factory=list)


def drawn_angle_deg(rec: Recognition) -> float:
    ramp = _ramp(rec)
    dx, dy = ramp.p2[0] - ramp.p1[0], ramp.p2[1] - ramp.p1[1]
    return math.degrees(math.atan2(abs(dy), abs(dx)))


def _ramp(rec: Recognition):
    ramp = next((line for line in rec.lines if line.role == "ramp"), None)
    if ramp is None:
        raise SceneError("no ramp found — draw a straight slanted line")
    return ramp


def build_scene(rec: Recognition, override: ParamsOverride | None = None) -> BuildResult:
    warnings: list[str] = []
    w, h = rec.warped_size
    scale = 1.0 / min(w, h)

    def to_m(p):
        return (p[0] * scale, (h - p[1]) * scale)

    ramp = _ramp(rec)
    a, b = to_m(ramp.p1), to_m(ramp.p2)
    top, bottom = (a, b) if a[1] >= b[1] else (b, a)
    full_len = math.dist(top, bottom)
    down = ((bottom[0] - top[0]) / full_len, (bottom[1] - top[1]) / full_len)

    # ball → distance of its contact point from the bottom, measured along the ramp
    defaults_used: list[str] = []
    if rec.ball is not None:
        c = to_m(rec.ball.center)
        radius = rec.ball.radius * scale
        along = (c[0] - top[0]) * down[0] + (c[1] - top[1]) * down[1]
        along = min(max(along, 0.0), full_len)
        travel = full_len - along
    else:
        radius = DEFAULT_BALL_RADIUS_M
        travel = full_len * (1 - DEFAULT_START_FRACTION)
        defaults_used.append("ball")
        warnings.append("no ball found — starting a ball near the top of the ramp")
    if travel < MIN_TRAVEL_M:
        travel = min(MIN_TRAVEL_M, full_len)
        warnings.append("ball is at the bottom of the ramp — moved it up a little")

    ground = next((line for line in rec.lines if line.role == "ground"), None)
    ground_y = to_m(ground.p1)[1] * 0.5 + to_m(ground.p2)[1] * 0.5 if ground else bottom[1]

    # params: drawing → labels → user override
    values: dict = {"angle_deg": drawn_angle_deg(rec), **DEFAULTS}
    sources = {"angle_deg": "drawing", **{k: "default" for k in DEFAULTS}}
    for label in rec.labels:
        values[label.key] = label.value
        sources[label.key] = "label"
    if override:
        for key, value in override.model_dump(exclude_none=True).items():
            values[key] = value
            sources[key] = "user"
    defaults_used += [k for k in DEFAULTS if sources[k] == "default"]

    angle = values["angle_deg"]
    if not ANGLE_MIN <= angle <= ANGLE_MAX:
        clamped = min(max(angle, ANGLE_MIN), ANGLE_MAX)
        warnings.append(f"angle {angle:.0f}° is outside {ANGLE_MIN:.0f}–{ANGLE_MAX:.0f}°; "
                        f"using {clamped:.0f}°")
        angle = clamped

    # Re-aim the ramp at the chosen angle around its fixed bottom end, keeping its length
    # (a no-op when the angle comes from the drawing itself).
    th = math.radians(angle)
    side = -1.0 if down[0] > 0 else 1.0  # top is left of bottom when the ramp goes down-right
    top = (bottom[0] + side * full_len * math.cos(th), bottom[1] + full_len * math.sin(th))

    params = Params(
        angle_deg=angle,
        mass_kg=values["mass_kg"],
        gravity=values["gravity"],
        friction=values["friction"],
        motion=values["motion"],
        defaults_used=defaults_used,
        sources=sources,
    )
    scene = Scene(
        scale_m_per_px=scale,
        ramp_top_m=top,
        ramp_bottom_m=bottom,
        ground_y_m=ground_y,
        ball_radius_m=radius,
        ramp_length_m=travel,
        params=params,
    )
    return BuildResult(scene, warnings)
