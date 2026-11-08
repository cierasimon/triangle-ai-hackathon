"""Recognition (warped-paper pixels, y down) → Scene (meters, y up).

Paper short side = 1 m. Ramp top = higher endpoint. The ball starts where its centre projects
onto the ramp, resting on it; ramp_length_m is from that contact point to the ramp bottom.
Params: angle from the drawing, handwritten labels override it, user overrides win; anything
missing is a default and listed in defaults_used.
"""

import math
from dataclasses import dataclass, field

from app.contracts import OtherBall, Params, ParamsOverride, Point, Recognition, Scene
from app.physics.simulate import ground_run

ANGLE_MIN, ANGLE_MAX = 0.0, 90.0
DEFAULTS = {"mass_kg": 1.0, "gravity": 9.81, "friction": 0.0, "initial_speed_m_s": 0.0, "motion": "slide"}
MAX_SPEED_M_S = 10.0
DEFAULT_BALL_RADIUS_M = 0.03
DEFAULT_START_FRACTION = 0.1  # no ball drawn: start 10% down from the ramp top
MIN_TRAVEL_M = 0.05
RAMP_MIN_M, RAMP_MAX_M = 0.2, 5.0
DEFAULT_OTHER_MASS_KG = 1.0
GROUND_PAST_OTHERS = 0.5  # room past the farthest other ball, in ground runs
REF_MASS_KG = 1.0  # a body drawn on paper has its drawn size at this mass


def size_factor(mass_kg: float) -> float:
    """How much bigger a body is than drawn at this mass: same material, so its size grows
    with the cube root of its mass (10 kg is about 2.2× as wide as 1 kg, 0.1 kg about half)."""
    return (max(mass_kg, 1e-3) / REF_MASS_KG) ** (1 / 3)


SCENE_ONLY = {"start_fraction", "ramp_full_m", "other_mass_kg", "other_surface", "other_at", "other_speeds_m_s"}  # not Params


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
    shape = rec.ball.shape if rec.ball is not None else "ball"
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
    others = _others(rec, to_m, top, down, full_len)
    if others and override and override.other_surface and override.other_at is not None:
        _, _, r, kind = others[0]  # the learner dragged the second object
        at = min(max(override.other_at, 0.0), 1.0) if override.other_surface == "ramp" else override.other_at
        others[0] = (override.other_surface, at, r, kind)

    if override and override.ramp_full_m is not None:  # stretched ramp: the ball keeps its place
        new_len = min(max(override.ramp_full_m, RAMP_MIN_M), RAMP_MAX_M)
        travel *= new_len / full_len
        full_len = new_len
    if override and override.start_fraction is not None:  # the learner dragged the ball
        travel = full_len * min(max(override.start_fraction, 0.0), 1.0)
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
        for key, value in override.model_dump(exclude_none=True, exclude=SCENE_ONLY).items():
            values[key] = value
            sources[key] = "user"
    defaults_used += [k for k in DEFAULTS if sources[k] == "default"]
    if shape == "block":
        values["motion"] = "slide"  # a block slides; it can't roll
    radius *= size_factor(values["mass_kg"])  # heavier looks bigger

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

    other_mass = override.other_mass_kg if override and override.other_mass_kg else DEFAULT_OTHER_MASS_KG
    other_size = size_factor(other_mass)
    placed = [
        OtherBall(surface=surface, at=at, shape=kind, radius_m=r * other_size, mass_kg=other_mass,
                  speed_m_s=_other_speed(i, rec, override),
                  center_m=other_center(surface, at, r * other_size, bottom, top))
        for i, (surface, at, r, kind) in enumerate(others)
    ]

    # the flat ground runs past the ramp foot (and past any other ball) to a stop, scaled
    # with the ramp so a longer ramp gets a proportionally longer ground
    away = 1.0 if bottom[0] >= top[0] else -1.0
    run = ground_run(top, bottom)
    run_out = max([run, *((o.center_m[0] - bottom[0]) * away + GROUND_PAST_OTHERS * run for o in placed)])
    ground_end = bottom[0] + away * run_out

    params = Params(
        angle_deg=angle,
        mass_kg=values["mass_kg"],
        gravity=values["gravity"],
        friction=values["friction"],
        initial_speed_m_s=min(max(values["initial_speed_m_s"], -MAX_SPEED_M_S), MAX_SPEED_M_S),
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
        ball_shape=shape,
        ramp_length_m=travel,
        params=params,
        others=placed,
        ground_end_x_m=ground_end,
    )
    return BuildResult(scene, warnings)


def _other_speed(i: int, rec: Recognition, override: ParamsOverride | None) -> float:
    """Initial speed of other body i: the learner's value, else the one read from the drawing."""
    given = override.other_speeds_m_s if override and override.other_speeds_m_s else []
    if i < len(given) and given[i] is not None:
        v = given[i]
    else:
        v = rec.other_speeds[i] if i < len(rec.other_speeds) else 0.0
    return min(max(v, -MAX_SPEED_M_S), MAX_SPEED_M_S)


def _others(rec: Recognition, to_m, top: Point, down: Point, full_len: float):
    """(surface, at, radius, shape) for each other body: on the ramp as a fraction from the
    bottom, on the ground as an x position (m)."""
    out = []
    for ball in rec.other_balls:
        c = to_m(ball.center)
        r = ball.radius / min(rec.warped_size)
        along = (c[0] - top[0]) * down[0] + (c[1] - top[1]) * down[1]
        height = (c[0] - top[0]) * -down[1] + (c[1] - top[1]) * down[0]  # signed, either side
        if 0 <= along <= full_len and abs(abs(height) - r) <= 2.5 * r:
            out.append(("ramp", 1 - along / full_len, r, ball.shape))
        else:
            out.append(("ground", c[0], r, ball.shape))
    return out


def other_center(surface: str, at: float, radius: float, bottom: Point, top: Point) -> Point:
    """Resting centre of another ball: on the ramp (at = fraction from the bottom) or on the
    ground, which is level with the ramp bottom (at = x)."""
    if surface == "ground":
        return (at, bottom[1] + radius)
    length = math.dist(top, bottom)
    ux, uy = (top[0] - bottom[0]) / length, (top[1] - bottom[1]) / length
    nx, ny = -uy, ux
    if ny < 0:
        nx, ny = -nx, -ny
    return (bottom[0] + ux * at * length + nx * radius, bottom[1] + uy * at * length + ny * radius)
