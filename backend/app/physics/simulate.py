"""pymunk simulation of a ball released from rest on a ramp.

S1 spike result (scripts/spike_roll_vs_slide.py, 15/30/45°, L = 0.5 m, dt = 1/240 s):
  slide, moment = inf, ramp friction 1, ball friction μ   → within 0.6% of g(sin θ − μ cos θ)
  roll, default circle moment (disk, ½mr²), high friction → within 0.5% of (2/3) g sin θ
  roll, moment = 0.4·m·r² (solid sphere), high friction  → within 0.5% of (5/7) g sin θ
Decision: "slide" is the default; "roll" uses the solid-sphere moment so it matches the
textbook (5/7) g sin θ.
"""

import math
from dataclasses import dataclass

import pymunk

from app.contracts import Collision, Frame, Scene, Shape

DT = 1 / 960  # 4 substeps of the plan's 1/240 s: cuts the ~half-step timing bias to ~0.5 ms
RECORD_EVERY = 16  # 60 fps
TIME_LIMIT_S = 10.0
ROLL_FRICTION = 5.0  # high enough to roll without slipping up to 75°
STUCK_AFTER_S = 0.5
STUCK_DIST_M = 1e-6
AT_REST_M_S = 1e-3
STILL_S = 0.1  # at rest this long = stopped (not just the top of a throw)


@dataclass
class SimResult:
    frames: list[Frame]
    time_to_bottom: float | None  # None if it never reached the bottom
    final_speed: float | None


def _unit(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    d = math.dist(a, b)
    return (b[0] - a[0]) / d, (b[1] - a[1]) / d


def launch(body: pymunk.Body, down: tuple[float, float], v0: float, radius: float, rolls: bool) -> None:
    """Give the body its initial speed along the ramp (+ down, - up); a rolling ball starts
    rolling without slipping."""
    body.velocity = (down[0] * v0, down[1] * v0)
    if rolls and v0:
        body.angular_velocity = -v0 * (1.0 if down[0] >= 0 else -1.0) / radius


def start_position(scene: Scene) -> tuple[tuple[float, float], tuple[float, float],
                                          tuple[float, float]]:
    """(ball centre, unit vector down the ramp, outward normal) for the release point."""
    down = _unit(scene.ramp_top_m, scene.ramp_bottom_m)
    normal = (-down[1], down[0])
    if normal[1] < 0:
        normal = (-normal[0], -normal[1])
    bx, by = scene.ramp_bottom_m
    contact = (bx - down[0] * scene.ramp_length_m, by - down[1] * scene.ramp_length_m)
    r = scene.ball_radius_m
    return (contact[0] + normal[0] * r, contact[1] + normal[1] * r), down, normal


def simulate(scene: Scene) -> SimResult:
    p = scene.params
    space = pymunk.Space()
    space.gravity = (0, -p.gravity)
    space.collision_slop = 1e-4  # meters; pymunk's default (0.1) assumes pixels

    center, down, _ = start_position(scene)
    bottom = scene.ramp_bottom_m
    # Only the ramp is a collider: we stop at the ramp bottom, and a ground segment there
    # would touch the ball (centre r·cos θ above it) just before the end and slow it down.
    ramp = pymunk.Segment(space.static_body, scene.ramp_top_m, bottom, 0.0)
    ramp.friction = 1.0

    r = scene.ball_radius_m
    if p.motion == "roll":
        body = pymunk.Body(p.mass_kg, 0.4 * p.mass_kg * r * r)
    else:
        body = pymunk.Body(p.mass_kg, math.inf)
    body.position = center
    ball = pymunk.Circle(body, r)
    ball.friction = ROLL_FRICTION if p.motion == "roll" else p.friction
    space.add(ramp, body, ball)
    launch(body, down, p.initial_speed_m_s, r, p.motion == "roll")

    frames = [Frame(t=0.0, x=center[0], y=center[1], angle=0.0)]
    length = scene.ramp_length_m
    up_room = math.dist(scene.ramp_top_m, bottom) - length  # how far it can rise before the top
    t, prev_s, step, still = 0.0, 0.0, 0, 0
    while t < TIME_LIMIT_S:
        space.step(DT)
        step += 1
        t = step * DT
        pos = body.position
        s = (pos.x - center[0]) * down[0] + (pos.y - center[1]) * down[1]
        if s >= length:
            frac = (s - length) / (s - prev_s) if s > prev_s else 0.0
            t_cross = t - DT * frac
            # final frame exactly at the bottom, interpolated back from this step
            fx = center[0] + down[0] * length
            fy = center[1] + down[1] * length
            frames.append(Frame(t=t_cross, x=fx, y=fy, angle=body.angle))
            return SimResult(frames, t_cross, body.velocity.length)
        if step % RECORD_EVERY == 0:
            frames.append(Frame(t=t, x=pos.x, y=pos.y, angle=body.angle))
        if s < -up_room:
            break  # ran off the top of the ramp
        still = still + 1 if body.velocity.length < AT_REST_M_S else 0
        if t >= STUCK_AFTER_S and still * DT >= STILL_S:
            break  # at rest (held by friction, or stopped on the way): it will not get there
        prev_s = s
    return SimResult(frames, None, None)


# ---- more than one ball -------------------------------------------------------------------

BALL_ELASTICITY = 0.9  # per ball; pymunk multiplies them, so ball–ball restitution e = 0.81
GROUND_HALF_WIDTH_M = 50.0
AFTER_HIT_S = 2.0  # keep going this long after the first hit
CONTACT_SETTLE_STEPS = 10  # read the "after" speeds once the contact has been resolved


GROUND_RUN_K = 0.9  # flat ground past the foot = 0.9 m · (base width / 1 m) ^ 0.75
GROUND_RUN_POWER = 0.75
GROUND_RUN_MIN = 0.3  # ...but at least this share of the ramp's length (steep ramps)


def ground_run(top: tuple[float, float], bottom: tuple[float, float]) -> float:
    """Length of flat ground past the ramp foot. It grows with the ramp, a little slower than
    the ramp does: about 1.1× the base width for a ramp drawn on paper (~0.5 m), 0.8× at 1.6 m
    and 0.65× at 4 m, so a stretched ramp doesn't get an endless run-out."""
    base = abs(top[0] - bottom[0])
    return max(GROUND_RUN_K * base ** GROUND_RUN_POWER, GROUND_RUN_MIN * math.dist(top, bottom))


def ground_end(scene: Scene) -> float:
    """x where the flat ground ends at a stop."""
    if scene.ground_end_x_m is not None:
        return scene.ground_end_x_m
    bottom, top = scene.ramp_bottom_m, scene.ramp_top_m
    away = 1.0 if bottom[0] >= top[0] else -1.0
    return bottom[0] + away * ground_run(top, bottom)


@dataclass
class MultiResult:
    frames: list[Frame]
    collision: Collision | None


GROUND = 0b10  # collision category of the flat ground past the ramp foot
ON_RAMP = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ GROUND)


def _body(space: pymunk.Space, scene: Scene, mass: float, radius: float, kind: Shape,
          center: tuple[float, float], angle: float, on_ramp: bool) -> pymunk.Body:
    """A ball (rolls if the scene says so) or a square block of side 2·radius (always slides)."""
    p = scene.params
    rolls = p.motion == "roll" and kind == "ball"
    body = pymunk.Body(mass, 0.4 * mass * radius * radius if rolls else math.inf)
    body.position = center
    body.angle = angle  # a block sits flat on its surface
    if kind == "block":
        shape = pymunk.Poly.create_box(body, (2 * radius, 2 * radius))
    else:
        shape = pymunk.Circle(body, radius)
    shape.friction = ROLL_FRICTION if rolls else p.friction
    shape.elasticity = BALL_ELASTICITY
    if on_ramp:
        shape.filter = ON_RAMP  # the ground starts at the foot: it can't catch a corner early
    space.add(body, shape)
    return body


def simulate_collisions(scene: Scene) -> MultiResult:
    """The animation: the main ball rolls down, along the ground, hits any other balls, and
    every ball halts at the stop where the ground ends. (The verified time and speed at the
    bottom still come from `simulate`, the ball alone on the ramp.)"""
    p = scene.params
    space = pymunk.Space()
    space.gravity = (0, -p.gravity)
    space.collision_slop = 1e-4

    bottom, top = scene.ramp_bottom_m, scene.ramp_top_m
    away = 1.0 if bottom[0] >= top[0] else -1.0  # direction along the ground, away from the ramp
    # The ground stops where the ramp starts. A body reaching the foot of the ramp is turned
    # onto the ground keeping its speed, as for a smooth join (the textbook setup), instead of
    # losing the part of it that points into the ground at a sharp corner. Until then it
    # ignores the ground, so a block's front corner can't dig in before the foot.
    ramp = pymunk.Segment(space.static_body, top, bottom, 0.0)
    end = ground_end(scene)
    ahead = pymunk.Segment(space.static_body, bottom, (end + away * 0.1, bottom[1]), 0.0)
    ahead.filter = pymunk.ShapeFilter(categories=GROUND)
    behind = pymunk.Segment(space.static_body, (top[0], bottom[1]),
                            (top[0] - away * GROUND_HALF_WIDTH_M, bottom[1]), 0.0)
    for surface in (ramp, ahead, behind):
        surface.friction = 1.0
    space.add(ramp, ahead, behind)

    center, down, _ = start_position(scene)
    up = (-down[0], -down[1])
    tilt = math.atan2(up[1], up[0])  # a block on the ramp is turned to lie flat on it
    on_ramp = [True, *(o.surface == "ramp" for o in scene.others)]
    main = _body(space, scene, p.mass_kg, scene.ball_radius_m, scene.ball_shape, center, tilt, True)
    others = [_body(space, scene, o.mass_kg, o.radius_m, o.shape, o.center_m,
                    tilt if o.surface == "ramp" else 0.0, o.surface == "ramp") for o in scene.others]
    launch(main, down, p.initial_speed_m_s, scene.ball_radius_m, p.motion == "roll" and scene.ball_shape == "ball")
    for body, o in zip(others, scene.others):
        rolls = p.motion == "roll" and o.shape == "ball"
        if o.surface == "ramp":
            launch(body, down, o.speed_m_s, o.radius_m, rolls)
        elif o.speed_m_s:  # on the ground: + is to the right
            body.velocity = (o.speed_m_s, 0.0)
            if rolls:
                body.angular_velocity = -o.speed_m_s / o.radius_m
    bodies = [main, *others]
    radii = [scene.ball_radius_m, *(o.radius_m for o in scene.others)]
    kinds = [scene.ball_shape, *(o.shape for o in scene.others)]
    halted = [False] * len(bodies)

    def frame(t: float) -> Frame:
        return Frame(t=t, x=main.position.x, y=main.position.y, angle=main.angle,
                     others=[(b.position.x, b.position.y, b.angle) for b in others])

    frames = [frame(0.0)]
    collision: Collision | None = None
    pending: tuple[float, int, float, float] | None = None  # (t, other, v1, v2) before the hit
    settle = 0
    still = 0
    t, step = 0.0, 0
    while t < TIME_LIMIT_S:
        for i, b in enumerate(bodies):
            if not on_ramp[i]:
                continue
            # at the foot of the ramp (a ball: about to touch the ground; a block: its contact
            # point reached the foot): carry on along the ground, same speed
            r = radii[i]
            if kinds[i] == "block":
                along = (b.position.x - bottom[0]) * up[0] + (b.position.y - bottom[1]) * up[1]
                if along + (b.velocity.x * up[0] + b.velocity.y * up[1]) * DT > 0:
                    continue
                b.position = (bottom[0] + away * r, bottom[1] + r)  # flat, just past the foot
                b.angle = 0.0
            else:
                if b.position.y + b.velocity.y * DT > bottom[1] + r:
                    continue
                b.position = (b.position.x, bottom[1] + r)
            on_ramp[i] = False
            b.velocity = (away * b.velocity.length, 0.0)
            for shape in b.shapes:
                shape.filter = pymunk.ShapeFilter()
        before = [b.velocity.length for b in bodies]
        space.step(DT)
        step += 1
        t = step * DT
        for i, b in enumerate(bodies):
            # reached the stop at the end of the ground: halt there, and stay as a wall
            if not halted[i] and (b.position.x - end) * away >= -radii[i]:
                halted[i] = True
                b.body_type = pymunk.Body.KINEMATIC
                b.velocity = (0.0, 0.0)
                b.angular_velocity = 0.0
                b.position = (end - away * radii[i], bottom[1] + radii[i])
        if step % RECORD_EVERY == 0:
            frames.append(frame(t))

        if collision is None and pending is None:
            # what the main body touches now (any shapes: ball–ball, ball–block, block–block)
            touching = {q.shape for q in space.shape_query(next(iter(main.shapes)))}
            for i, o in enumerate(others):
                if halted[i + 1]:
                    continue  # parked at the stop: that's the wall, not a collision
                if touching & set(o.shapes):
                    pending = (t, i, before[0], before[i + 1])
                    break
        elif pending is not None:
            settle += 1
            if settle >= CONTACT_SETTLE_STEPS:
                hit_t, i, v1, v2 = pending
                collision = Collision(t=hit_t, other=i, v1_before=v1, v1_after=main.velocity.length,
                                      v2_before=v2, v2_after=others[i].velocity.length)
                pending = None

        if collision and t - collision.t >= AFTER_HIT_S:
            break
        if all(halted):
            break
        still = still + 1 if all(b.velocity.length < AT_REST_M_S for b in bodies) else 0
        if t >= STUCK_AFTER_S and still * DT >= STILL_S:
            break  # nothing moves (held by friction, halted, or everything has stopped)
    if frames[-1].t < t:
        frames.append(frame(t))
    return MultiResult(frames, collision)
