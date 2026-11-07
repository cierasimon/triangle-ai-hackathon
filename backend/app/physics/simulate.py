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

from app.contracts import Frame, Scene

DT = 1 / 240
RECORD_EVERY = 4  # 60 fps
TIME_LIMIT_S = 10.0
ROLL_FRICTION = 5.0  # high enough to roll without slipping up to 75°
STUCK_AFTER_S = 0.5
STUCK_DIST_M = 1e-6


@dataclass
class SimResult:
    frames: list[Frame]
    time_to_bottom: float | None  # None if it never reached the bottom
    final_speed: float | None


def _unit(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    d = math.dist(a, b)
    return (b[0] - a[0]) / d, (b[1] - a[1]) / d


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

    frames = [Frame(t=0.0, x=center[0], y=center[1], angle=0.0)]
    length = scene.ramp_length_m
    t, prev_s, step = 0.0, 0.0, 0
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
        if t >= STUCK_AFTER_S and s < STUCK_DIST_M:
            break  # held by friction: it will not move
        prev_s = s
    return SimResult(frames, None, None)
