"""S1 spike: which pymunk setup matches which textbook formula?

pymunk is 2D, so a circle's default moment is a disk (I = ½mr²): rolling gives
a = (2/3) g sin θ, not the solid-sphere (5/7) g sin θ. Measures time-to-bottom error for
three setups at 15/30/45°, L = 0.5 m, dt = 1/240 s.

    .venv/bin/python scripts/spike_roll_vs_slide.py
"""

import math

import pymunk

DT = 1 / 240
L = 0.5
R = 0.03
G = 9.81


def time_to_bottom(theta_deg: float, moment: float, mu: float) -> float | None:
    theta = math.radians(theta_deg)
    space = pymunk.Space()
    space.gravity = (0, -G)
    down = (math.cos(theta), -math.sin(theta))
    normal = (math.sin(theta), math.cos(theta))
    top = (0.0, 0.0)
    bottom = (top[0] + 2 * L * down[0], top[1] + 2 * L * down[1])  # ramp longer than L
    ramp = pymunk.Segment(space.static_body, top, bottom, 0.0)
    ramp.friction = 1.0
    body = pymunk.Body(1.0, moment)
    body.position = (normal[0] * R, normal[1] * R)
    ball = pymunk.Circle(body, R)
    ball.friction = mu
    space.add(ramp, body, ball)
    start = body.position
    t, prev_s = 0.0, 0.0
    while t < 10:
        space.step(DT)
        t += DT
        d = body.position - start
        s = d.x * down[0] + d.y * down[1]
        if s >= L:
            return t - DT * (s - L) / (s - prev_s)  # interpolate the crossing
        prev_s = s
    return None


def main() -> None:
    rows = []
    for theta in (15, 30, 45):
        sin, cos = math.sin(math.radians(theta)), math.cos(math.radians(theta))
        mu = 0.1
        setups = [
            ("slide (moment=inf, mu=0.1)", math.inf, mu, G * (sin - mu * cos)),
            ("roll as disk (default moment)", pymunk.moment_for_circle(1, 0, R), 5.0, 2 / 3 * G * sin),
            ("roll as sphere (0.4 m r^2)", 0.4 * R * R, 5.0, 5 / 7 * G * sin),
        ]
        for name, moment, ball_mu, a in setups:
            sim = time_to_bottom(theta, moment, ball_mu)
            ref = math.sqrt(2 * L / a)
            err = abs(sim - ref) / ref * 100 if sim else float("nan")
            rows.append((theta, name, ref, sim, err))
    print(f"{'θ':>3}  {'setup':<32} {'formula t':>9} {'sim t':>8} {'err %':>6}")
    for theta, name, ref, sim, err in rows:
        print(f"{theta:>3}  {name:<32} {ref:9.4f} {sim:8.4f} {err:6.2f}")


if __name__ == "__main__":
    main()
