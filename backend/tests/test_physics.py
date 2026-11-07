"""P1.4 simulation and P1.5 formulas."""

import math
import time

import pytest
import sympy as sp

from app.contracts import Params, Scene
from app.physics import formulas
from app.physics.simulate import simulate


def make_scene(angle=30.0, length=0.6, gravity=9.81, friction=0.0, motion="slide", mass=1.0):
    th = math.radians(angle)
    bottom = (1.0, 0.1)
    full = length + 0.1
    top = (bottom[0] - full * math.cos(th), bottom[1] + full * math.sin(th))
    return Scene(
        scale_m_per_px=0.001, ramp_top_m=top, ramp_bottom_m=bottom, ground_y_m=bottom[1],
        ball_radius_m=0.03, ramp_length_m=length,
        params=Params(angle_deg=angle, mass_kg=mass, gravity=gravity, friction=friction,
                      motion=motion),
    )


@pytest.mark.parametrize("angle", [15, 30, 45, 60])
@pytest.mark.parametrize("friction", [0.0, 0.2])
@pytest.mark.parametrize("gravity", [9.81, 1.62])
def test_slide_matches_formula(angle, friction, gravity):
    scene = make_scene(angle, gravity=gravity, friction=friction)
    start = time.perf_counter()
    sim = simulate(scene)
    elapsed = time.perf_counter() - start
    v = formulas.verify(scene, sim.time_to_bottom, sim.final_speed)
    assert v.ok, v
    assert v.error_pct <= 3
    assert abs(sim.final_speed - v.predicted_speed_m_s) / v.predicted_speed_m_s < 0.03
    assert elapsed < 0.3


@pytest.mark.parametrize("angle", [15, 30, 45, 60, 75])
def test_roll_matches_sphere_formula(angle):
    scene = make_scene(angle, motion="roll")
    sim = simulate(scene)
    v = formulas.verify(scene, sim.time_to_bottom, sim.final_speed)
    assert v.ok and v.error_pct <= 3, v
    assert sim.frames[-1].angle != 0  # it actually rotates


def test_stuck_ball_stops_early():
    scene = make_scene(20, friction=0.5)  # tan 20° = 0.36 < 0.5
    start = time.perf_counter()
    sim = simulate(scene)
    assert time.perf_counter() - start < 0.3
    assert sim.time_to_bottom is None
    v = formulas.verify(scene, sim.time_to_bottom, sim.final_speed)
    assert v.ok and v.predicted_time_s is None and "will not slide" in v.note


def test_mass_is_a_real_input_but_does_not_change_time():
    light = simulate(make_scene(mass=1.0)).time_to_bottom
    heavy = simulate(make_scene(mass=2.0)).time_to_bottom
    assert abs(light - heavy) < 1e-6


def test_frames_are_60fps_and_start_on_the_ramp():
    sim = simulate(make_scene())
    dts = {round(b.t - a.t, 6) for a, b in zip(sim.frames[:-2], sim.frames[1:-1])}
    assert dts == {round(16 / 960, 6)}
    assert sim.frames[0].t == 0


def test_time_does_not_depend_on_mass():
    for t in formulas.TIME.values():
        assert sp.diff(t, formulas.m) == 0


def test_moon_earth_ratio():
    earth = formulas.predict(30, 1.0, gravity=9.81).time_s
    moon = formulas.predict(30, 1.0, gravity=1.62).time_s
    assert moon / earth == pytest.approx(math.sqrt(9.81 / 1.62))
    assert round(moon / earth, 2) == 2.46


def test_predict_is_fast():
    start = time.perf_counter()
    for i in range(1000):
        formulas.predict(10 + i % 60, 0.7, 9.81, 0.1, "slide")
    assert (time.perf_counter() - start) / 1000 < 0.001


def test_will_move():
    assert not formulas.predict(30, 1, friction=0.6).will_move  # tan 30° = 0.577
    assert formulas.predict(30, 1, friction=0.5).will_move
    assert formulas.predict(30, 1, friction=0.9, motion="roll").will_move
