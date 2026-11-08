"""P1.3 scene builder on hand-written Recognition objects."""

import math

import pytest

from app.contracts import Ball, Label, Line, ParamsOverride, Recognition
from app.scene.builder import SceneError, build_scene


def rec(angle_deg=30.0, ball=True, labels=(), rises="left", size=(1000, 800)):
    """Ramp of 500 px whose bottom end is at (700, 700); ball 100 px down from the top."""
    th = math.radians(angle_deg)
    bottom = (700.0, 700.0) if rises == "left" else (300.0, 700.0)
    sx = -1 if rises == "left" else 1
    top = (bottom[0] + sx * 500 * math.cos(th), bottom[1] - 500 * math.sin(th))
    lines = [
        Line(p1=bottom, p2=top, role="ramp", thickness_px=4, confidence=1),
        Line(p1=(50, 700), p2=(950, 700), role="ground", thickness_px=4, confidence=1),
    ]
    b = None
    if ball:
        contact = (top[0] - sx * 100 * math.cos(th), top[1] + 100 * math.sin(th))
        normal = (-sx * math.sin(th), -math.cos(th))  # outward: up and away from the triangle
        b = Ball(center=(contact[0] + 20 * normal[0], contact[1] + 20 * normal[1]),
                 radius=20, confidence=1)
    return Recognition(image_id="x", paper_found=True, paper_corners=None, warped_size=size,
                       lines=lines, ball=b, labels=list(labels))


@pytest.mark.parametrize("rises", ["left", "right"])
def test_meters_and_angle(rises):
    scene = build_scene(rec(30, rises=rises)).scene
    assert scene.scale_m_per_px == pytest.approx(1 / 800)
    assert scene.params.angle_deg == pytest.approx(30)
    assert scene.ramp_bottom_m == pytest.approx(((700 if rises == "left" else 300) / 800, 0.125))
    assert scene.ramp_top_m[1] == pytest.approx(0.125 + 500 * 0.5 / 800)
    assert scene.ground_y_m == pytest.approx(0.125)
    assert scene.ball_radius_m == pytest.approx(20 / 800)
    assert scene.ramp_length_m == pytest.approx(400 / 800)  # 500 px ramp, ball 100 px down
    assert scene.params.sources["angle_deg"] == "drawing"
    assert scene.params.defaults_used == ["mass_kg", "gravity", "friction", "initial_speed_m_s", "motion"]


def test_labels_and_override():
    labels = [Label(key="mass_kg", value=2.0, text="2 kg", confidence=0.9),
              Label(key="angle_deg", value=35, text="35°", confidence=0.9)]
    scene = build_scene(rec(30, labels=labels), ParamsOverride(gravity=1.62)).scene
    p = scene.params
    assert (p.mass_kg, p.angle_deg, p.gravity) == (2.0, 35, 1.62)
    assert p.sources["mass_kg"] == "label" and p.sources["gravity"] == "user"
    assert "mass_kg" not in p.defaults_used and "gravity" not in p.defaults_used
    # re-aimed around the fixed bottom, same ramp length
    (tx, ty), (bx, by) = scene.ramp_top_m, scene.ramp_bottom_m
    assert math.degrees(math.atan2(ty - by, bx - tx)) == pytest.approx(35)
    assert math.dist((tx, ty), (bx, by)) == pytest.approx(500 / 800)
    assert scene.ramp_length_m == pytest.approx(400 / 800)


def test_start_fraction_moves_the_ball():
    scene = build_scene(rec(30), ParamsOverride(start_fraction=0.5)).scene
    assert scene.ramp_length_m == pytest.approx(0.5 * 500 / 800)
    assert "start_fraction" not in scene.params.sources
    # never right at the bottom
    assert build_scene(rec(30), ParamsOverride(start_fraction=0)).scene.ramp_length_m > 0


def test_clamp_and_no_ball():
    result = build_scene(rec(30, ball=False), ParamsOverride(angle_deg=100))
    assert result.scene.params.angle_deg == 90
    assert "ball" in result.scene.params.defaults_used
    assert any("outside" in w for w in result.warnings)
    assert any("no ball" in w for w in result.warnings)


def test_no_ramp():
    r = rec()
    r.lines = [line for line in r.lines if line.role != "ramp"]
    with pytest.raises(SceneError):
        build_scene(r)
