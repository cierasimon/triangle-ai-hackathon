"""Two balls: CV finds the target, the scene places it, the simulation makes them collide."""

import json
import math

import pytest

from app.contracts import Ball, Line, OtherBall, ParamsOverride, Recognition
from app.cv import hybrid
from app.cv.capture import capture, load_image
from app.cv.recognize import recognize
from app.physics.simulate import BALL_ELASTICITY, ground_end, simulate, simulate_collisions
from app.scene.builder import build_scene, other_center
from tests.conftest import FIXTURES
from tests.test_physics import make_scene

E = BALL_ELASTICITY ** 2  # pymunk multiplies the two balls' elasticities


@pytest.mark.parametrize("name, surface", [("ball_target_flat.png", "ground"),
                                           ("ball_target_ramp.png", "ramp")])
def test_cv_finds_the_target_ball(name, surface):
    rec = recognize(capture(load_image((FIXTURES / "collision" / name).read_bytes())), "x").recognition
    assert rec.ball is not None
    assert len(rec.other_balls) == 1
    assert [o.surface for o in build_scene(rec).scene.others] == [surface]


def test_plain_fixture_has_no_other_balls():
    image = sorted((FIXTURES / "synthetic").glob("*.png"))[0]
    assert recognize(capture(load_image(image.read_bytes())), "x").recognition.other_balls == []


def two_ball_rec(other_center_px):
    """Ramp rising left from (700, 700), 500 px at 30°; ball near its top; ground at y = 700."""
    th = math.radians(30)
    bottom, top = (700.0, 700.0), (700 - 500 * math.cos(th), 700 - 500 * math.sin(th))
    lines = [Line(p1=bottom, p2=top, role="ramp", thickness_px=4, confidence=1),
             Line(p1=(50, 700), p2=(950, 700), role="ground", thickness_px=4, confidence=1)]
    contact = (top[0] + 100 * math.cos(th), top[1] + 100 * math.sin(th))
    main = Ball(center=(contact[0] + 20 * math.sin(th), contact[1] - 20 * math.cos(th)),
                radius=20, confidence=1)
    other = Ball(center=other_center_px, radius=20, confidence=1)
    return Recognition(image_id="x", paper_found=True, paper_corners=None, warped_size=(1000, 800),
                       lines=lines, ball=main, other_balls=[other])


def test_scene_places_ground_ball_and_follows_overrides():
    rec = two_ball_rec((850.0, 680.0))  # on the ground, right of the ramp bottom
    scene = build_scene(rec).scene
    (o,) = scene.others
    assert o.surface == "ground" and o.at == pytest.approx(850 / 800)
    assert o.center_m == pytest.approx((850 / 800, 0.125 + 20 / 800))
    assert o.mass_kg == 1.0

    longer = build_scene(rec, ParamsOverride(ramp_full_m=1.25, other_mass_kg=3)).scene
    assert math.dist(longer.ramp_top_m, longer.ramp_bottom_m) == pytest.approx(1.25)
    assert longer.ramp_length_m == pytest.approx(2 * scene.ramp_length_m)  # same place, 2× ramp
    assert longer.others[0].mass_kg == 3
    assert "ramp_full_m" not in longer.params.sources


def test_ramp_ball_keeps_its_place_on_the_ramp():
    th = math.radians(30)
    on_ramp = (700 - 150 * math.cos(th) + 20 * math.sin(th), 700 - 150 * math.sin(th) - 20 * math.cos(th))
    scene = build_scene(two_ball_rec(on_ramp), ParamsOverride(angle_deg=45)).scene
    (o,) = scene.others
    assert o.surface == "ramp" and o.at == pytest.approx(150 / 500, abs=0.01)
    assert o.center_m == pytest.approx(other_center("ramp", o.at, o.radius_m,
                                                    scene.ramp_bottom_m, scene.ramp_top_m))


@pytest.mark.parametrize("m2", [0.5, 1.0, 3.0])
def test_head_on_hit_on_the_ground(m2):
    scene = make_scene(30)
    r = scene.ball_radius_m
    x = scene.ramp_bottom_m[0] + 0.3
    scene.others = [OtherBall(surface="ground", at=x, radius_m=r, mass_kg=m2,
                              center_m=(x, scene.ramp_bottom_m[1] + r))]
    result = simulate_collisions(scene)
    c = result.collision
    assert c is not None and c.v2_before == pytest.approx(0, abs=1e-6)
    m1, v = scene.params.mass_kg, c.v1_before
    v1 = (m1 - E * m2) * v / (m1 + m2)  # 1-D collision with restitution E
    v2 = m1 * (1 + E) * v / (m1 + m2)
    assert c.v1_after == pytest.approx(abs(v1), rel=0.05, abs=0.02)
    assert c.v2_after == pytest.approx(v2, rel=0.05)
    assert all(len(f.others) == 1 for f in result.frames)
    # the target ends up past where it started, moving away from the ramp
    assert result.frames[-1].others[0][0] > x


@pytest.mark.parametrize("angle", [20, 40, 60])
def test_ball_keeps_its_speed_onto_the_ground(angle):
    """Smooth join: the hit speed is the speed at the bottom of the ramp (no corner loss)."""
    scene = make_scene(angle)
    r = scene.ball_radius_m
    x = scene.ramp_bottom_m[0] + 0.12  # on the flat, well before the stop
    scene.others = [OtherBall(surface="ground", at=x, radius_m=r, mass_kg=1.0,
                              center_m=(x, scene.ramp_bottom_m[1] + r))]
    bottom_speed = simulate(scene).final_speed
    assert simulate_collisions(scene).collision.v1_before == pytest.approx(bottom_speed, rel=0.03)


@pytest.mark.parametrize("with_target", [False, True])
def test_balls_halt_at_the_end_of_the_ground(with_target):
    scene = make_scene(40)
    r = scene.ball_radius_m
    if with_target:
        x = scene.ramp_bottom_m[0] + 0.2
        scene.others = [OtherBall(surface="ground", at=x, radius_m=r, mass_kg=0.3,
                                  center_m=(x, scene.ramp_bottom_m[1] + r))]
    end = ground_end(scene)
    assert end > scene.ramp_bottom_m[0]  # the ground runs away from the ramp (it rises left)
    result = simulate_collisions(scene)
    last = result.frames[-1]
    xs = [f.x for f in result.frames] + [o[0] for f in result.frames for o in f.others]
    assert max(xs) <= end - r + 1e-9  # nothing ever passes the stop
    fastest = last.others[0][0] if with_target else last.x  # the light target, or the lone ball
    assert fastest == pytest.approx(end - r)


@pytest.mark.parametrize("angle", [15, 40, 70])
def test_ground_scales_with_the_ramp(angle):
    """Stretching the ramp stretches the ground too, a little less than one-for-one."""
    rec = two_ball_rec((850.0, 680.0))
    rec.other_balls = []  # just the ramp: other balls add room past themselves
    runs = []
    for length in (0.5, 1.0, 4.0):
        s = build_scene(rec, ParamsOverride(angle_deg=angle, ramp_full_m=length, start_fraction=0.9)).scene
        runs.append((s.ground_end_x_m - s.ramp_bottom_m[0], math.dist(s.ramp_top_m, s.ramp_bottom_m)))
    grounds = [run for run, _ in runs]
    ratios = [run / length for run, length in runs]
    assert grounds[0] < grounds[1] < grounds[2]  # a longer ramp gets a longer ground...
    assert ratios[0] >= ratios[1] >= ratios[2]  # ...but not one-for-one
    assert ratios[2] <= 0.65  # a stretched 4 m ramp: well short of its own length


def gemini_from_annotation(name: str) -> "hybrid.VisionAnalysis":
    """What Gemini would propose for a generated drawing: its true ball/block boxes."""
    ann = json.loads((FIXTURES / "collision" / name.replace(".png", ".json")).read_text())
    w, h = ann["width"], ann["height"]
    objects = [{"kind": o["class"], "confidence": 0.95,
                "bbox": [o["bbox_xyxy"][0] * 1000 / w, o["bbox_xyxy"][1] * 1000 / h,
                         o["bbox_xyxy"][2] * 1000 / w, o["bbox_xyxy"][3] * 1000 / h]}
               for o in ann["objects"] if o["class"] in ("ball", "block")]
    return hybrid.VisionAnalysis(objects=objects, ramp=None, ground=None, summary="test")


def recognize_with_gemini(name: str, monkeypatch) -> Recognition:
    """The hybrid path, with the remote Gemini call replaced by the drawing's annotation."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "unit-test")
    monkeypatch.setattr(hybrid, "analyze", lambda *_: gemini_from_annotation(name))
    result = recognize(capture(load_image((FIXTURES / "collision" / name).read_bytes())), "x")
    return hybrid.assist(result).recognition


def test_blocks_are_found_without_gemini():
    """A drawn square is found from the ink alone, so a failed or missing Gemini call loses nothing."""
    rec = recognize(capture(load_image((FIXTURES / "collision" / "block_plain.png").read_bytes())), "x").recognition
    assert len(rec.blocks) == 1 and rec.ball is not None and rec.ball.shape == "block"


@pytest.mark.parametrize("name, main, others", [
    ("block_plain.png", "block", []),
    ("ball_target_block_flat.png", "ball", ["block"]),
    ("block_hits_ball_flat.png", "block", ["ball"]),
])
def test_gemini_blocks_slide_hit_and_halt(name, main, others, monkeypatch):
    rec = recognize_with_gemini(name, monkeypatch)
    assert rec.gemini_status == "success" and rec.blocks
    scene = build_scene(rec).scene
    assert scene.ball_shape == main and [o.shape for o in scene.others] == others
    assert scene.params.motion == "slide"
    result = simulate_collisions(scene)
    last = result.frames[-1]
    assert abs(last.angle) < 1e-9 or main == "ball"  # a block ends flat on the ground
    if others:
        c = result.collision
        assert c is not None and c.v2_before == pytest.approx(0, abs=1e-9)
        assert c.v2_after > c.v1_after  # momentum passed on


def test_heavier_looks_bigger():
    rec = two_ball_rec((850.0, 680.0))
    light = build_scene(rec, ParamsOverride(mass_kg=0.5, other_mass_kg=0.5)).scene
    heavy = build_scene(rec, ParamsOverride(mass_kg=8, other_mass_kg=8)).scene
    assert heavy.ball_radius_m / light.ball_radius_m == pytest.approx(16 ** (1 / 3))  # 16× the mass
    assert heavy.others[0].radius_m / light.others[0].radius_m == pytest.approx(16 ** (1 / 3))
    assert heavy.others[0].center_m[1] - heavy.ramp_bottom_m[1] == pytest.approx(heavy.others[0].radius_m)


def test_dragging_the_second_object():
    rec = two_ball_rec((850.0, 680.0))
    onto_ramp = build_scene(rec, ParamsOverride(other_surface="ramp", other_at=0.4)).scene.others[0]
    assert (onto_ramp.surface, onto_ramp.at) == ("ramp", 0.4)
    along = build_scene(rec, ParamsOverride(other_surface="ground", other_at=1.0)).scene.others[0]
    assert along.surface == "ground" and along.center_m[0] == pytest.approx(1.0)
