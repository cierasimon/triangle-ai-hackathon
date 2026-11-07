"""Phase 2: every supported change reruns, matches the formula, and gets a why-card."""

import math

import pytest

from app.contracts import Run
from tests.conftest import FIXTURES

PHOTO = FIXTURES / "synthetic" / "paperphysics_v3_00000.png"


@pytest.fixture
def image_id(client):
    with PHOTO.open("rb") as f:
        r = client.post("/api/upload", files={"photo": (PHOTO.name, f, "image/png")})
    return r.json()["image_id"]


def run(client, image_id, reason="change", **override) -> Run:
    r = client.post("/api/runs", json={"image_id": image_id, "params_override": override,
                                       "reason": reason})
    assert r.status_code == 200, r.text
    return Run.model_validate(r.json())


def card(client, a: Run, b: Run) -> dict:
    return client.post("/api/whycard", json={"run_a": a.run_id, "run_b": b.run_id}).json()


def sim_time(r: Run) -> float:
    return r.verification.simulated_time_s


CHANGES = [
    {"angle_deg": 15}, {"angle_deg": 30}, {"angle_deg": 45},
    {"mass_kg": 2.0},
    {"gravity": 1.62}, {"gravity": 3.71},
    {"friction": 0.1}, {"friction": 0.3}, {"friction": 0.6},
    {"motion": "roll"},
]


@pytest.mark.parametrize("override", CHANGES, ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()))
def test_every_change_matches_formula(client, image_id, override):
    base = run(client, image_id)
    changed = run(client, image_id, **override)
    assert changed.verification.ok, changed.verification
    key = next(iter(override))
    assert changed.scene.params.sources[key] == "user"
    if key != "motion":
        assert changed.changed[key] == (getattr(base.scene.params, key), override[key])


def test_double_mass_same_time_every_time(client, image_id):
    base = run(client, image_id)
    for _ in range(3):
        heavy = run(client, image_id, mass_kg=base.scene.params.mass_kg * 2)
        assert sim_time(heavy) == pytest.approx(sim_time(base), abs=1e-9)
    c = card(client, base, heavy)
    assert c["cancels"] and c["symbol"] == "m"
    assert "same time" in c["sentence"]


def test_moon_earth_ratio(client, image_id):
    earth = run(client, image_id)
    moon = run(client, image_id, gravity=1.62)
    assert round(sim_time(moon) / sim_time(earth), 2) == 2.46
    c = card(client, earth, moon)
    assert not c["cancels"] and c["symbol"] == "g"
    assert r"\sqrt" in c["latex"] and "2.46×" in c["sentence"]


def test_angle_card_shows_dependence(client, image_id):
    a = run(client, image_id, angle_deg=15)
    b = run(client, image_id, angle_deg=45)
    c = card(client, a, b)
    assert c["symbol"] == r"\theta" and r"\sin" in c["latex"] and "shorter" in c["sentence"]


def test_friction_stops_ball_card(client, image_id):
    a = run(client, image_id, angle_deg=30)
    b = run(client, image_id, angle_deg=30, friction=0.6)  # tan 30° = 0.577
    assert b.verification.ok and b.verification.simulated_time_s is None
    c = card(client, a, b)
    assert c["symbol"] == r"\mu" and r"\tan" in c["latex"] and "stays put" in c["sentence"]


def test_roll_card(client, image_id):
    a = run(client, image_id)
    b = run(client, image_id, motion="roll")
    c = card(client, a, b)
    assert "5/7" in c["sentence"] and not c["cancels"]
    assert sim_time(b) / sim_time(a) == pytest.approx(math.sqrt(7 / 5), rel=0.02)


def test_drag_rerun_is_logged(client, image_id):
    run(client, image_id)
    run(client, image_id, reason="drag", angle_deg=50)
    kinds = [e["kind"] for e in client.get("/api/session").json()["events"]]
    assert kinds[-1] == "drag_rerun"


def test_several_changes_card(client, image_id):
    a = run(client, image_id)
    b = run(client, image_id, gravity=1.62, mass_kg=3)
    assert "Several things changed" in card(client, a, b)["sentence"]
