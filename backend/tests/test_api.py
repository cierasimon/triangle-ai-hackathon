"""API: photo → recognition → run, plus the placeholder routes for later phases."""

import json

from app.contracts import Recognition, Run, SessionState
from tests.conftest import FIXTURES


def _upload(client, path):
    with path.open("rb") as f:
        r = client.post("/api/upload", files={"photo": (path.name, f, "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    assert client.get("/api/health").json() == {"ok": True}


def test_fixtures_have_expected_json(synthetic_images):
    assert len(synthetic_images) >= 20
    for image in synthetic_images:
        record = json.loads(image.with_suffix(".json").read_text())
        assert (record["primary_type"], record["scene_type"]) == ("ball", "plain")


def test_upload_and_run_every_fixture(client, synthetic_images):
    for path in synthetic_images:
        body = _upload(client, path)
        rec = Recognition.model_validate(body["recognition"])
        assert rec.ball is not None and {line.role for line in rec.lines} == {"ramp", "ground"}
        overlay = client.get(body["overlay_png_url"])
        assert overlay.status_code == 200 and overlay.content[:4] == b"\x89PNG"

        r = client.post("/api/runs", json={"image_id": body["image_id"]})
        assert r.status_code == 200, r.text
        run = Run.model_validate(r.json())
        assert run.frames and run.verification.ok
        assert run.frames[-1].y < run.frames[0].y  # ball goes down


def test_override_records_change(client, synthetic_images):
    image_id = _upload(client, synthetic_images[0])["image_id"]
    first = client.post("/api/runs", json={"image_id": image_id}).json()
    second = client.post("/api/runs", json={"image_id": image_id,
                                            "params_override": {"gravity": 1.62}}).json()
    assert (first["run_id"], second["run_id"]) == ("r1", "r2")
    assert second["changed"] == {"gravity": [9.81, 1.62]}
    assert second["scene"]["params"]["sources"]["gravity"] == "user"
    assert second["verification"]["simulated_time_s"] > first["verification"]["simulated_time_s"]

    state = SessionState.model_validate(client.get("/api/session").json())
    assert [e.kind for e in state.events] == ["upload", "run", "run", "change"]


def test_predict(client):
    r = client.get("/api/predict", params={"angle_deg": 30, "length_m": 1}).json()
    assert r["will_move"] and abs(r["time_s"] - (2 / (9.81 * 0.5)) ** 0.5) < 1e-9
    stuck = client.get("/api/predict",
                       params={"angle_deg": 10, "length_m": 1, "friction": 0.5}).json()
    assert not stuck["will_move"] and stuck["time_s"] is None


def test_whycard_ta_export(client, synthetic_images):
    image_id = _upload(client, synthetic_images[0])["image_id"]
    client.post("/api/runs", json={"image_id": image_id})
    client.post("/api/runs", json={"image_id": image_id, "params_override": {"mass_kg": 2}})

    card = client.post("/api/whycard", json={"run_a": "r1", "run_b": "r2"}).json()
    assert set(card) == {"latex", "symbol", "sentence", "cancels"}

    ans = client.post("/api/ta/ask", json={"question": "why?", "level": "beginner"}).json()
    assert ans["based_on"] == ["r2"] and len(ans["followups"]) == 2
    assert client.post("/api/ta/summary").json()["source"] == "template"
    quiz = client.post("/api/ta/quiz").json()
    result = client.post("/api/ta/quiz/answer", json={"quiz_id": quiz["quiz_id"], "guess": 2})
    assert result.status_code == 200

    exp = client.post("/api/export/r1").json()
    assert client.get(exp["gif_url"]).content[:3] == b"GIF"
    assert client.get(exp["qr_png_url"]).content[:4] == b"\x89PNG"
    share_id = exp["share_url"].rsplit("/", 1)[1]
    assert "Download GIF" in client.get(f"/s/{share_id}").text


def test_session_header_isolates_sessions(client, synthetic_images):
    _upload(client, synthetic_images[0])
    other = client.get("/api/session", headers={"X-Session": "someone-else"}).json()
    assert other["session_id"] == "someone-else" and other["events"] == []


def test_bad_upload(client):
    r = client.post("/api/upload", files={"photo": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_real_photos_run_end_to_end(client):
    for path in sorted((FIXTURES / "real").glob("*.jpg")):
        body = _upload(client, path)
        r = client.post("/api/runs", json={"image_id": body["image_id"]})
        assert r.status_code == 200, (path.name, r.text)
        run = Run.model_validate(r.json())
        assert run.verification.ok, (path.name, run.verification)
        if body["recognition"]["ball"] is None:  # IMG_9143: default ball, clearly marked
            assert "ball" in run.scene.params.defaults_used


def test_corners_redo_recognition(client, synthetic_images):
    body = _upload(client, synthetic_images[0])
    corners = [[0, 0], [767, 0], [767, 575], [0, 575]]
    r = client.post("/api/corners", json={"image_id": body["image_id"], "corners": corners})
    assert r.status_code == 200, r.text
    rec = Recognition.model_validate(r.json())
    assert rec.paper_found and rec.ball is not None
    assert client.post("/api/runs", json={"image_id": body["image_id"]}).json()["verification"]["ok"]


def test_blank_page_gives_clear_error(client):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(buf, format="PNG")
    r = client.post("/api/upload", files={"photo": ("blank.png", buf.getvalue(), "image/png")})
    rec = r.json()["recognition"]
    assert rec["lines"] == [] and any("no ramp" in w for w in rec["warnings"])
    r = client.post("/api/runs", json={"image_id": r.json()["image_id"]})
    assert r.status_code == 422 and "no ramp" in r.json()["detail"]


def test_root_files_stay_inside_frontend_dist(client):
    for path in ("/..%2F..%2F.env", "/..%2Fplan.md", "/%2E%2E%2FMakefile", "/nope.txt"):
        assert client.get(path).status_code == 404, path
