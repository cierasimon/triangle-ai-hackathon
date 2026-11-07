"""Phase 4: GIF export, share page, QR."""

import io
import time

import cv2
import numpy as np
import pytest
from PIL import Image

from app.contracts import Run
from app.export import gif
from tests.conftest import FIXTURES

PHOTO = FIXTURES / "real" / "IMG_9142.jpg"


@pytest.fixture
def image_id(client):
    with PHOTO.open("rb") as f:
        return client.post("/api/upload", files={"photo": (PHOTO.name, f, "image/jpeg")}).json()["image_id"]


def run(client, image_id, **override) -> dict:
    return client.post("/api/runs", json={"image_id": image_id, "params_override": override}).json()


def export(client, run_id, **headers) -> dict:
    start = time.perf_counter()
    r = client.post(f"/api/export/{run_id}", headers=headers)
    assert r.status_code == 200, r.text
    assert time.perf_counter() - start < 10
    return r.json()


@pytest.mark.parametrize("override", [{}, {"gravity": 1.62}, {"motion": "roll", "angle_deg": 15},
                                      {"angle_deg": 20, "friction": 0.6}],
                         ids=["slide", "moon", "roll", "stuck"])
def test_gif_meets_limits(client, image_id, override):
    r = run(client, image_id, **override)
    e = export(client, r["run_id"])
    data = client.get(e["gif_url"]).content
    assert data[:6] == b"GIF89a" and len(data) <= 5 * 1024 * 1024
    im = Image.open(io.BytesIO(data))
    frames = getattr(im, "n_frames", 1)
    total = 0
    for i in range(frames):
        im.seek(i)
        total += im.info.get("duration", 0)
    assert 5000 <= total <= 8000, total
    assert im.size == (gif.W, gif.H)
    assert im.info.get("loop") == 0  # loops forever


def test_card_text(client, image_id):
    r = run(client, image_id, gravity=1.62)
    head, numbers, setup = gif.card_lines(Run.model_validate(r))
    v = r["verification"]
    assert head == f"PaperPhysics · Run {r['index']}"
    assert f"predicted {v['predicted_time_s']:.2f} s" in numbers
    assert f"simulated {v['simulated_time_s']:.2f} s" in numbers
    assert setup.startswith("t = √(2L / a)") and "g = 1.62" in setup


def test_png_fallback(client, image_id, monkeypatch):
    r = run(client, image_id)
    monkeypatch.setattr(gif, "_gif", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    e = export(client, r["run_id"])
    assert e["gif_url"].endswith("run.png")
    assert client.get(e["gif_url"]).content[:4] == b"\x89PNG"


def test_qr_page_and_download(client, image_id):
    r = run(client, image_id)
    e = export(client, r["run_id"])
    share_id = e["share_url"].rsplit("/", 1)[1]
    assert e["share_url"] == f"http://testserver/s/{share_id}"  # the address it was opened on

    qr = cv2.imdecode(np.frombuffer(client.get(e["qr_png_url"]).content, np.uint8), cv2.IMREAD_COLOR)
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(qr)
    assert decoded == e["share_url"]

    page = client.get(f"/s/{share_id}").text
    assert f"/api/share/{share_id}/run.gif" in page and "Download GIF" in page
    assert f"Run {r['index']}" in page and "predicted" in page

    dl = client.get(f"/api/share/{share_id}/download")
    assert dl.headers["content-disposition"] == f'attachment; filename="paperphysics-run-{r["index"]}.gif"'
    assert client.get("/s/nope").status_code == 404
    events = [ev["kind"] for ev in client.get("/api/session").json()["events"]]
    assert events[-1] == "export"


def test_share_url_uses_lan_or_tunnel_host(client, image_id, monkeypatch):
    r = run(client, image_id)
    lan = export(client, r["run_id"], host="192.168.1.20:5173")
    assert lan["share_url"].startswith("http://192.168.1.20:5173/s/")
    tunnel = export(client, r["run_id"], **{"x-forwarded-host": "abc.trycloudflare.com",
                                            "x-forwarded-proto": "https"})
    assert tunnel["share_url"].startswith("https://abc.trycloudflare.com/s/")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://demo.example.org")
    configured = export(client, r["run_id"], host="192.168.1.20:5173")
    assert configured["share_url"].startswith("https://demo.example.org/s/")
