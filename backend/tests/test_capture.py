"""P1.1 capture: paper warp, whole-image fallback, ink mask."""

import json

import cv2
import numpy as np
import pytest

from app.cv import capture as cap
from tests.conftest import FIXTURES

SYNTHETIC = sorted((FIXTURES / "synthetic").glob("*.png"))


def tilted_on_desk(page: np.ndarray):
    """Paste a page onto a darker desk with a perspective tilt; return photo + true corners."""
    h, w = page.shape[:2]
    photo_w, photo_h = 1600, 1300
    corners = np.float32([(260, 180), (1380, 240), (1300, 1120), (200, 1050)])
    matrix = cv2.getPerspectiveTransform(np.float32([(0, 0), (w, 0), (w, h), (0, h)]), corners)
    desk = np.full((photo_h, photo_w, 3), (70, 85, 95), np.uint8)
    warped = cv2.warpPerspective(page, matrix, (photo_w, photo_h))
    inside = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, (photo_w, photo_h))
    desk[inside > 0] = warped[inside > 0]
    return desk, corners


def edge_lengths(quad) -> np.ndarray:
    q = cap.order_corners(quad)
    return np.array([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)])


@pytest.mark.parametrize("path", SYNTHETIC[:5], ids=lambda p: p.stem)
def test_warp_recovers_tilted_page(path):
    page = cv2.imread(str(path))
    photo, true_corners = tilted_on_desk(page)
    found = cap.find_paper(photo)
    assert found is not None
    rel = np.abs(edge_lengths(found) - edge_lengths(true_corners)) / edge_lengths(true_corners)
    assert rel.max() < 0.03

    result = cap.capture(photo)
    assert result.paper_found
    h, w = result.image.shape[:2]
    # averaged edge lengths only approximate the aspect under strong perspective
    assert abs(w / h - page.shape[1] / page.shape[0]) < 0.1
    assert max(w, h) == cap.WORK_LONG_SIDE


@pytest.mark.parametrize("path", SYNTHETIC, ids=lambda p: p.stem)
def test_synthetic_mask_is_sensible(path):
    img = cv2.imread(str(path))
    result = cap.capture(img)
    assert not result.paper_found and result.warnings  # flat render: whole image fallback
    ink = result.mask > 0
    assert 0.005 < ink.mean() < 0.08  # some ink, mostly paper
    # the ball centre's ring is ink: sample the circle from the annotation
    ann = json.loads(path.with_suffix(".json").read_text())
    ball = next(o for o in ann["objects"] if o["class"] == "ball")
    s = result.image.shape[1] / ann["width"]
    cx, cy, r = (v * s for v in (*ball["center_xy"], ball["radius_px"]))
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    ring = [result.mask[int(cy + r * np.sin(a)) - 2:int(cy + r * np.sin(a)) + 3,
                        int(cx + r * np.cos(a)) - 2:int(cx + r * np.cos(a)) + 3].any()
            for a in angles]
    assert np.mean(ring) > 0.9


def test_real_photos_drop_ruled_lines():
    for path in sorted((FIXTURES / "real").glob("*.jpg")):
        result = cap.capture(cap.load_image(path.read_bytes()))
        # ruled lines span the page; ink that does must be rare
        mask = result.mask > 0
        full_rows = (mask.mean(axis=1) > 0.5).sum() + (mask.mean(axis=0) > 0.5).sum()
        assert full_rows == 0, path.name
        assert mask.mean() < 0.05, path.name


def test_apply_corners():
    page = cv2.imread(str(SYNTHETIC[0]))
    photo, corners = tilted_on_desk(page)
    result = cap.apply_corners(photo, [tuple(c) for c in corners[[2, 0, 3, 1]]])  # any order
    assert result.paper_found
    assert abs(result.image.shape[1] / result.image.shape[0] - 768 / 576) < 0.1
