"""P1.2 acceptance: synthetic and real-photo recognition targets (via scripts/eval_cv.py)."""

import importlib.util

import numpy as np
import pytest

from app.cv.recognize import rotate_point
from tests.conftest import FIXTURES

_spec = importlib.util.spec_from_file_location(
    "eval_cv", FIXTURES.parent / "scripts" / "eval_cv.py")
eval_cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_cv)


def test_synthetic_targets():
    s = eval_cv.eval_synthetic(FIXTURES / "synthetic")
    assert s["all_pct"] >= 95, s
    assert s["angle_mae"] <= 2, s


def test_real_photo_targets():
    s = eval_cv.eval_real(FIXTURES / "real")
    rows = {r["file"]: r for r in s["rows"]}
    assert sum(r["ground"] for r in s["rows"]) >= 4, s
    no_ball = rows["IMG_9143.jpg"]
    assert no_ball["ball"]  # the drawn square on the ramp is found as a block


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_rotate_point_matches_rot90(k):
    img = np.zeros((30, 50), np.uint8)
    img[7, 41] = 255  # (x=41, y=7)
    rotated = np.rot90(img, k)
    ys, xs = np.nonzero(rotated)
    assert tuple(rotate_point(np.array([41, 7]), k, (50, 30))) == (xs[0], ys[0])
