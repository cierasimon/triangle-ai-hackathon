"""frontend/src/predict_golden.json pins physics.predict; the browser port is tested against
the same file (frontend/src/physics.test.ts). Regenerate after a formula change with
REGEN_GOLDEN=1 pytest tests/test_predict_golden.py."""

import itertools
import json
import os

import pytest

from app.physics import formulas
from tests.conftest import FIXTURES

GOLDEN = FIXTURES.parent / "frontend" / "src" / "predict_golden.json"


def cases() -> list[dict]:
    out = []
    for motion, angle, friction, gravity, length in itertools.product(
            ["slide", "roll"], [5, 15, 30, 45, 60, 75], [0.0, 0.1, 0.3, 0.6], [9.81, 1.62, 3.71],
            [0.3, 0.8]):
        p = formulas.predict(angle, length, gravity, friction, motion)
        out.append({"motion": motion, "angle_deg": angle, "friction": friction,
                    "gravity": gravity, "length_m": length, "time_s": p.time_s,
                    "speed_m_s": p.speed_m_s, "will_move": p.will_move})
    return out


def test_golden_matches_python():
    current = cases()
    if os.environ.get("REGEN_GOLDEN"):
        GOLDEN.write_text(json.dumps(current, indent=0) + "\n")
    stored = json.loads(GOLDEN.read_text())
    assert len(stored) == len(current)
    for s, c in zip(stored, current):
        assert {k: v for k, v in s.items() if k not in ("time_s", "speed_m_s")} == \
               {k: v for k, v in c.items() if k not in ("time_s", "speed_m_s")}
        for k in ("time_s", "speed_m_s"):
            assert (s[k] is None) == (c[k] is None)
            if c[k] is not None:
                assert s[k] == pytest.approx(c[k], rel=1e-12)
