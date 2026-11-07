"""P3.4 number check."""

from app.ta.numcheck import check

ALLOWED = [0.395, 0.972, 2.46, 9.81, 1.62, 37.0, 2.334, 0.53, 0.6]


def test_accepts_context_numbers_and_roundings():
    answer = ("In Run 1 the ball took 0.40 s; in Run 2, on the Moon (g = 1.62 m/s²), it took "
              "0.97 s — 2.46× longer. The ramp is at 37°, and the error was 0.53%.")
    assert check(answer, ALLOWED) == []


def test_flags_invented_numbers_as_written():
    answer = "It would take 3.7 s on Mars and reach 12 m/s."
    assert check(answer, ALLOWED) == ["3.7 s", "12 m/s"]


def test_run_labels_and_constants_are_not_numbers():
    answer = "Compare Run 3 and Run 12: rolling uses $a = \\tfrac{5}{7} g\\sin\\theta$ and $x^2$."
    assert check(answer, ALLOWED) == []


def test_fractions():
    assert check("the ratio is 9.81/1.62", ALLOWED + [6.056]) == []
    assert check("about 13/4 times", ALLOWED) == ["13/4"]
    assert check("$\\frac{2L}{a}$ with $\\dfrac{1}{3}$", ALLOWED) == []


def test_walk_collects_numbers():
    from app.ta.context import _walk

    assert _walk({"a": [1, {"b": 2.5}], "c": True, "d": None, "note": "took 0.40 s",
                  "latex": "\\textcolor{#dc2626}{g}"}) == [1.0, 2.5, 0.4]
