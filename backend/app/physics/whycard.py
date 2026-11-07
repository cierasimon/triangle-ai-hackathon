"""Why-cards: explain the difference between two runs from the formula (P2.4).

The changed parameter's symbol is looked up in t(θ, μ, g, m, L). If ∂t/∂param = 0 it
"cancels". Otherwise SymPy splits t into factors and the factor holding the parameter is the
scaling law (t ∝ 1/√g, t ∝ 1/√(sin θ − μ cos θ), ...). Numbers come from the two runs.
"""

import sympy as sp

from app.contracts import Run, WhyCard
from app.physics.formulas import LATEX, TIME, L, g, m, mu, theta

HIGHLIGHT = "#dc2626"
SYMBOLS = {"angle_deg": theta, "friction": mu, "gravity": g, "mass_kg": m}
NAMES = {"angle_deg": "the angle", "friction": "friction", "gravity": "gravity",
         "mass_kg": "the mass"}
UNITS = {"angle_deg": "°", "friction": "", "gravity": " m/s²", "mass_kg": " kg"}


def _hl(sym: sp.Symbol, highlighted: sp.Symbol | None) -> str:
    name = sp.latex(sym)
    return rf"\textcolor{{{HIGHLIGHT}}}{{{name}}}" if sym == highlighted else name


def scaling_factor(motion: str, sym: sp.Symbol) -> sp.Expr:
    """The part of t that depends on sym (1 if none)."""
    t = sp.powsimp(sp.expand_power_base(TIME[motion], force=True), force=True)
    parts = [sp.powdenest(f, force=True) for f in sp.Mul.make_args(t) if f.has(sym)]
    return sp.Mul(*parts)


def factor_latex(motion: str, sym: sp.Symbol) -> str:
    """LaTeX for scaling_factor in textbook order (SymPy prints −μ cos θ + sin θ)."""
    factor = scaling_factor(motion, sym)
    T, M = _hl(theta, sym), _hl(mu, sym)
    parts = []
    if factor.has(L):
        parts.append(rf"\sqrt{{{_hl(L, sym)}}}")
    if factor.has(g):
        parts.append(rf"\dfrac{{1}}{{\sqrt{{{_hl(g, sym)}}}}}")
    if factor.has(theta) or factor.has(mu):
        inner = rf"\sin{T} - {M}\cos{T}" if motion == "slide" else rf"\sin{T}"
        parts.append(rf"\dfrac{{1}}{{\sqrt{{{inner}}}}}")
    return " ".join(parts)


def _fmt(v: float, key: str) -> str:
    return f"{v:g}{UNITS[key]}"


def _time(run: Run) -> float | None:
    v = run.verification
    return v.simulated_time_s if v.simulated_time_s is not None else v.predicted_time_s


def why_card(a: Run, b: Run) -> WhyCard:
    pa, pb = a.scene.params, b.scene.params
    changed = [k for k in SYMBOLS if getattr(pa, k) != getattr(pb, k)]
    motion_changed = pa.motion != pb.motion
    if abs(a.scene.ramp_length_m - b.scene.ramp_length_m) > 1e-9:
        changed.append("length")
    ta, tb = _time(a), _time(b)
    ratio = f"{tb / ta:.2f}×" if ta and tb else None

    if motion_changed:
        return WhyCard(
            latex=r"\begin{gathered} a_\text{slide} = g(\sin\theta - \mu\cos\theta) \\ "
                  r"a_\text{roll} = \tfrac{5}{7}\, g\sin\theta \end{gathered}",
            symbol="a",
            sentence=(f"Run {b.index} {'rolls' if pb.motion == 'roll' else 'slides'} instead of "
                      f"{'rolling' if pa.motion == 'roll' else 'sliding'}. A rolling ball spends "
                      "part of its energy spinning, so it speeds up at 5/7 of the frictionless "
                      "sliding rate" + (f": time × {ratio}." if ratio else ".")),
            cancels=False,
        )
    if not changed:
        return WhyCard(latex=LATEX[pb.motion], symbol="", cancels=True,
                       sentence=f"Run {a.index} and Run {b.index} have the same setup, so the "
                                "same result.")
    if len(changed) > 1:
        names = ", ".join(NAMES.get(k, "the ramp length") for k in changed)
        return WhyCard(latex=LATEX[pb.motion], symbol="", cancels=False,
                       sentence=f"Several things changed between Run {a.index} and Run {b.index} "
                                f"({names}). Change one at a time to see each effect.")

    key = changed[0]
    if key == "length":
        sym, before, after = L, a.scene.ramp_length_m, b.scene.ramp_length_m
        name, unit_before, unit_after = "the distance down the ramp", f"{before:.2f} m", f"{after:.2f} m"
    else:
        sym, before, after = SYMBOLS[key], getattr(pa, key), getattr(pb, key)
        name, unit_before, unit_after = NAMES[key], _fmt(before, key), _fmt(after, key)
    t_expr = TIME[pb.motion]
    lead = f"{name.capitalize()} went from {unit_before} to {unit_after}."

    if sp.diff(t_expr, sym) == 0:
        return WhyCard(
            latex=rf"\begin{{gathered}} {LATEX[pb.motion]} \\ (\text{{no }} "
                  rf"{_hl(sym, sym)} \text{{ anywhere}}) \end{{gathered}}",
            symbol=sp.latex(sym),
            sentence=f"{lead} {sp.pretty(sym)} does not appear in the formula for t, so it "
                     f"cancels: Run {a.index} and Run {b.index} take the same time"
                     + (f" ({ta:.2f} s vs {tb:.2f} s)." if ta and tb else "."),
            cancels=True,
        )

    if key == "friction" and pb.motion == "slide" and tb is None and b.verification.predicted_time_s is None:
        return WhyCard(
            latex=rf"\begin{{gathered}} a = g(\sin\theta - {_hl(mu, mu)}\cos\theta) \le 0 "
                  rf"\\ \iff {_hl(mu, mu)} \ge \tan\theta \end{{gathered}}",
            symbol=r"\mu",
            sentence=f"{lead} Friction (μ cos θ) now cancels all of the pull down the slope "
                     f"(sin θ), so the ball stays put: μ = {pb.friction:g} ≥ tan θ.",
            cancels=False,
        )

    effect = ""
    if ta and tb:
        word = "longer" if tb > ta else "shorter"
        effect = f" The time is {ratio} ({ta:.2f} s → {tb:.2f} s), {word}."
    return WhyCard(
        latex=rf"t \propto {factor_latex(pb.motion, sym)}",
        symbol=sp.latex(sym),
        sentence=f"{lead}{effect}",
        cancels=False,
    )
