"""Session summary (P3.5): a template from the log always exists; an LLM rewrite is optional
and is only used if it keeps every run and passes the number check."""

import re

from app import store
from app.contracts import Level, Run, SummaryResponse
from app.physics.formula_table import FORMULAS
from app.physics.whycard import why_card
from app.ta import llm, numcheck
from app.ta.context import build_context, context_numbers
from app.ta.prompts import SUMMARY_TASK, system_prompt
from app.ta.templates import explain_recognition

NAMES = {"angle_deg": ("θ", "°"), "mass_kg": ("mass", " kg"), "gravity": ("g", " m/s²"),
         "friction": ("μ", "")}


def _params(run: Run) -> str:
    p = run.scene.params
    return (f"θ = {p.angle_deg:.1f}°, mass {p.mass_kg:g} kg, g = {p.gravity:g} m/s², "
            f"μ = {p.friction:g}, {p.motion}")


def _result(run: Run) -> str:
    v = run.verification
    if v.predicted_time_s is None:
        return "the ball doesn't move (μ ≥ tan θ)"
    t = v.simulated_time_s if v.simulated_time_s is not None else v.predicted_time_s
    speed = v.simulated_speed_m_s or v.predicted_speed_m_s
    check = f", formula {v.predicted_time_s:.3f} s, error {v.error_pct:.2f}%" \
        if v.error_pct is not None else ""
    return f"t = {t:.3f} s, v = {speed:.3f} m/s{check}"


def change_text(prev: Run, run: Run) -> str:
    parts = []
    for key, (a, b) in run.changed.items():
        name, unit = NAMES[key]
        parts.append(f"{name} {a:g}{unit} → {b:g}{unit}")
    pa, pb = prev.scene.params, run.scene.params
    if pa.motion != pb.motion:
        parts.append(f"motion {pa.motion} → {pb.motion}")
    return ", ".join(parts)


def template_summary(session: store.Session) -> str:
    runs = session.runs
    if not runs:
        return "No runs yet — take a photo of your drawing to start."
    lines = ["## Your session", "", explain_recognition(session), "", "### Runs"]
    for prev, run in zip([None, *runs], runs):
        if prev is None:
            lines.append(f"- **Run {run.index}** (as drawn): {_params(run)} → {_result(run)}.")
            continue
        change = change_text(prev, run)
        head = f"- **Run {run.index}**: changed {change}" if change else \
            f"- **Run {run.index}**: same setup as Run {prev.index}"
        lines.append(f"{head} → {_result(run)}.")
        if change:
            lines.append(f"  - *Conclusion:* {why_card(prev, run).sentence}")
    motions = sorted({r.scene.params.motion for r in runs})
    lines += ["", "### Formulas used"]
    for motion in motions:
        f = FORMULAS[motion]
        lines.append(f"- {motion}: ${f['acceleration']}$, ${f['time']}$")
    return "\n".join(lines)


def summarize(session: store.Session, level: Level) -> SummaryResponse:
    template = template_summary(session)
    if not session.runs:
        return SummaryResponse(summary_md=template, source="template")
    text = llm.chat([{"role": "system", "content": system_prompt(level) + "\n" + SUMMARY_TASK},
                     {"role": "user", "content": template}])
    if text:
        context = build_context(session, level)
        flags = numcheck.check(text, context_numbers(context, session.runs))
        keeps_runs = all(re.search(rf"\bRun {r.index}\b", text) for r in session.runs)
        if not flags and keeps_runs:
            return SummaryResponse(summary_md=text, source="llm")
    return SummaryResponse(summary_md=template, source="template")
