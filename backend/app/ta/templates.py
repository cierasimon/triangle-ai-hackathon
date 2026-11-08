"""Template-mode TA answers: no LLM, numbers only from the session (P3.3 fallback, P3.O2).

An intent is picked from keywords; the answer is a level-specific explanation plus, when the
learner already tried that change, the engine's numbers for it ("Run 1 → Run 2 ...").
"""

import re

from app import store
from app.contracts import Level, Run
from app.physics.whycard import why_card

INTENTS = [
    ("out_of_scope", r"initial speed|v0|v₀|start(ing)? speed|push|throw|spring|block|collision|"
                     r"pendulum|air( resistance)?|bounce"),
    ("recognition", r"\bread\b|recogni|detect|\bsee\b|\bsaw\b|photo|drawing|drew|defaults?|"
                    r"wrong angle|misread"),
    ("mass", r"mass|heav|light(er)?\b|weigh|\bkg\b"),
    ("gravity", r"gravit|moon|mars|earth|\bg\b"),
    ("friction", r"friction|μ|\bmu\b|rough|stuck|won'?t (slide|move)|doesn'?t move"),
    ("roll", r"roll|spin|sphere|rotat"),
    ("angle", r"angle|steep|tilt|slope|degree|°|θ|incline"),
    ("numbers", r"how (long|fast)|time|speed|velocity|seconds|result|verified|error"),
    ("formula", r"formula|equation|derive|derivation|why|explain|how does"),
]

CONCEPT_PARAM = {"mass": "mass_kg", "gravity": "gravity", "angle": "angle_deg",
                 "friction": "friction"}

EXPLAIN = {
    "mass": {
        "beginner": "A heavier ball is pulled harder by gravity, but it is also harder to speed "
                    "up — and those two effects cancel exactly. So the mass doesn't change how "
                    "fast it gets down the ramp.",
        "intermediate": "Gravity's pull along the slope is $mg\\sin\\theta$, and Newton's second "
                        "law divides by $m$: $a = g(\\sin\\theta - \\mu\\cos\\theta)$. The mass "
                        "cancels, so the time $t = \\sqrt{2L/a}$ doesn't depend on it.",
        "confident": "Along the slope: $ma = mg\\sin\\theta - \\mu mg\\cos\\theta$, so "
                     "$a = g(\\sin\\theta - \\mu\\cos\\theta)$ and $t = \\sqrt{2L/a}$. $m$ never "
                     "appears: $\\partial t/\\partial m = 0$.",
    },
    "gravity": {
        "beginner": "Gravity is what pulls the ball down the ramp. Weaker gravity, like on the "
                    "Moon, is a gentler pull, so the ball takes longer — but not as much longer "
                    "as you might guess.",
        "intermediate": "The acceleration is proportional to $g$, and $t = \\sqrt{2L/a}$, so "
                        "$t \\propto 1/\\sqrt{g}$: weaker gravity means a longer time, by the "
                        "square root of the gravity ratio.",
        "confident": "From $a = g(\\sin\\theta - \\mu\\cos\\theta)$ and $t = \\sqrt{2L/a}$: "
                     "$t \\propto g^{-1/2}$, so $t_\\text{Moon}/t_\\text{Earth} = "
                     "\\sqrt{g_\\text{Earth}/g_\\text{Moon}}$.",
    },
    "angle": {
        "beginner": "A steeper ramp turns more of gravity's pull along the slope — like a "
                    "steeper playground slide — so the ball speeds up faster and arrives sooner.",
        "intermediate": "Only the part of gravity along the slope, $g\\sin\\theta$, speeds the "
                        "ball up. A bigger $\\theta$ gives a bigger $\\sin\\theta$, a bigger $a$ "
                        "and a shorter $t = \\sqrt{2L/a}$.",
        "confident": "$a(\\theta) = g(\\sin\\theta - \\mu\\cos\\theta)$ grows with $\\theta$ on "
                     "the allowed range, and $t \\propto (\\sin\\theta - \\mu\\cos\\theta)^{-1/2}$, "
                     "so $dt/d\\theta < 0$: steeper is always faster here.",
    },
    "friction": {
        "beginner": "Friction is a backwards drag from the ramp's surface. A little friction "
                    "slows the ball down; enough friction and it doesn't slide at all.",
        "intermediate": "Friction subtracts $\\mu g\\cos\\theta$ from the pull $g\\sin\\theta$: "
                        "$a = g(\\sin\\theta - \\mu\\cos\\theta)$. When $\\mu \\ge \\tan\\theta$ "
                        "the ball stays put.",
        "confident": "The normal force is $N = mg\\cos\\theta$, kinetic friction is $\\mu N$, so "
                     "$a = g(\\sin\\theta - \\mu\\cos\\theta)$; motion needs $a > 0 \\iff "
                     "\\tan\\theta > \\mu$. (Only sliding uses $\\mu$ here.)",
    },
    "roll": {
        "beginner": "A rolling ball has to spend some of its energy on spinning, so it speeds "
                    "up a little more slowly than one that slides without friction.",
        "intermediate": "For a solid rolling ball, $a = \\tfrac{5}{7} g\\sin\\theta$ instead of "
                        "$g\\sin\\theta$ for frictionless sliding, so it takes longer.",
        "confident": "Energy: $mgh = \\tfrac{1}{2}mv^2 + \\tfrac{1}{2}I\\omega^2$ with "
                     "$I = \\tfrac{2}{5}mr^2$ and $\\omega = v/r$, so $v^2 = \\tfrac{10}{7}gh$ "
                     "and $a = \\tfrac{5}{7}g\\sin\\theta$.",
    },
}

TRY = {
    "mass": "Try the **×2 mass** button and compare the two runs.",
    "gravity": "Try the **Moon** button and compare the two runs.",
    "angle": "Drag the ramp's red handle, or try the **45°** button.",
    "friction": "Try the friction buttons (μ = 0.1, 0.3, 0.6).",
    "roll": "Try the **roll** button and compare it with sliding.",
}

OUT_OF_SCOPE = ("That's outside what this lab simulates: the ball always starts from rest on "
                "a straight ramp, so initial speeds written on the drawing (like an arrow with "
                "a speed), blocks, springs, collisions and air resistance are not modelled. "
                "What it *can* show: how the angle, gravity, friction, mass and rolling change "
                "the time to the bottom.")


def detect_intent(question: str) -> str:
    q = question.lower()
    for name, pattern in INTENTS:
        if re.search(pattern, q):
            return name
    return "general"


def _time(run: Run) -> float | None:
    v = run.verification
    return v.simulated_time_s if v.simulated_time_s is not None else v.predicted_time_s


def run_numbers(run: Run) -> str:
    v, p = run.verification, run.scene.params
    if v.predicted_time_s is None:
        return (f"In **Run {run.index}** the ball doesn't move: μ = {p.friction:g} is at least "
                f"tan θ for θ = {p.angle_deg:.1f}°.")
    sim = (f" (the formula predicts {v.predicted_time_s:.3f} s; error {v.error_pct:.2f}%)"
           if v.error_pct is not None else "")
    return (f"In **Run {run.index}** the ball took {_time(run):.3f} s to go "
            f"{run.scene.ramp_length_m:.3f} m down the ramp and reached "
            f"{(v.simulated_speed_m_s or v.predicted_speed_m_s):.3f} m/s{sim}.")


def evidence(runs: list[Run], concept: str) -> list[str]:
    """Why-card sentences for consecutive runs where exactly this concept changed."""
    out = []
    for a, b in zip(runs, runs[1:]):
        pa, pb = a.scene.params, b.scene.params
        if concept == "roll":
            hit = pa.motion != pb.motion
        else:
            key = CONCEPT_PARAM[concept]
            hit = getattr(pa, key) != getattr(pb, key)
        if hit:
            out.append(f"**Run {a.index} → Run {b.index}:** {why_card(a, b).sentence}")
    return out[-2:]


def explain_recognition(session: store.Session) -> str:
    """P3.O2: what was read from the drawing, what was assumed, how to redraw."""
    stored = session.images.get(session.current_image_id or "")
    if stored is None or stored.recognition is None:
        return "No photo yet — take a photo of your drawing first."
    rec = stored.recognition
    first = session.runs[0] if session.runs else None
    lines = ["**What I read from your drawing:**"]
    has_ramp = any(line.role == "ramp" for line in rec.lines)
    if has_ramp and first:
        lines.append(f"- a ramp at {first.scene.params.angle_deg:.1f}° with a ground line")
    elif not has_ramp:
        lines.append("- no ramp: I need one straight slanted line meeting a flat ground line")
    names = {"ball": "ball", "block": "block (a square: it slides)"}
    lines.append(f"- a {names[rec.ball.shape]} on the ramp" if rec.ball else
                 "- no ball, so I started one near the top of the ramp")
    for other in rec.other_balls:
        lines.append(f"- another {other.shape} for it to hit")
    if first:
        assumed = [k for k in first.scene.params.defaults_used if k != "ball"]
        if assumed:
            p = first.scene.params
            values = {"mass_kg": f"mass {p.mass_kg:g} kg", "gravity": f"g = {p.gravity:g} m/s²",
                      "friction": f"μ = {p.friction:g}", "motion": "sliding"}
            lines.append("**Assumed (not written on the drawing):** "
                         + ", ".join(values[k] for k in assumed) + ".")
    if rec.warnings:
        lines.append("**Problems:** " + "; ".join(rec.warnings) + ".")
    if rec.warnings or not rec.ball or not has_ramp:
        lines.append("**To redraw:** thick dark marker on plain paper, whole sheet in the "
                     "photo, one straight ramp touching a flat ground line, and one circle "
                     "(or square) resting on the ramp.")
    lines.append("Written numbers such as angles or masses aren't read yet; change them with "
                 "the sliders.")
    return "\n".join(lines)


def template_answer(session: store.Session, question: str, level: Level) -> str:
    runs = session.runs
    intent = detect_intent(question)
    if intent == "out_of_scope":
        return OUT_OF_SCOPE
    if intent == "recognition":
        return explain_recognition(session)
    if not runs:
        return "Take a photo of a ramp and a ball first — then I can explain what happens."
    current = runs[-1]
    if intent in EXPLAIN:
        parts = [EXPLAIN[intent][level]]
        seen = evidence(runs, intent)
        parts += seen if seen else [TRY[intent]]
        return "\n\n".join(parts)
    if intent == "numbers":
        return run_numbers(current)
    p = current.scene.params
    concept = "roll" if p.motion == "roll" else ("friction" if p.friction else "angle")
    return "\n\n".join([run_numbers(current), EXPLAIN[concept][level]])
