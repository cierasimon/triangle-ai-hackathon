"""Two follow-up experiments from concepts the learner hasn't tried yet (P3.6).

Each is a fixed template with a concrete change; clicking it runs that change.
"""

from app import store
from app.contracts import FollowUp, ParamsOverride, Run

ORDER = ["mass_independent", "gravity_effect", "angle_effect", "friction_effect", "roll_effect"]


def _option(concept: str, run: Run) -> FollowUp | None:
    p = run.scene.params
    if concept == "mass_independent":
        return FollowUp(text="Double the mass — will it reach the bottom sooner?",
                        change=ParamsOverride(mass_kg=p.mass_kg * 2), concept=concept)
    if concept == "gravity_effect":
        if abs(p.gravity - 1.62) < 1e-6:
            return FollowUp(text="Back to Earth — how much faster than on the Moon?",
                            change=ParamsOverride(gravity=9.81), concept=concept)
        return FollowUp(text="Try the Moon — will it be faster or slower?",
                        change=ParamsOverride(gravity=1.62), concept=concept)
    if concept == "angle_effect":
        if p.angle_deg >= 40:
            return FollowUp(text="Make the ramp gentler (15°) — how much slower?",
                            change=ParamsOverride(angle_deg=15), concept=concept)
        return FollowUp(text="Make the ramp steeper (45°) — how much faster?",
                        change=ParamsOverride(angle_deg=45), concept=concept)
    if concept == "friction_effect":
        if p.motion == "roll":
            return None  # friction only matters for sliding in this model
        mu = 0.3 if p.friction < 0.3 else 0.0
        text = ("Add friction (μ = 0.3) — does it still slide?" if mu else
                "Remove the friction — how much faster?")
        return FollowUp(text=text, change=ParamsOverride(friction=mu), concept=concept)
    if concept == "roll_effect":
        if p.motion == "roll":
            return FollowUp(text="Let it slide instead of roll — faster or slower?",
                            change=ParamsOverride(motion="slide"), concept=concept)
        return FollowUp(text="Let it roll instead of slide — faster or slower?",
                        change=ParamsOverride(motion="roll"), concept=concept)
    return None


def suggest(session: store.Session, n: int = 2) -> list[FollowUp]:
    if not session.runs:
        return []
    run = session.runs[-1]
    seen = session.profile.concepts_seen
    fresh = [c for c in ORDER if not seen.get(c)]
    rest = [c for c in ORDER if seen.get(c)]
    out = []
    for concept in fresh + rest:
        option = _option(concept, run)
        if option:
            out.append(option)
        if len(out) == n:
            break
    return out
