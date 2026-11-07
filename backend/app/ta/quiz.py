"""Quiz me (P3.O1): a what-if prompt, the learner guesses, the engine reruns and scores it.

Every number in the result comes from the engine run; the prompt only names the change.
"""

from app import store
from app.contracts import ParamsOverride, QuizPrompt, QuizResult, Run

# (concept, param, new value from the current params, prompt)
OPTIONS = [
    ("gravity_effect", "gravity", lambda p: 1.62 if abs(p.gravity - 1.62) > 1e-6 else 9.81,
     "What if gravity were {value:g} m/s²"),
    ("angle_effect", "angle_deg", lambda p: 45.0 if p.angle_deg < 40 else 15.0,
     "What if the ramp were at {value:g}°"),
    ("mass_independent", "mass_kg", lambda p: p.mass_kg * 2,
     "What if the ball's mass doubled to {value:g} kg"),
]


def user_override(run: Run) -> dict:
    """The params the learner has set so far (so a quiz builds on their current setup)."""
    p = run.scene.params
    return {k: getattr(p, k) for k, src in p.sources.items() if src == "user"}


def time_of(run: Run) -> float | None:
    v = run.verification
    return v.simulated_time_s if v.simulated_time_s is not None else v.predicted_time_s


def propose(session: store.Session) -> QuizPrompt | None:
    if not session.runs or session.current_image_id is None:
        return None
    run = session.runs[-1]
    if time_of(run) is None:
        return None  # nothing to compare against
    seen = session.profile.concepts_seen
    options = sorted(OPTIONS, key=lambda o: seen.get(o[0], False))  # untried concepts first
    concept, param, value_of, prompt = options[0]
    value = value_of(run.scene.params)
    quiz_id = store.new_id("q_")
    session.quizzes[quiz_id] = {
        "param": param, "value": value, "image_id": session.current_image_id,
        "base": user_override(run), "from_run": run.run_id,
    }
    text = (prompt.format(value=value)
            + f"? In Run {run.index} the ball took {time_of(run):.2f} s. "
              "Guess the new time in seconds.")
    return QuizPrompt(quiz_id=quiz_id, prompt=text, param=param, new_value=value)


def answer(session: store.Session, quiz_id: str, guess: float, make_run) -> QuizResult | None:
    quiz = session.quizzes.pop(quiz_id, None)
    if quiz is None:
        return None
    override = ParamsOverride(**{**quiz["base"], quiz["param"]: quiz["value"]})
    run = make_run(session, quiz["image_id"], override, "quiz")
    actual = time_of(run)
    if actual is None:  # can't happen for the offered options, but never divide by None
        actual = 0.0
    error = abs(guess - actual) / actual * 100 if actual else 100.0
    session.log("quiz", quiz_id=quiz_id, guess=guess, actual=round(actual, 3),
                error_pct=round(error, 1), run_id=run.run_id)
    return QuizResult(actual=actual, guess=guess, error_pct=error, run=run)
