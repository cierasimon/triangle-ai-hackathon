"""POST /ta/ask, /ta/summary, /ta/quiz, /ta/quiz/answer (Step 0: canned template answers).

No LLM is called here yet; ta/llm.py (P3.2) adds that.
"""

from fastapi import APIRouter, Depends

from app import store
from app.contracts import (
    AskRequest,
    AskResponse,
    QuizAnswerRequest,
    QuizPrompt,
    QuizResult,
    SummaryResponse,
)

router = APIRouter(prefix="/ta")


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, session: store.Session = Depends(store.session_dep)):
    session.log("question", question=body.question, level=body.level)
    based_on = [session.runs[-1].run_id] if session.runs else []
    answer = "*(FAKE DATA: the TA is not implemented yet.)* You asked: " + body.question
    session.log("answer", answer_md=answer, based_on=based_on)
    return AskResponse(
        answer_md=answer,
        based_on=based_on,
        flags=[],
        followups=["Try the Moon — will it be faster or slower?", "What if the mass doubles?"],
    )


@router.post("/summary", response_model=SummaryResponse)
def summary(session: store.Session = Depends(store.session_dep)):
    lines = ["## Your session (FAKE DATA template)"]
    for run in session.runs:
        t = run.verification.simulated_time_s
        lines.append(f"- **Run {run.index}**: θ = {run.scene.params.angle_deg:g}°, "
                     f"t = {t:.2f} s" if t is not None else f"- **Run {run.index}**: no motion")
    return SummaryResponse(summary_md="\n".join(lines), source="template")


@router.post("/quiz", response_model=QuizPrompt)
def quiz(session: store.Session = Depends(store.session_dep)):
    return QuizPrompt(quiz_id=store.new_id("q_"), prompt="FAKE: what is t on the Moon?",
                      param="gravity", new_value=1.62)


@router.post("/quiz/answer", response_model=QuizResult)
def quiz_answer(body: QuizAnswerRequest, session: store.Session = Depends(store.session_dep)):
    actual = 1.0
    session.log("quiz", quiz_id=body.quiz_id, guess=body.guess)
    return QuizResult(actual=actual, guess=body.guess,
                      error_pct=abs(body.guess - actual) / actual * 100, run=None)
