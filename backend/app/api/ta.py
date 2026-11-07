"""TA routes: ask, summary, follow-ups, quiz (Phase 3). Template mode works with no key."""

from fastapi import APIRouter, Body, Depends, HTTPException

from app import store
from app.api.runs import make_run
from app.contracts import (
    AskRequest,
    AskResponse,
    FollowUp,
    QuizAnswerRequest,
    QuizPrompt,
    QuizResult,
    SummaryRequest,
    SummaryResponse,
)
from app.ta import assistant, followups, quiz
from app.ta.summary import summarize

router = APIRouter(prefix="/ta")


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, session: store.Session = Depends(store.session_dep)):
    if not body.question.strip():
        raise HTTPException(422, "empty question")
    return assistant.ask(session, body.question.strip()[:1000], body.level)


@router.post("/summary", response_model=SummaryResponse)
def summary(body: SummaryRequest | None = Body(default=None),
            session: store.Session = Depends(store.session_dep)):
    level = body.level if body else session.profile.level
    return summarize(session, level)


@router.get("/followups", response_model=list[FollowUp])
def get_followups(session: store.Session = Depends(store.session_dep)):
    return followups.suggest(session)


@router.post("/quiz", response_model=QuizPrompt)
def quiz_prompt(session: store.Session = Depends(store.session_dep)):
    prompt = quiz.propose(session)
    if prompt is None:
        raise HTTPException(409, "run a moving ball first, then ask for a quiz")
    return prompt


@router.post("/quiz/answer", response_model=QuizResult)
def quiz_answer(body: QuizAnswerRequest, session: store.Session = Depends(store.session_dep)):
    result = quiz.answer(session, body.quiz_id, body.guess, make_run)
    if result is None:
        raise HTTPException(404, "unknown or already answered quiz")
    return result
