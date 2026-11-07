"""GET /session."""

from fastapi import APIRouter, Depends

from app import store
from app.contracts import SessionState

router = APIRouter()


@router.get("/session", response_model=SessionState)
def get_session(session: store.Session = Depends(store.session_dep)):
    return SessionState(
        session_id=session.session_id,
        runs=session.runs,
        events=session.events,
        profile=session.profile,
    )
