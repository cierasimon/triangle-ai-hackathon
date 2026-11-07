"""In-memory sessions, runs, events and uploaded files. Lost on restart (fine for a demo)."""

import secrets
import time
from dataclasses import dataclass, field

from fastapi import Cookie, Header, Response

from app.contracts import Event, EventKind, LearnerProfile, Recognition, Run

SESSION_COOKIE = "pp_session"


@dataclass
class StoredImage:
    image_id: str
    content_type: str
    original: bytes
    overlay_png: bytes | None = None
    overlay_version: int = 0  # bumped when /corners redoes recognition (cache-busting URL)
    recognition: Recognition | None = None


@dataclass
class Session:
    session_id: str
    images: dict[str, StoredImage] = field(default_factory=dict)
    runs: list[Run] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    profile: LearnerProfile = field(default_factory=LearnerProfile)
    started: float = field(default_factory=time.time)
    current_image_id: str | None = None  # the latest upload; runs and the TA use it
    quizzes: dict[str, dict] = field(default_factory=dict)  # quiz_id -> pending what-if

    def log(self, kind: EventKind, **data) -> Event:
        event = Event(t=round(time.time() - self.started, 3), kind=kind, data=data)
        self.events.append(event)
        return event

    def next_run_id(self) -> tuple[str, int]:
        index = len(self.runs) + 1
        return f"r{index}", index

    def get_run(self, run_id: str) -> Run | None:
        return next((r for r in self.runs if r.run_id == run_id), None)


@dataclass
class Share:
    share_id: str
    data: bytes  # GIF, or a PNG still if the GIF failed
    media_type: str
    run_id: str
    title: str  # "Run 2"
    lines: tuple[str, ...] = ()  # card text shown on the download page


_sessions: dict[str, Session] = {}
_shares: dict[str, Share] = {}


def new_id(prefix: str = "") -> str:
    return prefix + secrets.token_urlsafe(8)


def get_or_create(session_id: str | None) -> Session:
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    sid = session_id or new_id("s_")
    session = Session(session_id=sid)
    _sessions[sid] = session
    return session


def session_dep(
    response: Response,
    x_session: str | None = Header(default=None),
    pp_session: str | None = Cookie(default=None),
) -> Session:
    """FastAPI dependency: session from the X-Session header, else the cookie, else a new one."""
    session = get_or_create(x_session or pp_session)
    response.set_cookie(SESSION_COOKIE, session.session_id, httponly=True, samesite="lax")
    response.headers["X-Session"] = session.session_id
    return session


def put_share(share: Share) -> None:
    _shares[share.share_id] = share


def get_share(share_id: str) -> Share | None:
    return _shares.get(share_id)


def reset() -> None:
    """Clear everything (tests)."""
    _sessions.clear()
    _shares.clear()
