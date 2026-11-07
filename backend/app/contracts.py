"""ALL pydantic models shared between modules (plan.md section 4).

Mirrored by hand in frontend/src/types.ts: change both in the same commit.

Coordinates: Recognition is in the warped paper image (pixels, origin top-left, y down).
Scene/Frame are physics (meters/seconds/radians, y up).
"""

from typing import Literal

from pydantic import BaseModel

Point = tuple[float, float]
Motion = Literal["slide", "roll"]
ParamSource = Literal["drawing", "label", "default", "user"]


class Line(BaseModel):
    """A straight segment of the drawn thick line."""

    p1: Point
    p2: Point
    role: Literal["ramp", "ground"]
    thickness_px: float
    confidence: float


class Ball(BaseModel):
    center: Point
    radius: float
    confidence: float
    sprite_png_b64: str | None = None


class Label(BaseModel):
    """A handwritten value such as "30°" or "2 kg"."""

    key: Literal["angle_deg", "mass_kg", "friction", "gravity"]
    value: float
    text: str
    confidence: float


class Recognition(BaseModel):
    image_id: str
    paper_found: bool
    paper_corners: list[Point] | None  # in the original photo
    warped_size: tuple[int, int]  # (w, h) of the flattened paper
    lines: list[Line]
    ball: Ball | None
    labels: list[Label] = []
    warnings: list[str] = []


class Params(BaseModel):
    angle_deg: float
    mass_kg: float = 1.0
    gravity: float = 9.81
    friction: float = 0.0  # kinetic, sliding model only
    motion: Motion
    defaults_used: list[str] = []
    sources: dict[str, ParamSource] = {}


class Scene(BaseModel):
    scale_m_per_px: float  # paper short side = 1 m
    ramp_top_m: Point
    ramp_bottom_m: Point
    ground_y_m: float
    ball_radius_m: float
    ramp_length_m: float  # distance the ball travels along the ramp
    params: Params


class Frame(BaseModel):
    t: float
    x: float
    y: float
    angle: float


class Verification(BaseModel):
    predicted_time_s: float | None  # None if the ball does not move
    simulated_time_s: float | None
    predicted_speed_m_s: float | None
    simulated_speed_m_s: float | None
    error_pct: float | None
    ok: bool  # error_pct <= 3
    formula_latex: str
    note: str | None


class Run(BaseModel):
    run_id: str  # "r1", "r2", ...
    index: int  # 1-based, shown as "Run N"
    scene: Scene
    frames: list[Frame]  # ~60 fps, until bottom or 10 s
    verification: Verification
    changed: dict[str, tuple[float, float]] = {}  # param -> (old, new) vs previous run


EventKind = Literal["upload", "run", "change", "drag_rerun", "question", "answer", "quiz", "export"]


class Event(BaseModel):
    t: float
    kind: EventKind
    data: dict


# ---- API request/response bodies -------------------------------------------------


class UploadResponse(BaseModel):
    image_id: str
    recognition: Recognition
    overlay_png_url: str


class CornersRequest(BaseModel):
    image_id: str
    corners: list[Point]  # exactly 4, in the original photo


class ParamsOverride(BaseModel):
    angle_deg: float | None = None
    mass_kg: float | None = None
    gravity: float | None = None
    friction: float | None = None
    motion: Motion | None = None


class RunRequest(BaseModel):
    image_id: str
    params_override: ParamsOverride | None = None


class PredictResponse(BaseModel):
    time_s: float | None
    speed_m_s: float | None
    will_move: bool
    note: str | None


class WhyCardRequest(BaseModel):
    run_a: str
    run_b: str


class WhyCard(BaseModel):
    latex: str
    symbol: str
    sentence: str
    cancels: bool


class LearnerProfile(BaseModel):
    level: Literal["beginner", "intermediate", "confident"] = "beginner"
    concepts_seen: dict[str, bool] = {
        "mass_independent": False,
        "angle_effect": False,
        "gravity_effect": False,
        "friction_effect": False,
    }
    runs: int = 0


class SessionState(BaseModel):
    session_id: str
    runs: list[Run]
    events: list[Event]
    profile: LearnerProfile


class AskRequest(BaseModel):
    question: str
    level: Literal["beginner", "intermediate", "confident"] = "beginner"


class AskResponse(BaseModel):
    answer_md: str
    based_on: list[str]
    flags: list[str]
    followups: list[str]


class SummaryResponse(BaseModel):
    summary_md: str
    source: Literal["llm", "template"]


class QuizPrompt(BaseModel):
    quiz_id: str
    prompt: str
    param: str
    new_value: float


class QuizAnswerRequest(BaseModel):
    quiz_id: str
    guess: float


class QuizResult(BaseModel):
    actual: float
    guess: float
    error_pct: float
    run: Run | None


class ExportResponse(BaseModel):
    gif_url: str
    share_url: str
    qr_png_url: str
