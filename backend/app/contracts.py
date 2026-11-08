"""ALL pydantic models shared between modules (plan.md section 4).

Mirrored by hand in frontend/src/types.ts: change both in the same commit.

Coordinates: Recognition is in the warped paper image (pixels, origin top-left, y down).
Scene/Frame are physics (meters/seconds/radians, y up).
"""

from typing import Literal

from pydantic import BaseModel, Field

Point = tuple[float, float]
Motion = Literal["slide", "roll"]
Shape = Literal["ball", "block"]  # a drawn circle, or a drawn square (always slides)
Level = Literal["beginner", "intermediate", "confident"]
ParamSource = Literal["drawing", "label", "default", "user"]


class Line(BaseModel):
    """A straight segment of the drawn thick line."""

    p1: Point
    p2: Point
    role: Literal["ramp", "ground"]
    thickness_px: float
    confidence: float


class Ball(BaseModel):
    """A drawn body: a circle, or a square block (radius = half its side)."""

    center: Point
    radius: float
    confidence: float
    shape: Shape = "ball"
    sprite_png_b64: str | None = None


class DetectedObject(BaseModel):
    kind: Literal["ball", "block", "ramp", "spring", "platform", "wall"]
    bbox: tuple[float, float, float, float]
    confidence: float
    geometry_verified: bool = False
    body_id: str | None = None


class Block(BaseModel):
    vertices: list[Point] = Field(min_length=4, max_length=4)
    center: Point
    confidence: float


class DiagramParameter(BaseModel):
    key: str
    value: float | None = Field(default=None, allow_inf_nan=False)
    unit: str = ""
    text: str
    target: str = "scene"  # scene, ball-1, block-1, or unresolved
    confidence: float = Field(ge=0, le=1)
    validated: bool = False  # Recognized and eligible; does not imply physics support.
    applied: bool = False
    note: str = ""


class Label(BaseModel):
    """A handwritten value such as "30°" or "2 kg"."""

    key: Literal["angle_deg", "mass_kg", "friction", "gravity", "initial_speed_m_s"]
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
    balls: list[Ball] = []  # All detected balls; ball remains the selected simulation object.
    blocks: list[Block] = []
    parameters: list[DiagramParameter] = []
    selected_ball_index: int | None = None
    objects: list[DetectedObject] = []
    recognition_mode: Literal["opencv", "hybrid"] = "opencv"
    gemini_status: Literal["disabled", "success", "failed", "unavailable"] = "disabled"
    gemini_model: str | None = None
    gemini_summary: str | None = None
    other_balls: list[Ball] = []  # more balls on the ramp or the ground, hit by the main one
    other_speeds: list[float] = []  # initial speed read for each of other_balls (see OtherBall)
    labels: list[Label] = []
    warnings: list[str] = []


class Params(BaseModel):
    angle_deg: float
    mass_kg: float = 1.0
    gravity: float = 9.81
    friction: float = 0.0  # kinetic, sliding model only
    initial_speed_m_s: float = 0.0  # + down the ramp, - up the ramp
    motion: Motion
    defaults_used: list[str] = []
    sources: dict[str, ParamSource] = {}


class OtherBall(BaseModel):
    """A ball at rest that the main ball can hit."""

    surface: Literal["ramp", "ground"]
    at: float  # ramp: fraction of the ramp from the bottom (0) to the top (1); ground: x (m)
    shape: Shape = "ball"
    radius_m: float  # half the side for a block
    mass_kg: float
    speed_m_s: float = 0.0  # initial speed: on the ramp + down / - up; on the ground + right / - left
    center_m: Point  # resting position for this scene's ramp


class Scene(BaseModel):
    scale_m_per_px: float  # paper short side = 1 m
    ramp_top_m: Point
    ramp_bottom_m: Point
    ground_y_m: float
    ball_radius_m: float  # half the side for a block
    ball_shape: Shape = "ball"
    ramp_length_m: float  # distance the ball travels along the ramp
    params: Params
    others: list[OtherBall] = []
    ground_end_x_m: float | None = None  # the flat ground ends here, with a stop the balls halt at


class Frame(BaseModel):
    t: float
    x: float
    y: float
    angle: float
    others: list[tuple[float, float, float]] = []  # (x, y, angle) per Scene.others


class Collision(BaseModel):
    """The first hit between the main ball and another ball, from the simulation."""

    t: float
    other: int  # index into Scene.others
    v1_before: float  # speeds (m/s)
    v1_after: float
    v2_before: float
    v2_after: float


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
    frames: list[Frame]  # ~60 fps: down the ramp and along the ground, until everything stops
    verification: Verification
    changed: dict[str, tuple[float, float]] = {}  # param -> (old, new) vs previous run
    collision: Collision | None = None


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
    initial_speed_m_s: float | None = None  # + down the ramp, - up the ramp
    other_speeds_m_s: list[float | None] | None = None  # same, for each other body (B, C, ...)
    start_fraction: float | None = None  # ball start along the ramp: 0 = bottom, 1 = top
    ramp_full_m: float | None = None  # whole ramp length; the ball keeps its place on it
    other_mass_kg: float | None = None  # mass of every other ball
    other_surface: Literal["ramp", "ground"] | None = None  # the learner dragged the 2nd object:
    other_at: float | None = None  # ...to here (as OtherBall.at); both or neither


class RunRequest(BaseModel):
    image_id: str
    params_override: ParamsOverride | None = None
    reason: Literal["change", "drag"] = "change"  # "drag" logs a drag_rerun event


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
    level: Level = "beginner"
    concepts_seen: dict[str, bool] = {
        "mass_independent": False,
        "angle_effect": False,
        "gravity_effect": False,
        "friction_effect": False,
        "roll_effect": False,
    }
    runs: int = 0


class SessionState(BaseModel):
    session_id: str
    runs: list[Run]
    events: list[Event]
    profile: LearnerProfile


class AskRequest(BaseModel):
    question: str
    level: Level = "beginner"


class FollowUp(BaseModel):
    """A suggested next experiment; clicking it applies `change` as a new run."""

    text: str
    change: ParamsOverride
    concept: str


class AskResponse(BaseModel):
    answer_md: str
    based_on: list[str]  # run ids the answer relies on, e.g. ["r1", "r2"]
    flags: list[str]  # numbers in the answer that match nothing in the context, as written
    followups: list[FollowUp]
    suggestions: list[str] = []  # three likely next questions, shown as bubbles the learner can click
    source: Literal["llm", "template"] = "template"


class SummaryRequest(BaseModel):
    level: Level = "beginner"


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


class ExportRequest(BaseModel):
    """The run as the page shows it, so export works even after the server forgot it
    (sessions live in memory: a restart or a dev reload clears them)."""

    run: Run | None = None


class ExportResponse(BaseModel):
    gif_url: str
    share_url: str
    qr_png_url: str
