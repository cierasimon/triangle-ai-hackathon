# PaperPhysics — Implementation Plan (plan.md)

Source of truth for scope: `PaperPhysics Final Plan (English).pdf`. This file turns it into
tasks that coding agents (Claude Code / Codex) can pick up one at a time. Timeline, hours and
pitch are out of scope here.

**Product in one line:** draw a ramp and a ball on paper → take a photo → it becomes a running
simulation, checked against the textbook formula → drag the ramp, change mass/gravity, see
why → ask an AI TA about it → scan a QR code to take the animation home.

---

## 0. Ground rules for every agent

1. **Scope is one situation: a ball on a ramp** (plus a flat ground line). No blocks, springs,
   collisions, pendulums or velocity arrows. Code is written so a situation is a plug-in
   (detector + scene builder + formulas + why-cards), but only `ball_on_ramp` is built.
2. **Never break the demo.** After every merge, `make dev` must still run photo → simulation.
   Half-built features go behind a flag.
3. **Stay in your folder.** Each task lists the folders it owns. Touch files outside them only
   to wire things up, and say so in the PR/commit message.
4. **Contracts are law.** Modules talk only through the models in `backend/app/contracts.py`
   (section 4). Changing a contract is its own small commit that updates every user of it.
5. **Prove "done" with a command.** Every task has acceptance checks; add them as tests
   (`pytest`) or as a script, and run them before calling the task done.
6. **No paid API calls, ever.** LLM model ids must end in `:free` or be `openrouter/free`; the
   LLM client refuses anything else (section 3).
7. **Never read, print or commit `.env`.** Use `.env.example` for documentation.
8. **The physics engine is the only source of numbers.** The LLM may phrase things, never
   compute them.
9. **Don't stop for the human; leave a note.** Tasks marked `📝 HUMAN TODO` need something only
   the human can do (confirming angles, more photos, phone testing). Don't wait: add a line
   to `HUMAN_TODO.md` (what to do, why, which task it unblocks) and keep building. Never fake
   the human's part: don't treat synthetic images as real photos, don't mark a phone test as
   passed, and label any unconfirmed numbers as "unconfirmed" in reports.

---

## 1. Stack and repo layout

One app, one server, one URL (simpler for agents, for the phone, and for the QR tunnel):

- **Backend:** Python 3.12 (managed with `uv`; the system Python is 3.9 — don't use it),
  FastAPI + Uvicorn, OpenCV, NumPy, pymunk, SymPy, Pillow, httpx, qrcode.
- **Frontend:** Vite + React + TypeScript, KaTeX, react-markdown. Built to `frontend/dist`
  and served by FastAPI at `/`. In dev, Vite proxies `/api` to the backend.
- **Simulation runs on the backend**; the frontend only plays back the returned trajectory.

```text
triangle-ai-hackathon/
├── AGENTS.md               # rules above + how to run/test (CLAUDE.md just points here)
├── CLAUDE.md
├── plan.md
├── Makefile                # setup, dev, test, eval, data, build, tunnel
├── .env.example
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py         # FastAPI app, mounts routers + static frontend
│   │   ├── config.py       # reads env
│   │   ├── contracts.py    # ALL pydantic models shared between modules
│   │   ├── store.py        # in-memory sessions, runs, events, uploaded files
│   │   ├── api/            # routers: upload, runs, predict, ta, export
│   │   ├── cv/             # capture.py, recognize.py, sprite.py, labels.py
│   │   ├── scene/          # builder.py (pixels → meters, Scene)
│   │   ├── physics/        # simulate.py (pymunk), formulas.py (SymPy), whycard.py
│   │   ├── ta/             # llm.py, context.py, prompts.py, numcheck.py, summary.py, followups.py, quiz.py
│   │   └── export/         # gif.py, share.py
│   └── tests/
├── frontend/
│   └── src/ (components/, api.ts, types.ts)   # types.ts mirrors contracts.py
├── fixtures/
│   ├── synthetic/          # ~20 small images + expected JSON, committed (from dataset)
│   └── real/               # phone photos + labels.json (added by humans, see §6)
├── scripts/
│   ├── data/paperphysics_generate_v3.py   # existing
│   ├── data/fetch_dataset.sh              # downloads the release zip
│   ├── eval_cv.py                         # prints recognition accuracy
│   └── spike_roll_vs_slide.py
└── data/generated/         # gitignored: full synthetic dataset lives here
```

---

## 2. Environment

`.env` (already exists locally, holds the key) / `.env.example` (committed, no values):

```bash
OPENROUTER_API_KEY=
LLM_PROVIDER=openrouter            # "openrouter" or "none" (template mode)
LLM_MODEL=nvidia/nemotron-3.5-super-vl:free
LLM_FALLBACK_MODEL=openrouter/free
LLM_VISION_MODEL=nvidia/nemotron-3.5-super-vl:free
PUBLIC_BASE_URL=http://localhost:8000   # set to the tunnel URL for QR codes
```

Setup commands the scaffold must support:

```bash
make setup   # uv venv --python 3.12, uv pip install -e backend[dev], npm ci in frontend
make dev     # backend on :8000 with reload + vite on :5173
make test    # pytest + frontend typecheck
make eval    # python scripts/eval_cv.py
make data    # fetch the synthetic dataset into data/generated/
make tunnel  # cloudflared quick tunnel to :8000 (brew install cloudflared)
```

---

## 3. LLM via OpenRouter (free models only)

Tested on 2026-11-07 with the project key:

| Model | Result |
|---|---|
| `nvidia/nemotron-3.5-super-vl:free` | Works (~3 s), reads images too → **primary + vision** |
| `openrouter/free` | Works (~5 s), routes to some free model → **fallback** |
| `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free` | 429 rate-limited upstream at test time |

`backend/app/ta/llm.py` requirements:

- OpenAI-compatible POST to `https://openrouter.ai/api/v1/chat/completions` with httpx.
- **Free-only guard:** raise if the model id doesn't end in `:free` and isn't `openrouter/free`.
- These are reasoning models: set `max_tokens` ≥ 1000, and treat an empty/`null` `content` as a
  failure (it means the budget went to reasoning).
- On 429/5xx/timeout (8 s): try `LLM_FALLBACK_MODEL` once, then return `None`.
- Callers treat `None` as "use template mode". With `LLM_PROVIDER=none` or no key, nothing is
  ever sent and the app still works end to end.
- Free models are rate-limited: never call the LLM in a loop, on every drag, or in tests
  (tests mock `llm.py`; one opt-in test `pytest -m live_llm` hits the real API).

---

## 4. Contracts (`backend/app/contracts.py`, mirrored in `frontend/src/types.ts`)

All coordinates in the **warped paper image**, pixels, origin top-left, y down. All physics in
meters/seconds, y up.

```python
class Line(BaseModel):        # a straight segment of the drawn thick line
    p1: tuple[float, float]
    p2: tuple[float, float]
    role: Literal["ramp", "ground"]
    thickness_px: float
    confidence: float

class Ball(BaseModel):
    center: tuple[float, float]
    radius: float
    confidence: float
    sprite_png_b64: str | None = None     # optional cutout (Phase 1 optional)

class Label(BaseModel):                   # handwritten "30°", "2 kg"
    key: Literal["angle_deg", "mass_kg", "friction", "gravity"]
    value: float
    text: str
    confidence: float

class Recognition(BaseModel):
    image_id: str
    paper_found: bool
    paper_corners: list[tuple[float, float]] | None   # in the original photo
    warped_size: tuple[int, int]                      # (w, h) of the flattened paper
    lines: list[Line]
    ball: Ball | None
    labels: list[Label] = []
    warnings: list[str] = []                          # human-readable problems

class Params(BaseModel):
    angle_deg: float
    mass_kg: float = 1.0
    gravity: float = 9.81
    friction: float = 0.0          # kinetic, sliding model only
    motion: Literal["slide", "roll"]
    defaults_used: list[str] = []  # e.g. ["mass_kg", "friction"]
    sources: dict[str, Literal["drawing", "label", "default", "user"]] = {}

class Scene(BaseModel):
    scale_m_per_px: float          # paper short side = 1 m
    ramp_top_m: tuple[float, float]
    ramp_bottom_m: tuple[float, float]
    ground_y_m: float
    ball_radius_m: float
    ramp_length_m: float           # distance the ball travels along the ramp
    params: Params

class Frame(BaseModel):
    t: float; x: float; y: float; angle: float   # ball pose in meters/radians

class Verification(BaseModel):
    predicted_time_s: float | None   # None if the ball does not move
    simulated_time_s: float | None
    predicted_speed_m_s: float | None
    simulated_speed_m_s: float | None
    error_pct: float | None
    ok: bool                         # error_pct <= 3
    formula_latex: str
    note: str | None                 # e.g. "ball will not slide: μ ≥ tan θ"

class Run(BaseModel):
    run_id: str                      # "r1", "r2", ...
    index: int                       # 1-based, shown as "Run N"
    scene: Scene
    frames: list[Frame]              # ~60 fps, until bottom or 10 s
    verification: Verification
    changed: dict[str, tuple[float, float]] = {}   # param -> (old, new) vs previous run

class Event(BaseModel):
    t: float
    kind: Literal["upload", "run", "change", "drag_rerun", "question", "answer", "quiz", "export"]
    data: dict
```

**API** (all under `/api`, session id in a cookie or header `X-Session`):

| Method & path | Body → Response |
|---|---|
| `POST /upload` | photo (multipart) → `{image_id, recognition: Recognition, overlay_png_url}` |
| `POST /corners` | `{image_id, corners[4]}` → `Recognition` (manual four-corner fallback) |
| `POST /runs` | `{image_id, params_override?}` → `Run` |
| `GET /predict?angle_deg&mass_kg&gravity&friction&motion&length_m` | formula-only `{time_s, speed_m_s, will_move, note}`, must answer in < 20 ms |
| `POST /whycard` | `{run_a, run_b}` → `{latex, symbol, sentence, cancels: bool}` |
| `GET /session` | runs, events, learner profile |
| `POST /ta/ask` | `{question, level}` → `{answer_md, based_on: ["r1"], flags: [...], followups: [..2]}` |
| `POST /ta/summary` | → `{summary_md, source: "llm" | "template"}` |
| `POST /ta/quiz` / `POST /ta/quiz/answer` | what-if prompt / `{guess}` → engine result + error % |
| `POST /export/{run_id}` | → `{gif_url, share_url, qr_png_url}` |
| `GET /s/{share_id}` | mobile download page |

---

## 5. Data

- **Synthetic dataset** (teammate's release, 3,000 images, ~31 MB):
  `https://github.com/cierasimon/triangle-ai-hackathon/releases/download/dataset-v3/paperphysics_v3_release.zip`
  → `scripts/data/fetch_dataset.sh` downloads and unzips into `data/generated/` (gitignored).
  Alternative with no download: `python scripts/data/paperphysics_generate_v3.py --count 3000 --output data/generated/paperphysics_v3`.
- **Only the `ball_plain` scenes (375) match our scope.** Use them for development and the
  automatic accuracy check. Ignore block/spring/collision scenes (or use them as "should not
  crash, should warn" negatives).
- Synthetic images are flat renders, not photos: there's no paper edge to find. Capture must
  fall back to "use the whole image" when no paper quad is found.
- Ground truth for scoring comes from `annotations.jsonl` (ball box/polygon, ramp polygon,
  `physics.angle_degrees`). **First check** whether the drawn ramp slope actually equals
  `angle_degrees` (it may be a printed label, not the drawn geometry); if not, score angle
  against the ramp polygon instead and note it in `scripts/eval_cv.py`.
- Synthetic images do **not** prove real-photo accuracy. Real photos go in `fixtures/real/`.

### 5.1 What the first 5 real photos showed (`fixtures/real/IMG_9140–9144.jpg`)

These are realistic textbook-style sketches and much harder than the synthetic set. P1.1/P1.2
must handle them (or warn clearly), not just the "ideal" sketch:

1. **Ruled notebook paper**: light-blue lines everywhere and a red margin line. A plain
   grayscale threshold will pick them up as ink. Extract ink by **color + darkness** (pen is
   dark navy; rules are light cyan/pink), then remove long, thin, evenly spaced parallel lines.
2. **Blue ballpoint, not thick black marker**: strokes are thin. "Thick vs thin" by stroke
   width is unreliable; use length/straightness instead (the ramp, ground and wall are the
   longest straight strokes).
3. **The drawing is sideways**: the phone was held in portrait while the notebook was
   landscape, so the **ground is a vertical line** in most photos. Find the upright
   orientation: try 0/90/180/270°, and pick the one where a long line is horizontal at the
   bottom and the right-angle mark sits on it (text orientation, if read by the vision model,
   is a tie-breaker). Photos are already EXIF-corrected.
4. **Full right triangles**: they draw ground + vertical wall + ramp (hypotenuse), plus a
   right-angle mark and an angle arc. Wall = the line perpendicular to ground; ramp = the
   remaining long slanted line. Ignore arcs, right-angle marks, arrows and text.
5. **Extra objects**: blocks (squares) on or beside the ramp, two balls (9141), no ball at
   all (9143, block only), a ball floating above the ramp (9144). Rules: pick the ball
   closest to the ramp top; snap a floating ball onto the ramp; if there's no circle, return
   `ball=None` with the warning "no ball found — draw a circle on the ramp"; ignore squares.
6. **Written angle ≠ drawn angle** (35° written vs ~25° drawn; 65° vs ~60°). Show both
   ("drawn 25°, written 35°"), default to the **written** value when the vision model reads it,
   and let the user switch.
7. **Labels the scope doesn't use**: "a = 5 m/s" (initial speed) appears on every sketch.
   Read it but ignore it, and say "initial speed is not simulated" in the UI/TA.
8. **No paper edge visible** (the sheet fills the frame) and **shadows/page curl** (9141,
   9142, 9143). Paper detection will usually fail → whole-image fallback; the adaptive
   threshold must survive shadows.

`fixtures/real/labels.json` holds a first draft of the ground truth (written labels and rough
drawn angles from eyeballing). `angle_deg` is null until an agent measures it from the overlay
and the human confirms. For the **demo**, still prepare 3 "ideal" sketches (black marker,
plain paper, upright, one ball on the ramp); the 5 photos above are the robustness test set.

`fixtures/real/labels.json` format (humans fill this; agents can draft it from the overlay and
a human confirms):

```json
[{"file": "IMG_0001.jpg", "ball": true, "ramp": true, "ground": true,
  "angle_deg": 32, "labels": {"angle_deg": 30}, "notes": "thick marker, desk lamp"}]
```

---

## 6. Build order

Tasks marked ∥ can run in parallel once their dependencies are merged. Each has **Owns**,
**Needs**, **Done when**.

### Step 0 — Scaffold (one agent, alone, first)

- **Owns:** everything in section 1 that doesn't exist yet.
- Create the layout, `pyproject.toml`, Vite app, Makefile, `.env.example`, `AGENTS.md`
  (section 0 + run/test commands), `CLAUDE.md` (one line: "Read AGENTS.md"), `contracts.py`
  + `types.ts`, `store.py`, empty routers that return **fake data matching the contracts**,
  and `fetch_dataset.sh`. Copy ~20 `ball_plain` images + expected JSON into
  `fixtures/synthetic/`.
- **Done when:** `make setup && make test` pass on a clean clone; `make dev` shows a page that
  uploads a photo and displays fake recognition + a fake run playing; README says how to run.

### Step 0.5 — Spikes (∥, right after Step 0)

**S1 Roll vs slide** — `scripts/spike_roll_vs_slide.py`. pymunk is 2D, so a circle's default
moment is a **disk** (I = ½mr²) and rolling gives a = (2/3)g sin θ, not the plan's (5/7)g sin θ
(solid sphere). Test three setups at 15/30/45°, L = 0.5 m:
1. Slide: `moment = inf` (no rotation), ramp friction 1, ball friction μ → compare with
   a = g(sin θ − μ cos θ).
2. Roll as disk: default moment, high friction → compare with (2/3)g sin θ.
3. Roll as sphere: `moment = 0.4·m·r²`, high friction → compare with (5/7)g sin θ.

Measure time-to-bottom error with dt = 1/240 s. Pick the model(s) within 3%. **Default to
slide** for the demo unless a rolling variant is clearly within 3%; offer roll as a toggle if it
works. Write the result at the top of `physics/simulate.py`.

**S2 Phone drag** — `📝 HUMAN TODO` (a human tries it on a phone later; build P2.2 anyway): a throwaway page with a draggable handle using
Pointer events + `touch-action: none`; open it on a phone through `make dev` on the LAN.

### Phase 1 — Photo → verified simulation (the floor, never cut)

**Seed CV pipeline (read before P1.1/P1.2).** `scripts/cv/vision_pipeline_seed.py` is a
teammate's working OpenCV pipeline: flatten → clean → Hough ramp → erase lines → classify
components (ball/block/spring/text) → merge text boxes, with every setting in one `params`
dict marked `# tune`, debug images, and optional AI label reading. **Port it into
`backend/app/cv/` instead of writing capture/recognition from scratch**; keep the `params`
dict, the debug-image output and its CLI (move it to `scripts/cv/debug_cv.py`). Leave the
seed file untouched as reference. Required changes while porting:

1. **No Gemini.** Replace `enrich_with_gemini` (google-genai, `GEMINI_API_KEY`,
   `gemini-2.5-flash`) with a call through `ta/llm.py` using `LLM_VISION_MODEL` (free
   OpenRouter model). Keep its prompt, JSON schema and the "never change geometry / only
   known IDs / only allowed roles" checks. No key → skip silently, add a warning. Do not add
   `google-genai` as a dependency.
2. **Ruled paper:** `clean()` thresholds grayscale, so notebook rules become ink and Hough
   finds them as lines. Build the ink mask from color + darkness and remove long, evenly
   spaced parallel rules (§5.1 item 1).
3. **Orientation:** ground is assumed horizontal in the lower half and the ramp 8–60° from
   horizontal. Add the 0/90/180/270° upright search (§5.1 item 3) before detection, and widen
   `ramp_angle_max` to 75° (IMG_9142 is drawn at ~60°).
4. **Scale-relative sizes:** `ball_diam_*`, `block_side_*`, `hough_min_len`, `glyph_*`,
   `erase_flat_min_len` etc. are fixed pixels tuned for 768×576 synthetic images. Express
   them as fractions of the image's short side (or resize every input to a fixed short side
   first) so 2000 px photos work.
5. **Paper not found:** `flatten()` raises in photo mode. Instead fall back to the whole
   image, `paper_found=false`, plus a warning; keep manual corners (`POST /corners`).
6. **Aspect ratio:** `flatten()` always warps to landscape letter 1100×850. Use the detected
   quad's own aspect ratio and orientation.
7. **Contract:** map its typed `objects` output to `Recognition` (§4): ramp/ground segments →
   `lines`, chosen circle → `ball`, the rest → `warnings` (e.g. "block found — only balls
   are simulated"). Keep block detection (needed for IMG_9143). Springs, arrows and pendulums
   may stay in the code but are not used downstream.
8. **Run detection once:** `save_debug()` calls `detect()` up to three times per image;
   compute once and reuse.
9. Its comments mention `eval_dataset.py --split val`; if the teammate's script exists, use it
   as the starting point for `scripts/eval_cv.py`.

**P1.1 Capture ∥** — Owns `cv/capture.py`. Needs Step 0. Port `ingest`, `find_paper`,
`order_corners`, `flatten`, `clean` from the seed with changes 2, 5, 6 above.
- Find the paper (largest 4-point contour after edge detection/threshold), perspective-warp so
  the short side is horizontal or vertical as photographed, resize long side to ~1200 px.
- No paper found → use the whole image, set `paper_found=false`, add a warning.
- Ink mask: adaptive threshold on grayscale (handles shadows), small morphological cleanup.
- Connected components → strokes, drop specks.
- `apply_corners(image, corners)` for the manual fallback.
- **Done when:** on `fixtures/synthetic/` returns a sensible mask for every image; on a
  synthetic image pasted onto a darker background with a perspective tilt (generate in the
  test), the warp recovers the page within 3% of its edge lengths.

**P1.2 Recognition ∥** — Owns `cv/recognize.py`, `scripts/eval_cv.py`. Needs Step 0 (uses
capture output; stub it until P1.1 lands). Port the rest of the seed (Hough ramp, line
erasing, component classification, text boxes, `recognize`) with changes 3, 4, 7, 8 above;
the bullets below describe the intended behaviour.
- Ball: closed stroke with high circularity (4πA/P² > ~0.7) or `HoughCircles` on the mask;
  return the best one.
- Lines: thick strokes (distance transform → median stroke width) → skeleton or
  `approxPolyDP` → straight segments. Near-horizontal (< 5°) → `ground`; the steepest long
  segment → `ramp`. A single connected "ramp + ground" stroke must be split into both.
- Ignore thin strokes (handwriting) for geometry; pass their boxes to label reading.
- Angle = slope of the ramp segment against the warped image's horizontal.
- Fill `warnings` (e.g. "no ball found", "two possible ramps").
- `scripts/eval_cv.py --set synthetic|real` prints: ball found %, ramp found %, angle mean
  abs error (deg), overall "all objects correct" %.
- Must handle the real-photo issues in §5.1 (ruled paper, sideways drawings, full triangles,
  extra blocks, multiple/missing/floating balls).
- **Done when:** on the `ball_plain` synthetic set: all-objects-correct ≥ 95%, angle error
  ≤ 2° mean; and on `fixtures/real/` it finds the right ground and ramp in at least 4 of 5
  photos and gives the expected warning for 9143 (no ball). (Full real-photo target is §7
  Checkpoint A.)

**P1.3 Scene builder ∥** — Owns `scene/builder.py`. Needs contracts only.
- Paper short side = 1 m → `scale_m_per_px`. Flip y. Ramp top = higher endpoint. Ball start
  = projection of the ball onto the ramp, resting on it (center offset by r along the normal).
  `ramp_length_m` = distance from the ball's contact point to the ramp bottom.
- Params: angle from the drawing, labels override (`source="label"`), else defaults
  (mass 1 kg, g 9.81, μ 0.0 for slide) and list them in `defaults_used`.
- Clamp angle to 5–75°; warn outside.
- **Done when:** unit tests on hand-written `Recognition` objects give exact expected meters
  and angles.

**P1.4 Simulation ∥** — Owns `physics/simulate.py`. Needs S1.
- `simulate(scene) -> list[Frame], time_to_bottom, final_speed`. Static segment for ramp and
  ground, dynamic circle, dt 1/240, record every 4th step (60 fps), stop at the ramp bottom
  (log the crossing time by interpolation) or 10 s. Ball doesn't move (μ ≥ tan θ) → stop early.
- Mass genuinely passed to pymunk (so "double the mass" is a real rerun).
- **Done when:** for θ ∈ {15, 30, 45, 60}, μ ∈ {0, 0.2}, g ∈ {9.81, 1.62}, simulated time is
  within 3% of the formula; runs in < 300 ms.

**P1.5 Verification ∥** — Owns `physics/formulas.py`. Needs contracts only.
- SymPy symbols θ, μ, g, m, L. Slide: a = g(sin θ − μ cos θ); roll-disk a = (2/3)g sin θ;
  roll-sphere a = (5/7)g sin θ. t = √(2L/a), v = a·t. `lambdify` each to a plain function for
  `/predict`.
- `verify(scene, sim) -> Verification`, LaTeX for the active formula, `ok = error_pct ≤ 3`.
- `will_move`: slide requires tan θ > μ.
- **Done when:** tests assert `diff(t, m) == 0`, Moon/Earth time ratio = √(9.81/1.62) ≈ 2.46,
  lambdified `/predict` < 1 ms per call.

**P1.6 API wiring** — Owns `api/upload.py`, `api/runs.py`, `api/predict.py`, `store.py`.
Needs P1.1–P1.5.
- Upload → capture → recognize → store; return overlay PNG (outlines drawn on the warped
  photo: ramp red, ground blue, ball green). Runs → scene → simulate → verify → store, log
  events.
- **Done when:** an integration test posts each `fixtures/synthetic/` image and gets a `Run`
  with `verification.ok == true`.

**P1.7 Dashboard ∥** — Owns `frontend/`. Needs Step 0 (works against fake data first).
- Mobile-friendly upload: `<input type="file" accept="image/*" capture="environment">`.
- Left: photo with recognized outlines. Right: canvas playing the run (ramp, ground, ball;
  Play / Reset). Parameter panel (angle, mass, gravity, friction, motion; each with its source
  badge; "defaults used" notice). Predicted vs simulated table with error %, red if > 3%.
  KaTeX formula. Recognition warnings shown plainly.
- **Done when:** works at 375 px and laptop widths; full flow against the real backend on a
  synthetic image.

**P1.8 Eval & fixtures** — Owns `scripts/eval_cv.py` (shared with P1.2), `fixtures/`.
- `make eval` runs synthetic and (if present) real sets and prints both accuracy tables.

**Checkpoint A (real photos) — no waiting.** Trigger: P1.1–P1.6 are merged and the synthetic
`make eval` target is met. Run recognition on the 5 photos in `fixtures/real/` (§5.1), fill
`angle_deg` in `labels.json` from the overlays (mark them `"angle_confirmed": false`), run
`make eval --set real`, tune what you can, report the accuracy, and continue to Phase 2.
Add to `HUMAN_TODO.md`: "confirm the measured angles in `fixtures/real/labels.json`" and
"take 3 ideal demo sketches (black marker, plain paper, upright, one ball on the ramp) and
more photos of any failure cases". CV re-tuning happens later, whenever new photos arrive.

**Phase 1 optionals (after P1 is green):**
- **P1.O1 Manual four corners** (never cut): if `paper_found=false` or the user taps "fix
  paper", let them tap 4 corners on the photo → `POST /corners`.
- **P1.O2 Label reading:** start from the seed's `enrich_with_gemini` prompt and schema
  (change 1 above: free OpenRouter vision model, no Gemini key). Crop thin-stroke regions near the ramp/ball, send the warped image
  to `LLM_VISION_MODEL` asking for **strict JSON** `{"angle_deg": number|null, "mass_kg":
  number|null}`; validate with pydantic; anything unparsable → null → defaults + UI notice.
  Show the read value as editable. Never block upload on it (run it async, 8 s timeout).
- **P1.O3 Ball sprite:** cut the ball's ink inside its circle to a transparent PNG; the canvas
  draws it at the ball's pose (rotation from `Frame.angle`); toggle to vector look.

### Phase 2 — Playable lab (the floor, never cut)

**P2.1 Changes ∥** — backend `POST /runs` with `params_override`; frontend buttons: angle
presets 15/30/45°, "×2 mass", gravity Earth/Moon/Mars (9.81/1.62/3.71). Every change makes a
new Run and records `changed`.

**P2.2 Drag the ramp ∥** — frontend: ramp's lower end fixed, handle on the upper end,
Pointer events (mouse + touch), angle 5–75°, live angle readout, optional snap to 5°. While
dragging, call `/predict` (throttled to ≥ 10/s, or compute in the browser with the same
formula ported to TS and cross-checked against `/predict` in a test) and update predicted time
and speed live. 150 ms after release → `POST /runs`, play it. Original angle stays as a faint
dashed line; "reset angle" button and a number box as fallback. If it won't move, show "the
ball will not slide at this angle" with the reason.

**P2.3 Side-by-side ∥** — pick any two runs; two canvases play in sync; table of time and
final speed with the ratio.

**P2.4 Why-card ∥** — backend `physics/whycard.py`: given two runs and the changed param, use
SymPy: if ∂t/∂param == 0 → "cancels" card; else show how t scales (e.g. t ∝ 1/√g with the
numeric ratio). Return LaTeX + highlighted symbol + one plain sentence. Frontend renders it.

**P2.O1 Friction button** — μ presets (0, 0.1, 0.3, 0.6); when μ ≥ tan θ the why-card shows
μ cos θ being subtracted from sin θ.

**Phase 2 done when:** "×2 mass" gives the same time every time; Moon/Earth ratio shows 2.46;
every supported change matches the formula within 3%; drag updates ≥ 10×/s on a laptop and a
phone (human check); a rerun appears < 3 s after release.

### Phase 3 — Personal TA

**P3.1 Session log & profile** — `store.py`: append `Event`s for upload/run/change/drag_rerun/
question/answer/quiz/export. Learner profile `{level, concepts_seen{mass_independent,
angle_effect, gravity_effect, friction_effect}, runs}` updated from `changed`.

**P3.2 LLM client** — section 3. **Done when:** live test (opt-in) returns text; with no key the
whole TA still works in template mode.

**P3.3 Context pack & prompt** — `ta/context.py` builds a compact JSON of: current scene, each
run's verified numbers labelled "Run N", changes, profile, formula table (P3.7). System
prompt: answer only from context, cite "Run N", LaTeX for formulas, say when something is out
of scope, adapt to level (beginner: analogies, no formulas; confident: derivation).

**P3.4 Number check** — `ta/numcheck.py`: extract every number in the answer (handle
"2.46×", "1.4 s", "30°", fractions like 5/7), accept if it matches any context number within
2% (or a small whitelist: 1, 2, 0.5, 9.81, …); return flags. UI shows a ⚠ next to flagged
numbers. No regeneration.

**P3.5 Summary** — "Summarize my session": template built from the log always exists (what was
drawn, each run, each change and its conclusion, formulas). LLM version optional, then number-
checked; on any failure return the template.

**P3.6 Follow-ups & levels** — 2 follow-up buttons from concepts not yet tried (fixed
templates, e.g. "Try the Moon — will it be faster or slower?"); clicking one performs the
change. Level chooser on entry; same question gives different explanations per level.

**P3.7 Formula table** — `physics/formula_table.py`: what each setup can report (time, speed,
acceleration, ratios) with LaTeX, fed to the context pack.

**P3.8 TA drawer UI** — side drawer: chat, level chooser, summarize, follow-ups, Markdown +
KaTeX, "based on Run 1, Run 2" chips, ⚠ flags.

**Phase 3 optionals:** **P3.O1 Quiz me** — TA (or template) proposes a what-if; user types a
guess; engine reruns; show "off by x%". All numbers from the engine. **P3.O2 Explain recognition
problems** — if `defaults_used` or warnings or low confidence: TA says what was read, what was
assumed, how to redraw (template first, LLM phrasing optional).

**Phase 3 done when:** on 20 scripted questions (`backend/tests/ta_questions.json`, run with the
live model, opt-in) no flagged invented numbers; summary lists every change in 10 scripted
sessions (template path, deterministic test); template mode passes everything with no key.

### Phase 4 — Take it home

**P4.1 GIF** — `export/gif.py` renders frames **server-side with Pillow** from the run's
trajectory (deterministic, testable): warped photo left, simulation right, one line
"predicted 1.42 s · simulated 1.41 s" below; 5–8 s loop; ≤ 5 MB; < 10 s to produce.
Fallback: still PNG with the numbers.

**P4.2 Share card** — project name + formula line on the card.

**P4.3 Download page & QR** — store the GIF under a random `share_id`; `/s/{share_id}` is a
tiny mobile page with the GIF and a Download button; QR (python `qrcode`) encodes
`PUBLIC_BASE_URL/s/{share_id}`. Fallback: plain "Download GIF" button.

**P4.4 Deployment** — `📝 HUMAN TODO` at the end: "scan the QR code on your phone 3 times".
`make tunnel` (cloudflared quick tunnel) sets a public URL; document
setting `PUBLIC_BASE_URL`. Optional later: Cloud Run container (`Dockerfile`). Same-Wi-Fi LAN
URL as a backup.

**P4.O1 Handwriting-style card** — render the question/answer in a handwriting-style font
(bundle an OFL font such as Caveat or Patrick Hand in `backend/app/export/fonts/`) on a paper
background inside the card. Call it "handwriting style", not "your handwriting".

**Phase 4 done when:** export < 10 s; GIF plays on a phone; scan → download works 3/3 on a real
phone over the tunnel (human check).

---

## 7. Human to-dos (done whenever the human is ready; agents never wait)

Agents collect these in `HUMAN_TODO.md`. None of them blocks building. CV re-tuning is a loop
you can run at any time: add photos → `make eval` → ask an agent to fix the failures.

| What you do (later) | What agents do with it |
|---|---|
| ~~Take 5 phone photos~~ ✅ done (`fixtures/real/IMG_9140–9144`) | Robustness test set (§5.1). |
| Confirm the measured angles in `fixtures/real/labels.json` | Accuracy numbers become trustworthy. |
| Take 3 ideal demo sketches, plus photos of any failure cases | Tune thresholds; aim for ≥ 70% then 85% on ~25 photos. If still poor: enforce "dark marker only" and rely on tap-to-confirm. |
| Try 2 live photos in `make dev` on your phone | Fix whatever breaks. |
| Drag the ramp on your phone | Fix jank or switch to number box + presets. |
| Ask the TA a few questions; check tone per level | Tune prompts. |
| Scan the QR 3 times on your phone via the tunnel | Fix URL/headers/size. |

Real-photo guidance to give anyone drawing: thick black marker, plain white paper, whole sheet in
frame, even light, no other drawings.

---

## 8. Parallel waves (for running several agents)

1. **Wave 1 (1 agent):** Step 0.
2. **Wave 2 (up to 5):** S1, P1.1, P1.3, P1.5, P1.7 (on fake data). P1.2 starts once P1.1's
   function signature exists.
3. **Wave 3 (2–3):** P1.4 (after S1), P1.2 tuning, P1.6 wiring → **Checkpoint A**.
4. **Wave 4 (up to 4):** P1.O1, P2.1, P2.2, P2.4, P1.O2; then P2.3, P2.O1, P1.O3.
5. **Wave 5 (up to 4):** P3.1 + P3.2 first, then P3.3–P3.8 split backend / frontend; P4.1 can
   start here too (only needs `Run`).
6. **Wave 6:** P4.2–P4.4, P3.O1, P3.O2, P4.O1.

Each agent works on its own branch/worktree, runs `make test` before merging, and merges to
`main` only when green.

## 9. Cut order (if something has to go, first to last)

1. P4.O1 handwriting-style card → 2. P3.O2 explain recognition problems → 3. P3.O1 quiz →
4. P2.O1 friction → 5. P1.O2 label reading → 6. P1.O3 sprite → 7. P4.3 QR (keep Download
button) → 8. P3.6 level choice (keep follow-ups). **Never cut:** P1.O1 four corners, Phase 1,
Phase 2 including drag.

## 10. Definition of done (whole project)

- 3 prepared real sketches + 1 live sketch run photo → simulation with no manual help.
- Predicted vs simulated within 3% for every run shown.
- Recognition ≥ 85% on 25 real photos (`make eval --set real`).
- Phase 2 demo moments work every time; drag works on phone and laptop.
- TA never shows an unflagged invented number; works with no key in template mode.
- GIF + QR download works 3/3 on a phone.
- `make setup && make test && make dev` works on a clean clone; `AGENTS.md` is accurate.
