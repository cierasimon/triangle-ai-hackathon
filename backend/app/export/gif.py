"""Shareable animation of a run, rendered server-side with Pillow (P4.1, P4.2).

Layout (960 × 450): the photo with recognized outlines on the left, the simulation on the
right, and a card strip below: "PaperPhysics · Run N", predicted vs simulated, the formula.
The motion plays in real time, holds on the final frame, and repeats to fill a ~6 s loop
(5–8 s). Frames share one palette, so the static photo costs almost nothing after frame 1.
Fallback: a still PNG of the final frame with the same card.
"""

import io
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.contracts import Run

W, PANEL_H, CARD_H = 960, 360, 90
H = PANEL_H + CARD_H
PANEL_W = W // 2
GIF_FPS = 20
LOOP_TARGET_S = 6.0
LOOP_MIN_S, LOOP_MAX_S = 5.0, 8.0
HOLD_S = 0.8  # pause on the final frame before repeating
MAX_BYTES = 5 * 1024 * 1024

BG = (250, 250, 247)
INK = (26, 26, 26)
MUTED = (107, 107, 107)
RAMP, GROUND, BALL = (220, 38, 38), (37, 99, 235), (22, 163, 74)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
UNICODE_SAMPLE = "θ√μ²→·°×"
ASCII = {"θ": "theta", "√": "sqrt", "μ": "mu", "²": "^2", "→": "->", "·": "-", "°": " deg",
         "×": "x", "−": "-"}


@dataclass
class Rendered:
    data: bytes
    media_type: str  # "image/gif", or "image/png" for the still fallback
    seconds: float  # loop length (0 for a still)
    render_s: float


def _glyph(font: ImageFont.FreeTypeFont, ch: str) -> bytes:
    img = Image.new("L", (64, 64))
    ImageDraw.Draw(img).text((4, 4), ch, font=font, fill=255)
    return img.tobytes()


def _has_glyphs(font: ImageFont.FreeTypeFont, text: str) -> bool:
    """A font without a glyph draws the same 'missing' box for every such character."""
    missing = _glyph(font, "\uffff")
    return all(_glyph(font, ch) != missing for ch in text)


@lru_cache(maxsize=8)
def _font(size: int) -> tuple[ImageFont.FreeTypeFont, bool]:
    """(font, supports the Greek/math symbols we print)."""
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                font = ImageFont.truetype(path, size)
            except OSError:
                continue
            if _has_glyphs(font, UNICODE_SAMPLE):
                return font, True
    font = ImageFont.load_default(size=size)
    return font, _has_glyphs(font, UNICODE_SAMPLE)


def _text(draw: ImageDraw.ImageDraw, xy, text: str, size: int, fill=INK) -> None:
    font, unicode_ok = _font(size)
    if not unicode_ok:
        for k, v in ASCII.items():
            text = text.replace(k, v)
    draw.text(xy, text, font=font, fill=fill)


# ---- card (P4.2) ------------------------------------------------------------------------

FORMULA_TEXT = {
    "slide": "t = √(2L / a),  a = g(sin θ − μ cos θ)",
    "roll": "t = √(2L / a),  a = 5/7 · g sin θ",
}


def card_lines(run: Run) -> tuple[str, str, str]:
    v, p = run.verification, run.scene.params
    head = f"PaperPhysics · Run {run.index}"
    if v.predicted_time_s is None:
        numbers = "the ball does not move: μ ≥ tan θ"
    elif v.simulated_time_s is None:
        numbers = f"predicted {v.predicted_time_s:.2f} s · simulation stopped at 10 s"
    else:
        numbers = (f"predicted {v.predicted_time_s:.2f} s · simulated {v.simulated_time_s:.2f} s"
                   f" · error {v.error_pct:.1f}%")
    setup = (f"{FORMULA_TEXT[p.motion]}   ·   θ = {p.angle_deg:.0f}°, g = {p.gravity:g} m/s², "
             f"μ = {p.friction:g}, m = {p.mass_kg:g} kg")
    return head, numbers, setup


def _base(run: Run, photo: Image.Image | None) -> Image.Image:
    """Everything that doesn't move: photo panel, card strip, divider."""
    img = Image.new("RGB", (W, H), BG)
    if photo is not None:
        p = photo.convert("RGB")
        p.thumbnail((PANEL_W - 16, PANEL_H - 16), Image.LANCZOS)
        img.paste(p, ((PANEL_W - p.width) // 2, (PANEL_H - p.height) // 2))
    draw = ImageDraw.Draw(img)
    draw.line([(PANEL_W, 12), (PANEL_W, PANEL_H - 12)], fill=(227, 227, 221), width=1)
    draw.rectangle([0, PANEL_H, W, H], fill=(255, 255, 255))
    head, numbers, setup = card_lines(run)
    _text(draw, (16, PANEL_H + 10), head, 20)
    _text(draw, (16, PANEL_H + 38), numbers, 17, fill=INK if run.verification.ok else RAMP)
    _text(draw, (16, PANEL_H + 63), setup, 14, fill=MUTED)
    return img


# ---- simulation panel -------------------------------------------------------------------


class _View:
    """Meters (y up) → panel pixels, fixed for the whole animation."""

    def __init__(self, run: Run):
        s = run.scene
        (tx, ty), (bx, by) = s.ramp_top_m, s.ramp_bottom_m
        pad = 0.12 * math.dist((tx, ty), (bx, by)) + 2 * s.ball_radius_m
        self.min_x, self.max_x = min(tx, bx) - pad, max(tx, bx) + pad
        self.min_y, self.max_y = min(s.ground_y_m, by) - pad, max(ty, by) + pad
        self.scale = min((PANEL_W - 24) / (self.max_x - self.min_x),
                         (PANEL_H - 24) / (self.max_y - self.min_y))
        self.ox = PANEL_W + (PANEL_W - (self.max_x - self.min_x) * self.scale) / 2
        self.oy = (PANEL_H - (self.max_y - self.min_y) * self.scale) / 2

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        return (self.ox + (x - self.min_x) * self.scale,
                PANEL_H - self.oy - (y - self.min_y) * self.scale)


def _draw_scene(img: Image.Image, run: Run, view: _View) -> None:
    draw = ImageDraw.Draw(img)
    s = run.scene
    draw.line([view(view.min_x, s.ground_y_m), view(view.max_x, s.ground_y_m)], fill=GROUND, width=4)
    draw.line([view(*s.ramp_bottom_m), view(*s.ramp_top_m)], fill=RAMP, width=4)


def _draw_ball(img: Image.Image, run: Run, view: _View, x: float, y: float, spin: float) -> None:
    draw = ImageDraw.Draw(img)
    cx, cy = view(x, y)
    r = max(3.0, run.scene.ball_radius_m * view.scale)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=BALL)
    draw.line([(cx, cy), (cx + r * math.cos(-spin), cy + r * math.sin(-spin))],
              fill=(255, 255, 255), width=2)


def _frame_at(run: Run, t: float):
    frames = run.frames
    for f in frames:
        if f.t >= t:
            return f
    return frames[-1]


def render(run: Run, photo: Image.Image | None) -> Rendered:
    """GIF of the run; a still PNG if the GIF fails or is too big."""
    start = time.perf_counter()
    base = _base(run, photo)
    view = _View(run)
    _draw_scene(base, run, view)
    try:
        data, seconds = _gif(run, base, view)
        if len(data) <= MAX_BYTES:
            return Rendered(data, "image/gif", seconds, time.perf_counter() - start)
    except Exception:  # never lose the export: fall through to the still
        pass
    return Rendered(still_png(run, base, view), "image/png", 0.0, time.perf_counter() - start)


def still_png(run: Run, base: Image.Image, view: _View) -> bytes:
    img = base.copy()
    last = run.frames[-1]
    _draw_ball(img, run, view, last.x, last.y, last.angle)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _gif(run: Run, base: Image.Image, view: _View) -> tuple[bytes, float]:
    motion = run.frames[-1].t if run.frames else 0.0
    if motion <= 0 or run.verification.predicted_time_s is None:
        # the ball doesn't move: one still frame shown for the whole loop
        img = base.copy()
        f = run.frames[0]
        _draw_ball(img, run, view, f.x, f.y, f.angle)
        return _encode([img], [int(LOOP_TARGET_S * 1000)]), LOOP_TARGET_S

    # play faster than real time only if one pass wouldn't fit in the loop
    speed = max(1.0, (motion + HOLD_S) / (LOOP_MAX_S - 0.5))
    pass_s = motion / speed + HOLD_S
    repeats = max(1, round(LOOP_TARGET_S / pass_s))
    while repeats * pass_s < LOOP_MIN_S:
        repeats += 1
    while repeats > 1 and repeats * pass_s > LOOP_MAX_S:
        repeats -= 1

    n = max(2, math.ceil(motion / speed * GIF_FPS))
    images, durations = [], []
    for i in range(n + 1):
        f = _frame_at(run, min(i / GIF_FPS * speed, motion))
        img = base.copy()
        _draw_ball(img, run, view, f.x, f.y, f.angle)
        images.append(img)
        durations.append(int(1000 / GIF_FPS))
    durations[-1] = int(HOLD_S * 1000)
    images, durations = images * repeats, durations * repeats
    return _encode(images, durations), sum(durations) / 1000


def _encode(images: list[Image.Image], durations: list[int]) -> bytes:
    palette = images[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False, disposal=1)
    return buf.getvalue()
