"""Shareable animation of a run, rendered server-side with Pillow (P4.1, P4.2).

Layout (960 × 480), in the app's dark theme: the simulation as the web player draws it
(ground up to its stop, ramp, θ, balls and blocks with their weight and velocity arrows) and
a card strip below: "PaperPhysics · latest run", the time and speed at the bottom, the setup and
any collision. The motion plays at the web player's slow-motion speed, after the same short pause, and is
cut at MAX_GIF_S rather than sped up; it holds on the final frame, then loops.
Fallback: a still PNG of the final frame with the same card.
"""

import io
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.contracts import Frame, Run, Shape
from app.physics.simulate import ground_end

W, PANEL_H, CARD_H = 960, 390, 90
H = PANEL_H + CARD_H
GIF_FPS = 20
PLAYBACK_SPEED = 0.12  # physics s per real s: the web player's slow motion (usePlayback.ts)
START_HOLD_S = 0.7  # pause before the motion starts, as in the web player
HOLD_S = 0.8  # pause on the final frame before repeating
MAX_GIF_S = 15.0  # longer runs are cut here (not sped up)
STILL_S = 6.0  # a run where nothing moves: one frame for this long
MAX_BYTES = 5 * 1024 * 1024

MIN_BODY_PX = 6

# the web app's dark, colorless theme (frontend/src/styles.css)
BG = (11, 11, 12)
CARD = (20, 20, 21)
BORDER = (42, 42, 45)
INK = (236, 236, 236)
MUTED = (140, 140, 140)
MAIN_FILL, OTHER_FILL = (189, 189, 189), (111, 111, 111)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
UNICODE_SAMPLE = "θμ→·°₁₂"
ASCII = {"θ": "theta", "μ": "mu", "→": "->", "·": "-", "°": " deg", "₁": "1", "₂": "2"}


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


def card_lines(run: Run) -> tuple[str, str, str]:
    v, s = run.verification, run.scene
    p = s.params
    head = "PaperPhysics · latest run"
    if v.predicted_time_s is None:
        numbers = "it does not move: μ ≥ tan θ"
    else:
        t = v.simulated_time_s if v.simulated_time_s is not None else v.predicted_time_s
        speed = v.simulated_speed_m_s if v.simulated_speed_m_s is not None else v.predicted_speed_m_s
        numbers = f"time to bottom {t:.2f} s · speed at bottom {speed:.2f} m/s"
    length = math.dist(s.ramp_top_m, s.ramp_bottom_m)
    masses = (f"m₁ = {p.mass_kg:g} kg, m₂ = {s.others[0].mass_kg:g} kg" if s.others
              else f"m = {p.mass_kg:g} kg")
    setup = f"θ = {p.angle_deg:.0f}°, L = {length:.2f} m, {masses}"
    c = run.collision
    if c:
        setup += (f"   ·   hit at {c.t:.2f} s: 1: {c.v1_before:.2f} → {c.v1_after:.2f} m/s,"
                  f" 2: {c.v2_before:.2f} → {c.v2_after:.2f} m/s")
    return head, numbers, setup


def _base(run: Run) -> Image.Image:
    """Everything that doesn't change: background and the card strip."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, PANEL_H, W, H], fill=CARD)
    draw.line([(0, PANEL_H), (W, PANEL_H)], fill=BORDER, width=1)
    head, numbers, setup = card_lines(run)
    _text(draw, (20, PANEL_H + 10), head, 20)
    _text(draw, (20, PANEL_H + 38), numbers, 17)
    _text(draw, (20, PANEL_H + 63), setup, 14, fill=MUTED)
    return img


# ---- simulation panel -------------------------------------------------------------------


class _View:
    """Meters (y up) → panel pixels, fixed for the whole animation."""

    def __init__(self, run: Run):
        s = run.scene
        (tx, ty), (bx, by) = s.ramp_top_m, s.ramp_bottom_m
        biggest = max([s.ball_radius_m, *(o.radius_m for o in s.others)])
        pad = 0.08 * math.dist((tx, ty), (bx, by)) + 2 * biggest
        xs = [tx, bx, *(o.center_m[0] for o in s.others), ground_end(s)]  # every body stays in view
        self.min_x, self.max_x = min(xs) - pad, max(xs) + pad
        self.min_y, self.max_y = min(s.ground_y_m, by) - pad, max(ty, by) + pad
        self.scale = min((W - 40) / (self.max_x - self.min_x),
                         (PANEL_H - 40) / (self.max_y - self.min_y))
        self.ox = (W - (self.max_x - self.min_x) * self.scale) / 2
        self.oy = (PANEL_H - (self.max_y - self.min_y) * self.scale) / 2

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        return (self.ox + (x - self.min_x) * self.scale,
                PANEL_H - self.oy - (y - self.min_y) * self.scale)


def _arrow(draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float) -> None:
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 3:
        return
    a = math.atan2(y1 - y0, x1 - x0)
    head = min(10.0, 0.4 * length)
    draw.line([(x0, y0), (x1, y1)], fill=INK, width=2)
    draw.polygon([(x1, y1), (x1 - head * math.cos(a - 0.45), y1 - head * math.sin(a - 0.45)),
                  (x1 - head * math.cos(a + 0.45), y1 - head * math.sin(a + 0.45))], fill=INK)


def _draw_scene(img: Image.Image, run: Run, view: _View) -> None:
    draw = ImageDraw.Draw(img)
    s = run.scene
    (bx, by), (tx, ty) = s.ramp_bottom_m, s.ramp_top_m
    end = ground_end(s)
    far = view.max_x if tx > bx else view.min_x  # the ground behind the ramp runs off the panel
    gy = view(0, s.ground_y_m)[1]
    draw.line([(view(far, 0)[0], gy), (view(end, 0)[0], gy)], fill=MUTED, width=2)
    wall = 22.0
    draw.line([(view(end, 0)[0], gy), (view(end, 0)[0], gy - wall)], fill=INK, width=4)
    draw.line([view(bx, by), view(tx, ty)], fill=INK, width=3)
    # θ arc at the foot of the ramp, label under the ground
    cx, cy = view(bx, by)
    r = min(48.0, 0.22 * math.dist((tx, ty), (bx, by)) * view.scale)
    a = s.params.angle_deg
    start, stop = (360 - a, 360) if tx > bx else (180, 180 + a)
    draw.arc([cx - r, cy - r, cx + r, cy + r], start, stop, fill=MUTED, width=2)
    side = 1 if tx > bx else -1
    label = f"θ = {a:.0f}°"
    font, _ = _font(14)
    width = draw.textlength(label, font=font)
    _text(draw, (cx + side * (r + 6) - (width if side < 0 else 0), cy + 6), label, 14)


def _pos(f: Frame, which: int) -> tuple[float, float]:
    return (f.x, f.y) if which < 0 else tuple(f.others[which][:2])


def _frame_at(frames: list[Frame], t: float) -> tuple[Frame, float, int]:
    """(frame interpolated at t, blend factor k, index j of the frame after t)."""
    j = next((i for i, f in enumerate(frames) if f.t >= t), len(frames) - 1)
    if j == 0:
        return frames[0], 0.0, 1
    a, b = frames[j - 1], frames[j]
    k = min(1.0, max(0.0, (t - a.t) / ((b.t - a.t) or 1)))
    mix = lambda p, q: p + (q - p) * k  # noqa: E731
    others = [tuple(mix(p, q) for p, q in zip(oa, ob)) for oa, ob in zip(a.others, b.others)]
    return Frame(t=t, x=mix(a.x, b.x), y=mix(a.y, b.y), angle=mix(a.angle, b.angle),
                 others=others), k, j


def _velocity_at_frame(frames: list[Frame], i: int, which: int) -> tuple[float, float]:
    a, b = frames[max(0, i - 1)], frames[min(len(frames) - 1, i + 1)]
    dt = b.t - a.t
    if dt <= 0:
        return 0.0, 0.0
    (xa, ya), (xb, yb) = _pos(a, which), _pos(b, which)
    return (xb - xa) / dt, (yb - ya) / dt


def _velocity(frames: list[Frame], j: int, k: float, which: int) -> tuple[float, float]:
    """Velocity (m/s) between frames j-1 and j, blended like the position (smooth arrows)."""
    va, vb = _velocity_at_frame(frames, j - 1, which), _velocity_at_frame(frames, j, which)
    return va[0] + (vb[0] - va[0]) * k, va[1] + (vb[1] - va[1]) * k


def _draw_body(draw: ImageDraw.ImageDraw, view: _View, x: float, y: float, spin: float,
               radius_m: float, shape: Shape, fill) -> tuple[float, float, float]:
    """Draw one ball or block; returns its centre and size in pixels."""
    real = radius_m * view.scale
    r = max(real, MIN_BODY_PX)
    cx, cy = view(x, y)
    cy -= r - real  # drawn bigger than real: lift it so it still sits on the surface
    if shape == "block":
        corners = [(-r, -r), (r, -r), (r, r), (-r, r)]
        c, s = math.cos(-spin), math.sin(-spin)
        draw.polygon([(cx + u * c - v * s, cy + u * s + v * c) for u, v in corners], fill=fill)
    else:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    return cx, cy, r


def _draw_bodies(img: Image.Image, run: Run, view: _View, t: float) -> None:
    """Every body as it is at time t, with its weight and (while moving) velocity arrows."""
    draw = ImageDraw.Draw(img)
    s = run.scene
    f, k, j = _frame_at(run.frames, t)
    unit = math.dist(s.ramp_top_m, s.ramp_bottom_m) * view.scale
    c = run.collision
    v_max = max(run.verification.simulated_speed_m_s or run.verification.predicted_speed_m_s or 0,
                c.v1_before if c else 0, c.v2_after if c else 0)
    moving = 0 < t < run.frames[-1].t
    bodies = [(f.x, f.y, f.angle, s.ball_radius_m, s.ball_shape, s.params.mass_kg, MAIN_FILL, -1)]
    bodies += [(x, y, a, o.radius_m, o.shape, o.mass_kg, OTHER_FILL, k)
               for k, ((x, y, a), o) in enumerate(zip(f.others, s.others))]
    for x, y, spin, radius, shape, mass, fill, which in bodies:
        cx, cy, _ = _draw_body(draw, view, x, y, spin, radius, shape, fill)
        w_len = min(unit * (0.08 + 0.017 * mass), PANEL_H - cy - 12)
        _arrow(draw, cx, cy, cx, cy + w_len)
        vx, vy = _velocity(run.frames, j, k, which) if moving else (0.0, 0.0)
        speed = math.hypot(vx, vy)
        if speed > 1e-3 and v_max:
            length = unit * 0.4 * min(1.0, speed / v_max)
            _arrow(draw, cx, cy, cx + vx / speed * length, cy - vy / speed * length)


def render(run: Run, photo: Image.Image | None = None) -> Rendered:
    """GIF of the run; a still PNG if the GIF fails or is too big. (`photo` is unused: the
    export shows the simulation only, as the app does.)"""
    start = time.perf_counter()
    base = _base(run)
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
    _draw_bodies(img, run, view, run.frames[-1].t)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _gif(run: Run, base: Image.Image, view: _View) -> tuple[bytes, float]:
    motion = run.frames[-1].t if run.frames else 0.0
    if motion <= 0 or run.verification.predicted_time_s is None:
        # nothing moves: one still frame
        img = base.copy()
        _draw_bodies(img, run, view, 0.0)
        return _encode([img], [int(STILL_S * 1000)]), STILL_S

    # the app's slow motion, after its pause; a long run is cut at MAX_GIF_S, not sped up
    shown = min(motion, (MAX_GIF_S - START_HOLD_S - HOLD_S) * PLAYBACK_SPEED)
    n = max(2, math.ceil(shown / PLAYBACK_SPEED * GIF_FPS))
    images, durations = [], []
    for i in range(n + 1):
        img = base.copy()
        _draw_bodies(img, run, view, min(i / GIF_FPS * PLAYBACK_SPEED, shown))
        images.append(img)
        durations.append(int(1000 / GIF_FPS))
    durations[0] += int(START_HOLD_S * 1000)
    durations[-1] = int(HOLD_S * 1000)
    return _encode(images, durations), sum(durations) / 1000


def _encode(images: list[Image.Image], durations: list[int]) -> bytes:
    palette = images[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False, disposal=1)
    return buf.getvalue()
