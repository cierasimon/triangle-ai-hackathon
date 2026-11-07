"""Ink mask → ramp, ground and ball (P1.2).

1. Straight strokes: probabilistic Hough on the ink mask, then merge collinear pieces into
   long lines (stroke-width based tolerances, so thick marker and thin ballpoint both work).
2. Ball candidates: closed, round contours (outer outlines and holes) of a plausible size.
3. Orientation: drawings are often sideways (portrait phone, landscape notebook). If the page
   has ruled lines, "horizontal" is the rule direction (people draw the ground along the
   rules); otherwise the photo is assumed upright unless only a sideways reading works. Each
   of the 0/90/180/270° rotations is scored by its best ground + ramp pair: a near-horizontal
   ground, a slanted ramp whose low end meets it, and the ramp's top above the ground.
4. The chosen rotation (plus a small deskew so the ground is exactly horizontal) is applied
   to the image, so Recognition coordinates are in the upright, flattened image.
5. Ball: the candidate closest to the ramp top among those resting on (or floating just
   above) the ramp; blocks, arcs, arrows and handwriting are ignored.
"""

import io
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.contracts import Ball, Line, Recognition
from app.cv.capture import Capture, darkness

GROUND_MAX_TILT = 15.0  # degrees from horizontal
RAMP_MIN_ANGLE, RAMP_MAX_ANGLE = 4.0, 84.0
MIN_RAMP_LEN = 0.15  # of the short side
MIN_GROUND_LEN = 0.08
MEET_TOL = 0.06  # ramp low end to ground line, of the short side
BALL_MIN_R, BALL_MAX_R = 0.012, 0.12  # of the short side
BALL_MIN_CIRCULARITY = 0.70
BALL_MIN_ROUNDNESS = 0.55  # minor/major axis of the fitted ellipse


@dataclass
class Seg:
    """A merged straight stroke."""

    p1: np.ndarray
    p2: np.ndarray
    coverage: float = 1.0

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p2 - self.p1))

    @property
    def angle(self) -> float:
        """Direction in degrees, 0..180 (image coords, y down)."""
        d = self.p2 - self.p1
        return math.degrees(math.atan2(d[1], d[0])) % 180

    def tilt(self) -> float:
        """Angle from horizontal, 0..90."""
        a = self.angle
        return min(a, 180 - a)


@dataclass
class Circle:
    center: np.ndarray
    radius: float
    score: float  # roundness quality 0..1


@dataclass
class Result:
    recognition: Recognition
    image: np.ndarray  # upright, flattened BGR image the coordinates refer to
    mask: np.ndarray
    rotation_k: int  # np.rot90 k applied (counter-clockwise quarter turns)
    rules_angle: float | None
    debug: dict = field(default_factory=dict)


# ---- geometry helpers -------------------------------------------------------------------


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _unit(angle_deg: float) -> np.ndarray:
    t = math.radians(angle_deg)
    return np.array([math.cos(t), math.sin(t)])


def _point_line_dist(p, a, b) -> float:
    d = b - a
    n = np.linalg.norm(d)
    if n == 0:
        return float(np.linalg.norm(p - a))
    return float(abs(d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) / n)


def stroke_width(mask: np.ndarray) -> float:
    ink = mask > 0
    if not ink.any():
        return 3.0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    # thickest point across each stroke ~ the skeleton: take a high percentile of the
    # distance values (most ink pixels are near an edge)
    return max(2.0, 2.0 * float(np.percentile(dist[ink], 90)))


# ---- 1. straight strokes ----------------------------------------------------------------


def detect_segments(mask: np.ndarray, stroke: float) -> list[Seg]:
    short = min(mask.shape[:2])
    raw = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=max(30, int(0.04 * short)),
                          minLineLength=int(0.04 * short), maxLineGap=int(max(6, 1.5 * stroke)))
    if raw is None:
        return []
    pieces = [Seg(np.array(r[:2], float), np.array(r[2:], float)) for r in raw.reshape(-1, 4)]
    return [trace_extend(s, mask, stroke) for s in merge_segments(pieces, mask, stroke)]


def merge_segments(pieces: list[Seg], mask: np.ndarray, stroke: float) -> list[Seg]:
    short = min(mask.shape[:2])
    perp_tol = max(5.0, 2.0 * stroke)
    gap_tol = 0.04 * short
    groups: list[list[Seg]] = []
    fits: list[Seg] = []
    for piece in sorted(pieces, key=lambda s: -s.length):
        for i, line in enumerate(fits):
            if _angle_diff(piece.angle, line.angle) > 5:
                continue
            if max(_point_line_dist(piece.p1, line.p1, line.p2),
                   _point_line_dist(piece.p2, line.p1, line.p2)) > perp_tol:
                continue
            u = _unit(line.angle)
            lo, hi = sorted((float(line.p1 @ u), float(line.p2 @ u)))
            plo, phi = sorted((float(piece.p1 @ u), float(piece.p2 @ u)))
            if plo > hi + gap_tol or phi < lo - gap_tol:
                continue
            groups[i].append(piece)
            fits[i] = _fit(groups[i])
            break
        else:
            groups.append([piece])
            fits.append(piece)
    for line in fits:
        line.coverage = _coverage(line, mask, perp_tol / 2)
    return [line for line in fits if line.coverage >= 0.6]


def trace_extend(seg: Seg, mask: np.ndarray, stroke: float) -> Seg:
    """Follow the ink past both ends of a line, so slightly curved hand strokes that Hough
    split into pieces are recovered to their real ends. Re-centres on the ink at each step
    and stops at a gap or where the stroke turns away (e.g. the corner with the wall)."""
    h, w = mask.shape[:2]
    band = max(4.0, 1.5 * stroke)
    max_gap = max(8.0, 3 * stroke)
    step = 2.0

    def walk(start: np.ndarray, u: np.ndarray) -> np.ndarray:
        n = np.array([-u[1], u[0]])
        p, last_ink, gap = start.copy(), start.copy(), 0.0
        offsets = np.arange(-band, band + 1)
        while gap <= max_gap:
            p = p + step * u
            pts = (p[None, :] + offsets[:, None] * n[None, :]).astype(int)
            ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
            hits = offsets[ok][mask[pts[ok, 1], pts[ok, 0]] > 0] if ok.any() else []
            if len(hits):
                # re-centre on the stroke, but only a little per step: enough to follow a
                # gentle curve, not enough to turn a corner onto another line
                p = p + float(np.clip(np.mean(hits), -0.5, 0.5)) * n
                last_ink, gap = p.copy(), 0.0
            else:
                gap += step
        return last_ink

    u = (seg.p2 - seg.p1) / max(seg.length, 1e-6)
    return Seg(walk(seg.p1, -u), walk(seg.p2, u), seg.coverage)


def refit(seg: Seg, mask: np.ndarray, stroke: float) -> Seg:
    """Least-squares line through the ink along the middle 80% of a segment (the ends are
    where other strokes join), spanning the same extent."""
    band = max(4.0, 1.5 * stroke)
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    d = seg.p2 - seg.p1
    length = max(seg.length, 1e-6)
    u = d / length
    rel = pts - seg.p1
    t = rel @ u / length
    off = np.abs(rel @ np.array([-u[1], u[0]]))
    sel = pts[(t > 0.1) & (t < 0.9) & (off <= band)]
    if len(sel) < 10:
        return seg
    vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    v, c = np.array([vx, vy]), np.array([x0, y0])
    return Seg(c + ((seg.p1 - c) @ v) * v, c + ((seg.p2 - c) @ v) * v, seg.coverage)


def intersect(a: Seg, b: Seg) -> np.ndarray | None:
    d1, d2 = a.p2 - a.p1, b.p2 - b.p1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((b.p1[0] - a.p1[0]) * d2[1] - (b.p1[1] - a.p1[1]) * d2[0]) / den
    return a.p1 + t * d1


def _fit(group: list[Seg]) -> Seg:
    """Length-weighted least-squares line through the pieces' endpoints, full extent."""
    pts = np.array([p for s in group for p in (s.p1, s.p2)])
    w = np.repeat([s.length for s in group], 2)
    c = (pts * w[:, None]).sum(0) / w.sum()
    cov = ((pts - c).T * w) @ (pts - c)
    u = np.linalg.eigh(cov)[1][:, -1]
    proj = (pts - c) @ u
    return Seg(c + proj.min() * u, c + proj.max() * u)


def _coverage(line: Seg, mask: np.ndarray, radius: float) -> float:
    n = max(10, int(line.length / 4))
    h, w = mask.shape[:2]
    r = max(1, int(round(radius)))
    hits = 0
    for t in np.linspace(0, 1, n):
        x, y = (line.p1 + t * (line.p2 - line.p1)).astype(int)
        if mask[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)].any():
            hits += 1
    return hits / n


# ---- 2. ball candidates -----------------------------------------------------------------


def detect_circles(mask: np.ndarray, stroke: float) -> list[Circle]:
    short = min(mask.shape[:2])
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    found: list[Circle] = []
    for contour in contours:
        if len(contour) < 12:
            continue
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0 or area == 0:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(contour)
        if not BALL_MIN_R * short <= r <= BALL_MAX_R * short:
            continue
        circularity = 4 * math.pi * area / perimeter**2
        fill = area / (math.pi * r * r)
        (_, _), axes, _ = cv2.fitEllipse(contour)
        roundness = min(axes) / max(axes) if max(axes) > 0 else 0
        if circularity < BALL_MIN_CIRCULARITY or roundness < BALL_MIN_ROUNDNESS or fill < 0.5:
            continue
        # a drawn square (a block) is round enough by circularity (π/4) but simplifies to
        # four corners and fills only 2/π of its enclosing circle; a hand circle fills more
        corners = len(cv2.approxPolyDP(contour, 0.04 * perimeter, True))
        if corners <= 4 or (corners == 5 and fill < 0.7):
            continue
        found.append(Circle(np.array([cx, cy]), r, (circularity + roundness) / 2))
    # a drawn circle gives an outer outline and an inner hole: keep one, mid-stroke radius
    merged: list[Circle] = []
    for c in sorted(found, key=lambda c: -c.radius):
        twin = next((m for m in merged if np.linalg.norm(m.center - c.center) < 0.5 * m.radius),
                    None)
        if twin is None:
            merged.append(c)
        elif twin.radius - c.radius < 3 * stroke:
            twin.radius = (twin.radius + c.radius) / 2
            twin.score = max(twin.score, c.score)
    return merged


# ---- 3. orientation ---------------------------------------------------------------------


def detect_rules(image: np.ndarray, mask: np.ndarray) -> float | None:
    """Direction (deg, 0..180) of faint ruled notebook lines, or None."""
    dark = darkness(image)
    ink = cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0
    faint = ((dark > 15) & ~ink).astype(np.uint8) * 255
    small = cv2.resize(faint, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    long_side = max(small.shape)
    raw = cv2.HoughLinesP(small, 1, np.pi / 360, threshold=80, minLineLength=int(0.35 * long_side),
                          maxLineGap=12)
    if raw is None or len(raw) < 3:
        return None
    segs = [Seg(np.array(r[:2], float), np.array(r[2:], float)) for r in raw.reshape(-1, 4)]
    best, best_len = None, 0.0
    for s in segs:
        cluster = [o for o in segs if _angle_diff(o.angle, s.angle) < 4]
        total = sum(o.length for o in cluster)
        if len(cluster) >= 3 and total > best_len:
            best, best_len = cluster, total
    if best is None or best_len < 0.6 * sum(s.length for s in segs):
        return None
    ref = best[0].angle
    # average the angles around ref (handles the 0/180 wrap)
    diffs = [((o.angle - ref + 90) % 180) - 90 for o in best]
    return (ref + float(np.average(diffs, weights=[o.length for o in best]))) % 180


def rotate_point(p: np.ndarray, k: int, size: tuple[int, int]) -> np.ndarray:
    """Where point p lands after np.rot90(img, k) (CCW quarter turns); size = (w, h)."""
    x, y = float(p[0]), float(p[1])
    w, h = size
    for _ in range(k % 4):
        x, y = y, w - 1 - x
        w, h = h, w
    return np.array([x, y])


def _rotated_size(size: tuple[int, int], k: int) -> tuple[int, int]:
    return size if k % 2 == 0 else (size[1], size[0])


@dataclass
class Pair:
    k: int
    ramp: Seg
    ground: Seg
    score: float


def best_pair(segs: list[Seg], circles: list[Circle], size: tuple[int, int], k: int,
              short: float) -> Pair | None:
    rs = [Seg(rotate_point(s.p1, k, size), rotate_point(s.p2, k, size), s.coverage)
          for s in segs]
    rc = [Circle(rotate_point(c.center, k, size), c.radius, c.score) for c in circles]
    grounds = [s for s in rs if s.tilt() <= GROUND_MAX_TILT and s.length >= MIN_GROUND_LEN * short]
    ramps = [s for s in rs if RAMP_MIN_ANGLE <= s.tilt() <= RAMP_MAX_ANGLE
             and s.length >= MIN_RAMP_LEN * short]
    best = None
    for ramp in ramps:
        low, high = (ramp.p1, ramp.p2) if ramp.p1[1] > ramp.p2[1] else (ramp.p2, ramp.p1)
        for ground in grounds:
            if _point_line_dist(low, ground.p1, ground.p2) > MEET_TOL * short:
                continue
            u = ground.p2 - ground.p1
            t = float((low - ground.p1) @ u / (u @ u))
            if not -MEET_TOL * short / ground.length <= t <= 1 + MEET_TOL * short / ground.length:
                continue
            # ground y under the ramp's top; the top must be clearly above it
            gy = ground.p1[1] + (high[0] - ground.p1[0]) * (u[1] / u[0] if u[0] else 0)
            if gy - high[1] < 0.05 * short:
                continue
            score = ramp.length + 0.5 * ground.length
            if any(_on_ramp(c, low, high) for c in rc):
                score += 0.3 * short
            if best is None or score > best.score:
                best = Pair(k, ramp, ground, score)
    return best


def _on_ramp(c: Circle, low: np.ndarray, high: np.ndarray) -> bool:
    t, d = _ramp_coords(c.center, low, high)
    return -0.15 <= t <= 1.1 and 0 < d < 6 * c.radius


def _ramp_coords(p: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    """(t, d): t = 0 at the ramp top, 1 at the bottom; d = height above the ramp (px)."""
    u = low - high
    t = float((p - high) @ u / (u @ u))
    foot = high + t * u
    d = float(np.linalg.norm(p - foot))
    return t, d if p[1] < foot[1] else -d  # above = smaller y in the upright image


# ---- main entry -------------------------------------------------------------------------


def recognize(cap: Capture, image_id: str) -> Result:
    image, mask = cap.image, cap.mask
    h, w = mask.shape[:2]
    short = float(min(w, h))
    warnings = list(cap.warnings)
    stroke = stroke_width(mask)
    segs = detect_segments(mask, stroke)
    circles = detect_circles(mask, stroke)
    rules = detect_rules(image, mask)

    if rules is not None:
        # rotations that make the rules horizontal
        ks = [0, 2] if min(rules, 180 - rules) < 45 else [1, 3]
    else:
        ks = [0, 1, 2, 3]
    pairs = []
    for k in ks:
        pair = best_pair(segs, circles, (w, h), k, short)
        if pair:
            if rules is None and k == 0:
                pair.score += 0.5 * short  # photos without rules are usually upright
            pairs.append(pair)
    pair = max(pairs, key=lambda p: p.score) if pairs else None
    k = pair.k if pair else 0

    up_image = np.ascontiguousarray(np.rot90(image, k))
    up_mask = np.ascontiguousarray(np.rot90(mask, k))
    rot = lambda p: rotate_point(p, k, (w, h))  # noqa: E731
    up_w, up_h = _rotated_size((w, h), k)
    up_circles = [Circle(rot(c.center), c.radius, c.score) for c in circles]

    lines: list[Line] = []
    ball = None
    if pair is None:
        warnings.append("no ramp found — draw one straight slanted line meeting a flat ground line")
    else:
        ramp, ground = refit(pair.ramp, up_mask, stroke), refit(pair.ground, up_mask, stroke)
        # deskew: rotate about the image centre so the drawn ground is exactly horizontal
        gd = ground.p2 - ground.p1
        tilt = math.degrees(math.atan2(gd[1], gd[0]))
        tilt = (tilt + 90) % 180 - 90
        if abs(tilt) > 0.3:
            matrix = cv2.getRotationMatrix2D((up_w / 2, up_h / 2), tilt, 1.0)
            up_image = cv2.warpAffine(up_image, matrix, (up_w, up_h), borderMode=cv2.BORDER_REPLICATE)
            up_mask = cv2.warpAffine(up_mask, matrix, (up_w, up_h), flags=cv2.INTER_NEAREST)
            fix = lambda p: matrix[:, :2] @ p + matrix[:, 2]  # noqa: E731
            ramp = Seg(fix(ramp.p1), fix(ramp.p2), ramp.coverage)
            ground = Seg(fix(ground.p1), fix(ground.p2), ground.coverage)
            up_circles = [Circle(fix(c.center), c.radius, c.score) for c in up_circles]

        low, high = (ramp.p1, ramp.p2) if ramp.p1[1] > ramp.p2[1] else (ramp.p2, ramp.p1)
        corner = intersect(ramp, ground)
        if corner is not None:
            # the ramp ends exactly where it meets the ground: clip it if it overshoots
            # (tracing into the angle label below), extend it if it stops just short
            t = float((corner - high) @ (low - high) / ((low - high) @ (low - high)))
            if 0.5 <= t <= 1 + MEET_TOL * short / max(1.0, float(np.linalg.norm(low - high))):
                low = corner
        lines = [
            Line(p1=tuple(high), p2=tuple(low), role="ramp", thickness_px=stroke,
                 confidence=round(ramp.coverage, 3)),
            Line(p1=tuple(ground.p1), p2=tuple(ground.p2), role="ground", thickness_px=stroke,
                 confidence=round(ground.coverage, 3)),
        ]
        ball, ball_warnings = pick_ball(up_circles, low, high, stroke)
        warnings += ball_warnings

    rec = Recognition(
        image_id=image_id,
        paper_found=cap.paper_found,
        paper_corners=cap.paper_corners,
        warped_size=(up_w, up_h),
        lines=lines,
        ball=ball,
        labels=[],
        warnings=warnings,
    )
    return Result(rec, up_image, up_mask, k, rules,
                  debug={"segments": len(segs), "circles": len(circles), "stroke": stroke})


def pick_ball(circles: list[Circle], low: np.ndarray, high: np.ndarray,
              stroke: float) -> tuple[Ball | None, list[str]]:
    warnings = []
    on, floating = [], []
    for c in circles:
        t, d = _ramp_coords(c.center, low, high)
        if not -0.15 <= t <= 1.05:
            continue
        if -0.02 <= t and -0.5 * c.radius <= d <= 2.2 * c.radius + stroke:
            on.append((t, c))
        elif 2.2 * c.radius + stroke < d <= 8 * c.radius or t < -0.02 and d > 0:
            floating.append((t, c))
    pool = on or floating
    if pool:
        # loops in handwriting ("o", "g", "°") are much smaller than a drawn ball
        biggest = max(c.radius for _, c in pool)
        pool = [(t, c) for t, c in pool if c.radius >= 0.5 * biggest]
    if not pool:
        return None, ["no ball found — draw a circle on the ramp"]
    if len(pool) > 1:
        warnings.append(f"{len(pool)} balls found — using the one nearest the top of the ramp")
    t, c = min(pool, key=lambda tc: tc[0])
    if not on:
        warnings.append("ball is not touching the ramp — placed it on the ramp")
    return Ball(center=tuple(c.center), radius=c.radius, confidence=round(c.score, 3)), warnings


def overlay_png(image: np.ndarray, rec: Recognition) -> bytes:
    """Outlines on the upright photo: ramp red, ground blue, ball green."""
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    width = max(3, round(min(pil.size) / 150))
    colors = {"ramp": (220, 38, 38), "ground": (37, 99, 235)}
    for line in rec.lines:
        draw.line([line.p1, line.p2], fill=colors[line.role], width=width)
    if rec.ball:
        (cx, cy), r = rec.ball.center, rec.ball.radius
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(22, 163, 74), width=width)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()
