"""Photo → flattened paper image + ink mask (P1.1).

1. Find the paper: the largest 4-corner contour of the bright region; perspective-warp it.
   No paper edge (synthetic renders, or the sheet fills the frame) → use the whole image.
2. Resize so the long side is WORK_LONG_SIDE px.
3. Ink = pixels clearly darker than their local background (a large median blur), which
   survives shadows. The threshold adapts per image (Otsu on the "darkness" values) so faint
   ruled-notebook lines (light cyan/pink, much less dark than pen) stay out of the mask.
4. Drop specks with connected components.
"""

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

WORK_LONG_SIDE = 1200
MIN_PAPER_AREA = 0.2  # of the photo
MAX_PAPER_AREA = 0.97  # bigger than this = the photo border itself, not a paper edge
MIN_DARKNESS = 40  # 0-255 below local background; nothing lighter than this is ink
MIN_SPECK_AREA_FRAC = 2e-5  # components smaller than this fraction of the image are noise

Corners = list[tuple[float, float]]


@dataclass
class Capture:
    image: np.ndarray  # flattened paper, BGR uint8
    mask: np.ndarray  # ink, uint8 0/255, same size as image
    paper_found: bool
    paper_corners: Corners | None  # in the original photo
    warnings: list[str] = field(default_factory=list)


def load_image(data: bytes) -> np.ndarray:
    """Bytes → BGR array with EXIF rotation applied."""
    pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def resize_long_side(img: np.ndarray, long_side: int = WORK_LONG_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if abs(scale - 1) < 1e-3:
        return img
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)


def order_corners(pts) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def find_paper(img: np.ndarray) -> Corners | None:
    """Largest bright 4-corner region, in img coordinates, or None."""
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_total = small.shape[0] * small.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        area = cv2.contourArea(contour)
        if not MIN_PAPER_AREA * area_total <= area <= MAX_PAPER_AREA * area_total:
            continue
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                quad = order_corners(approx)
                if (cv2.contourArea(approx) >= 0.85 * cv2.contourArea(hull)
                        and _paper_shaped(quad) and _sharp_edges(gray, quad)):
                    return [(float(x) / scale, float(y) / scale) for x, y in quad]
                break
    return None


def _paper_shaped(quad: np.ndarray) -> bool:
    """Sheets are between square and ~1:1.8 (A4 1.41, letter 1.29); shadows often aren't."""
    tl, tr, br, bl = quad
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return max(width, height) / max(1.0, min(width, height)) <= 1.8


def _sharp_edges(gray: np.ndarray, quad: np.ndarray, offset: float = 4.0,
                 min_step: float = 25.0) -> bool:
    """Every side that isn't on the photo border must be a crisp bright→dark step.

    A shadow boundary is a soft ramp, so the brightness a few px inside vs outside the side
    differs little; a real paper edge on a darker desk differs a lot.
    """
    h, w = gray.shape
    centre = quad.mean(axis=0)
    margin = 0.02 * max(h, w)
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        # lies along one photo edge (both ends near the same border), not cutting across it
        dists = lambda p: (p[0], p[1], w - 1 - p[0], h - 1 - p[1])  # noqa: E731
        if any(da < margin and db < margin for da, db in zip(dists(a), dists(b))):
            continue
        direction = (b - a) / max(1e-6, np.linalg.norm(b - a))
        normal = np.array([-direction[1], direction[0]])
        if np.dot(centre - a, normal) < 0:
            normal = -normal  # point inwards
        steps = []
        for t in np.linspace(0.1, 0.9, 15):
            p = a + t * (b - a)
            inside, outside = p + offset * normal, p - offset * normal
            if not (0 <= outside[0] < w and 0 <= outside[1] < h):
                continue
            steps.append(float(gray[int(inside[1]), int(inside[0])])
                         - float(gray[int(outside[1]), int(outside[0])]))
        if not steps or np.median(steps) < min_step:
            return False
    return True


def warp(img: np.ndarray, corners: Corners) -> np.ndarray:
    tl, tr, br, bl = order_corners(corners)
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    scale = WORK_LONG_SIDE / max(width, height)
    w, h = round(width * scale), round(height * scale)
    dst = np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), dst)
    return cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def darkness(img: np.ndarray) -> np.ndarray:
    """How much darker each pixel is than its local background (float32, 0-255)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = max(21, (min(gray.shape) // 25) | 1)
    # median of a half-size image is ~4× faster and just as good for a background estimate
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(small, (k // 2) | 1)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(bg.astype(np.float32) - gray.astype(np.float32), 0, 255)


def ink_mask(img: np.ndarray) -> np.ndarray:
    dark = darkness(img)
    candidates = dark[dark > MIN_DARKNESS / 2]
    threshold = float(MIN_DARKNESS)
    if candidates.size > 100:
        otsu, _ = cv2.threshold(candidates.astype(np.uint8).reshape(-1, 1), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = max(threshold, float(otsu))
    mask = (dark > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return remove_specks(mask)


def remove_specks(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(8, MIN_SPECK_AREA_FRAC * mask.size)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return (keep[labels] * 255).astype(np.uint8)


def capture(img: np.ndarray) -> Capture:
    corners = find_paper(img)
    warnings = []
    if corners is not None:
        flat = warp(img, corners)
    else:
        flat = resize_long_side(img)
        warnings.append("paper edge not found — using the whole photo")
    return Capture(image=flat, mask=ink_mask(flat), paper_found=corners is not None,
                   paper_corners=corners, warnings=warnings)


def apply_corners(img: np.ndarray, corners: Corners) -> Capture:
    """Manual four-corner fallback: the user tapped the paper corners on the photo."""
    if len(corners) != 4:
        raise ValueError("need exactly 4 corners")
    flat = warp(img, corners)
    return Capture(image=flat, mask=ink_mask(flat), paper_found=True,
                   paper_corners=[tuple(map(float, c)) for c in order_corners(corners)])
