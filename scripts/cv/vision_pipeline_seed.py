"""
SEED CODE (teammate's original CV pipeline, kept as-is for reference).
P1.1 / P1.2 in plan.md start from this file and port it into backend/app/cv/
with the changes listed in plan.md section 6 ("Seed CV pipeline"). Do not import
this file from the app.

pipeline: ingest -> flatten -> clean -> detect ramp -> erase lines
          -> components -> classify (ball / block / spring / text glyph)
          -> merge glyphs into text boxes -> JSON

OpenCV handles precise geometry.
Gemini optionally interprets labels and semantic roles.

every knob you may want to tune lives in params and is marked # tune.
"""
import cv2
import numpy as np
import math
import json
import os


# parameters: fill-in-the-blank section. start with the defaults, then run
# `python eval_dataset.py --split val` and adjust one knob at a time.
params = dict(
    # flatten (photo mode only; screenshots skip this)
    out_size=(1100, 850),        # tune: flattened letter-paper canvas (w, h)
    paper_min_area_frac=0.2,     # tune: paper must cover at least this much of the photo
    # clean
    blur_k=3,                    # tune: 3 or 5; larger erases thin strokes
    adapt_block=31,              # tune: adaptive-threshold window (photo mode), odd
    adapt_c=15,                  # tune: lower = more ink kept, higher = less paper noise
    ink_thresh=None,             # tune: None = Otsu; or a fixed 0-255 cut (screenshot mode)
    # ramp (hypotenuse) via Hough
    hough_thresh=60,             # tune: votes needed for a line
    hough_min_len=80,            # tune: shortest line (px) that can be a ramp
    hough_max_gap=12,            # tune: gap bridged along a line (ball/block sit on it)
    ramp_angle_min=8,            # tune: degrees from horizontal
    ramp_angle_max=60,           # tune
    ramp_angle_tol=3.0,          # tune: degrees, for merging collinear segments
    ramp_rho_tol=6.0,            # tune: px, for merging collinear segments
    bbox_pad=2,                  # matches the dataset's 2 px box padding
    # line erasing (so ball/block stop touching the ramp)
    erase_thick=5,               # tune: px painted over detected lines
    erase_flat_min_len=70,       # tune: flat/vertical lines at least this long are erased
    # component classification
    rejoin_dilate=1,             # tune: 0 to 2; reconnects ring gaps after erasing
    glyph_max_h=26,              # tune: a component this short can be text
    glyph_max_w=64,              # tune: touching letters ("m/s") fuse into wide blobs
    glyph_single_w=26,           # tune: narrower than this is always text, no density check
    glyph_min_extent=0.18,       # tune: ink/bbox area for wide blobs; thin arrows are ~0.05-0.12
    ball_diam_min=30,            # tune: px
    ball_diam_max=56,            # tune: px
    ball_circularity_min=0.84,   # tune: 4*pi*area/perimeter^2 of the convex hull; circle ~0.9+, square 0.785
    block_side_min=28,           # tune
    block_side_max=80,           # tune
    block_rect_fill_min=0.88,    # tune: hull area / min-area-rect area
    spring_min_verts=18,         # tune: polygon vertices (zigzag has many, arrow has few)
    spring_min_len=45,           # tune: px
    # text boxes
    text_block=15,               # tune: odd; local window for the text-only ink mask
    text_c=8,                    # tune: lower = more faint strokes kept, more noise
    text_min_area=2,             # tune: px; keep tiny stroke fragments of faint text
    text_gap=12,                 # tune: px; glyphs closer than this join into one label
    text_gap_y=5,                # tune
    text_min_w=20,               # tune: drop boxes narrower than this (stray marks)
    text_pad=1,                  # tune
)


# ingest img
def ingest(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


# flatten img if it's a photo, skip if it's a screenshot
def find_paper(img):
    gray = cv2.GaussianBlur(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0
    )
    edges = cv2.dilate(
        cv2.Canny(gray, 50, 150), None, iterations=2
    )
    cnts, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_area = (
        params["paper_min_area_frac"] * img.shape[0] * img.shape[1]
    )
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(
            c, 0.02 * cv2.arcLength(c, True), True
        )
        if len(approx) == 4 and cv2.contourArea(approx) > min_area:
            return approx.reshape(4, 2).astype("float32")
    return None  # caller falls back to four clicked corners


def order_corners(p):
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.array(
        [p[s.argmin()], p[d.argmin()],
         p[s.argmax()], p[d.argmax()]], "float32"
    )


def flatten(img, is_screenshot=True, corners=None):
    if is_screenshot:
        return img

    corners = (
        np.asarray(corners, dtype="float32")
        if corners is not None else find_paper(img)
    )
    if corners is None:
        raise ValueError("paper not found: pass four manual corners")

    # Preserve US letter landscape aspect ratio
    w, h = params["out_size"]
    dst = np.array(
        [[0, 0], [w - 1, 0],
         [w - 1, h - 1], [0, h - 1]], "float32"
    )
    M = cv2.getPerspectiveTransform(
        order_corners(corners), dst
    )
    return cv2.warpPerspective(img, M, (w, h))


# clean -> binary ink (white = ink)
def clean(flat, is_screenshot=True):
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    k = params["blur_k"]

    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    if is_screenshot:
        if params["ink_thresh"] is None:
            _, ink = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            _, ink = cv2.threshold(
                gray, params["ink_thresh"],
                255, cv2.THRESH_BINARY_INV
            )

        if ink.mean() > 127:
            ink = 255 - ink
    else:
        ink = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            params["adapt_block"], params["adapt_c"]
        )
        ink = cv2.morphologyEx(
            ink, cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )
        ink[:12, :] = ink[-12:, :] = 0
        ink[:, :12] = ink[:, -12:] = 0

    return ink


# ramp components, merged across gaps left by ball/block/spring
def find_line_segments(ink):
    segs = cv2.HoughLinesP(
        ink, 1, np.pi / 180,
        params["hough_thresh"],
        minLineLength=params["hough_min_len"],
        maxLineGap=params["hough_max_gap"]
    )
    if segs is None:
        return []
    segs = np.asarray(segs).reshape(-1, 4)
    return [tuple(map(float, s)) for s in segs]


def segment_angle(s):
    x1, y1, x2, y2 = s
    return math.degrees(
        math.atan2(abs(y2 - y1), abs(x2 - x1))
    )


def detect_ramp(ink):
    segs = find_line_segments(ink)
    slanted = [
        s for s in segs
        if params["ramp_angle_min"]
        <= segment_angle(s)
        <= params["ramp_angle_max"]
    ]

    if not slanted:
        return None, segs

    best = max(
        slanted,
        key=lambda s: math.hypot(
            s[2] - s[0], s[3] - s[1]
        )
    )
    x1, y1, x2, y2 = best
    ux, uy = (x2 - x1), (y2 - y1)
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    nx, ny = -uy, ux
    rho0 = x1 * nx + y1 * ny
    ang0 = math.degrees(math.atan2(uy, ux)) % 180
    pts = []

    for s in slanted:
        a = math.degrees(
            math.atan2(
                s[3] - s[1], s[2] - s[0]
            )
        ) % 180
        da = min(abs(a - ang0), 180 - abs(a - ang0))
        r = s[0] * nx + s[1] * ny

        if (
            da <= params["ramp_angle_tol"]
            and abs(r - rho0) <= params["ramp_rho_tol"]
        ):
            pts += [
                (s[0], s[1]),
                (s[2], s[3])
            ]

    proj = [
        (p[0] * ux + p[1] * uy, p)
        for p in pts
    ]
    p_lo, p_hi = min(proj)[1], max(proj)[1]
    return (p_lo, p_hi), segs


def box_around_points(pts, img_w, img_h, pad):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [
        max(0, min(xs) - pad),
        max(0, min(ys) - pad),
        min(img_w - 1, max(xs) + pad),
        min(img_h - 1, max(ys) + pad)
    ]


# erase long straight lines so touching objects separate
def erase_lines(ink, ramp, segs):
    mask = np.zeros_like(ink)
    t = params["erase_thick"]

    if ramp is not None:
        a, b = ramp
        cv2.line(
            mask,
            tuple(map(int, a)),
            tuple(map(int, b)),
            255, t
        )

    for s in segs:
        ang = segment_angle(s)
        if (
            (ang < 4 or ang > 86)
            and math.hypot(
                s[2] - s[0], s[3] - s[1]
            ) >= params["erase_flat_min_len"]
        ):
            cv2.line(
                mask,
                (int(s[0]), int(s[1])),
                (int(s[2]), int(s[3])),
                255, t
            )

    return cv2.bitwise_and(
        ink, cv2.bitwise_not(mask)
    )


# classify remaining components
def shape_measurements(cnt):
    hull = cv2.convexHull(cnt)
    area = cv2.contourArea(hull)
    peri = cv2.arcLength(hull, True)
    circ = (
        4 * math.pi * area /
        (peri * peri + 1e-9)
    )
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(hull)
    fill = area / (rw * rh + 1e-9)
    return hull, area, circ, (cx, cy), (rw, rh), fill


def classify_components(rest, img_w, img_h):
    if params["rejoin_dilate"] > 0:
        rest = cv2.dilate(
            rest,
            np.ones((3, 3), np.uint8),
            iterations=params["rejoin_dilate"]
        )

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            rest, connectivity=8
        )
    )
    out = dict(
        ball=[], block=[], spring=[],
        glyph=[], other=[]
    )
    pad = params["bbox_pad"]

    for i in range(1, n):
        x, y, w, h, area = stats[i]

        if area < 8:
            continue

        comp = (
            (labels == i).astype(np.uint8) * 255
        )
        extent = area / float(w * h)

        if (
            h <= params["glyph_max_h"]
            and (
                w <= params["glyph_single_w"]
                or (
                    w <= params["glyph_max_w"]
                    and extent >= params["glyph_min_extent"]
                )
            )
        ):
            out["glyph"].append(
                [x, y, x + w, y + h]
            )
            continue

        cnts, _ = cv2.findContours(
            comp,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cnt = max(cnts, key=cv2.contourArea)

        hull, harea, circ, ctr, (rw, rh), fill = (
            shape_measurements(cnt)
        )

        box = [
            max(0, x),
            max(0, y),
            min(img_w - 1, x + w),
            min(img_h - 1, y + h)
        ]

        nverts = len(
            cv2.approxPolyDP(cnt, 1.5, True)
        )
        side_min = min(rw, rh)
        side_max = max(rw, rh)

        if (
            params["ball_diam_min"] <= w <= params["ball_diam_max"]
            and params["ball_diam_min"] <= h <= params["ball_diam_max"]
            and circ >= params["ball_circularity_min"]
        ):
            out["ball"].append([
                x - pad, y - pad,
                x + w + pad, y + h + pad
            ])

        elif (
            params["block_side_min"] <= side_min
            and side_max <= params["block_side_max"]
            and fill >= params["block_rect_fill_min"]
        ):
            out["block"].append([
                x - pad, y - pad,
                x + w + pad, y + h + pad
            ])

        elif (
            nverts >= params["spring_min_verts"]
            and max(w, h) >= params["spring_min_len"]
        ):
            out["spring"].append([
                x - pad, y - pad,
                x + w + pad, y + h + pad
            ])

        else:
            out["other"].append(box)

    return out


def text_glyphs(flat, ramp, segs, exclude):
    """local-contrast ink mask for faint text."""
    img_h, img_w = flat.shape[:2]
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    b = params["text_block"] | 1

    tink = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        b, params["text_c"]
    )
    tink = erase_lines(tink, ramp, segs)

    n, _, stats, cents = (
        cv2.connectedComponentsWithStats(
            tink, connectivity=8
        )
    )
    out = []

    for i in range(1, n):
        x, y, w, h, area = stats[i]

        if (
            area < params["text_min_area"]
            or h > params["glyph_max_h"]
            or w > params["glyph_max_w"]
        ):
            continue

        if (
            w > params["glyph_single_w"]
            and area / float(w * h)
            < params["glyph_min_extent"]
        ):
            continue

        cx, cy = cents[i]

        if any(
            e[0] <= cx <= e[2]
            and e[1] <= cy <= e[3]
            for e in exclude
        ):
            continue

        out.append(
            [x, y, x + w, y + h]
        )

    return out


# detector text: merge glyphs into label boxes
def merge_glyphs(glyphs, img_w, img_h):
    if not glyphs:
        return []

    canvas = np.zeros(
        (img_h, img_w), np.uint8
    )
    for x0, y0, x1, y1 in glyphs:
        canvas[y0:y1, x0:x1] = 255

    kx, ky = (
        params["text_gap"],
        params["text_gap_y"]
    )
    merged = cv2.dilate(
        canvas,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (2 * kx + 1, 2 * ky + 1)
        )
    )
    n, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            merged, connectivity=8
        )
    )
    boxes = []

    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        sub = canvas[y:y + h, x:x + w]
        ys, xs = np.where(sub > 0)

        if len(xs) == 0:
            continue

        bx0, by0 = x + xs.min(), y + ys.min()
        bx1, by1 = x + xs.max() + 1, y + ys.max() + 1

        if bx1 - bx0 < params["text_min_w"]:
            continue

        p = params["text_pad"]
        boxes.append([
            bx0 - p, by0 - p,
            bx1 + p, by1 + p
        ])

    return boxes


# end-to-end pipeline
def detect(path, is_screenshot=True, corners=None):
    img = ingest(path)
    flat = flatten(
        img, is_screenshot, corners
    )
    img_h, img_w = flat.shape[:2]
    ink = clean(flat, is_screenshot)
    ramp, segs = detect_ramp(ink)
    rest = erase_lines(ink, ramp, segs)
    comps = classify_components(
        rest, img_w, img_h
    )

    det = {
        "ball": comps["ball"],
        "block": comps["block"],
        "spring": comps["spring"],
        "ramp": [],
        "ramp_angle_deg": None
    }

    excl = (
        comps["ball"] +
        comps["block"] +
        comps["spring"]
    )

    det["text"] = merge_glyphs(
        text_glyphs(flat, ramp, segs, excl),
        img_w, img_h
    )

    if ramp is not None:
        det["ramp"] = [
            box_around_points(
                ramp, img_w, img_h,
                params["bbox_pad"]
            )
        ]
        (ax, ay), (bx, by) = ramp
        det["ramp_angle_deg"] = math.degrees(
            math.atan2(
                abs(by - ay), abs(bx - ax)
            )
        )
        det["ramp_points"] = [
            list(ramp[0]), list(ramp[1])
        ]

    return flat, ink, det


def process(path, image_id, is_screenshot=True,
            corners=None, out_dir="masks"):
    """contract A -> B: ink components with masks."""
    flat, ink, det = detect(
        path, is_screenshot, corners
    )
    os.makedirs(out_dir, exist_ok=True)

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            ink, connectivity=8
        )
    )
    comps = []

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 40:
            continue

        mp = f"{out_dir}/{image_id}_k{len(comps)}.png"

        cv2.imwrite(
            mp,
            ((labels == i) * 255).astype("uint8")
        )

        comps.append({
            "id": f"k{len(comps)}",
            "bbox": [
                int(x), int(y),
                int(w), int(h)
            ],
            "mask_path": mp
        })

    return {
        "image_id": image_id,
        "paper_px": [
            flat.shape[1], flat.shape[0]
        ],
        "components": comps,
        "detections": json.loads(
            json.dumps(det, default=float)
        )
    }


def draw_debug(flat, det):
    vis = flat.copy()
    colors = dict(
        ball=(0, 0, 255),
        block=(255, 0, 0),
        ramp=(0, 160, 0),
        spring=(200, 0, 200),
        text=(0, 140, 255)
    )

    for k, c in colors.items():
        for b in det[k]:
            cv2.rectangle(
                vis,
                (int(b[0]), int(b[1])),
                (int(b[2]), int(b[3])),
                c, 2
            )
            cv2.putText(
                vis, k,
                (int(b[0]), int(b[1]) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, c, 1
            )

    return vis


# geometric helpers for the B -> C handoff
def line_width(ink, p1, p2):
    """Estimate median stroke width in pixels."""
    dist = cv2.distanceTransform(
        ink, cv2.DIST_L2, 3
    )
    a = np.array(p1, float)
    b = np.array(p2, float)
    readings = []

    for t in np.linspace(.15, .85, 13):
        x, y = np.rint(
            a * (1 - t) + b * t
        ).astype(int)

        if (
            0 <= y < ink.shape[0]
            and 0 <= x < ink.shape[1]
        ):
            if ink[y, x]:
                readings.append(
                    float(2 * dist[y, x])
                )

    return (
        round(float(np.median(readings)), 2)
        if len(readings) >= 4 else None
    )


def detect_extra_shapes(ink, det):
    """Heuristic arrow and pendulum recognition."""
    H, W = ink.shape
    ramp = det.get("ramp_points")
    balls = det["ball"]

    centers = [
        np.array([
            (b[0] + b[2]) / 2,
            (b[1] + b[3]) / 2
        ])
        for b in balls
    ]
    radii = [
        ((b[2] - b[0]) + (b[3] - b[1])) / 4
        for b in balls
    ]

    erased = ink.copy()

    if ramp:
        cv2.line(
            erased,
            tuple(map(int, ramp[0])),
            tuple(map(int, ramp[1])),
            0, 9
        )

    for s in find_line_segments(ink):
        length = math.hypot(
            s[2] - s[0], s[3] - s[1]
        )
        if segment_angle(s) < 3.5 and length > 90:
            cv2.line(
                erased,
                (int(s[0]), int(s[1])),
                (int(s[2]), int(s[3])),
                0, 8
            )

    for b in balls + det["block"] + det["text"]:
        x0, y0, x1, y1 = map(int, b)
        cv2.rectangle(
            erased,
            (max(0, x0 - 4), max(0, y0 - 4)),
            (min(W - 1, x1 + 4),
             min(H - 1, y1 + 4)),
            0, -1
        )

    n, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            erased, connectivity=8
        )
    )

    arrows, dots = [], []

    for i in range(1, n):
        x, y, w, h, area = map(
            int, stats[i]
        )

        if not 5 <= area <= 1400:
            continue

        mask = (
            (labels[y:y+h, x:x+w] == i)
            .astype("uint8") * 255
        )
        cnts, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not cnts:
            continue

        c = max(cnts, key=cv2.contourArea)
        (_, _), radius = cv2.minEnclosingCircle(c)

        # Small circular mark: potential pendulum pivot
        if (
            4 <= w <= 17
            and 4 <= h <= 17
            and .65 <= w / max(1, h) <= 1.5
            and cv2.contourArea(c) /
            (math.pi * radius * radius + 1e-6) > .40
        ):
            dots.append([
                float(centroids[i][0]),
                float(centroids[i][1])
            ])
            continue

        # Long narrow mark near a ball with wider end
        if (
            max(w, h) < 26
            or max(w, h) / max(1, min(w, h)) < 1.8
        ):
            continue

        ys, xs = np.where(mask > 0)
        pts = np.column_stack(
            (xs + x, ys + y)
        ).astype(float)
        axis = cv2.PCACompute(
            pts, mean=None
        )[1][0]

        proj = pts @ axis
        lo, hi = np.percentile(
            proj, [3, 97]
        )

        if hi - lo < 24:
            continue

        cross = pts @ np.array(
            [-axis[1], axis[0]]
        )
        spreads = []

        for v in (lo, hi):
            side = cross[
                np.abs(proj - v)
                < max(5, .13 * (hi - lo))
            ]
            spreads.append(
                float(np.ptp(side))
                if len(side) > 3 else 0
            )

        if (
            max(spreads)
            < 1.45 * max(1, min(spreads))
            or max(spreads) < 7
        ):
            continue

        middle = pts.mean(axis=0)
        tail = middle + axis * (
            lo - middle @ axis
        )
        head = middle + axis * (
            hi - middle @ axis
        )

        if spreads[0] < spreads[1]:
            tail, head = head, tail

        if not any(
            min(
                np.linalg.norm(tail - c),
                np.linalg.norm(head - c)
            ) < 2.5 * r + 25
            for c, r in zip(centers, radii)
        ):
            continue

        arrows.append({
            "tail": tail.round(2).tolist(),
            "head": head.round(2).tolist()
        })

    # Dot connected to a ball by a line
    pivots = []
    lines = find_line_segments(ink)

    for dot in dots:
        d = np.array(dot)

        for c, r in zip(centers, radii):
            if not (
                1.8 * r
                < np.linalg.norm(d - c)
                < 8 * r
            ):
                continue

            for s in lines:
                a = np.array(s[:2])
                b = np.array(s[2:])

                if min(
                    np.linalg.norm(a - d),
                    np.linalg.norm(b - d)
                ) > 18:
                    continue

                if min(
                    np.linalg.norm(a - c),
                    np.linalg.norm(b - c)
                ) > r + 20:
                    continue

                if np.linalg.norm(a - b) < 25:
                    continue

                if segment_angle(s) < 8:
                    continue

                pivots.append({
                    "pivot": dot,
                    "string_end": c.round(2).tolist()
                })
                break

            if pivots and pivots[-1]["pivot"] == dot:
                break

    return arrows, pivots


# Module B -> C typed geometry
def recognize(path, image_id, is_screenshot=True,
              corners=None):
    """Convert OpenCV detections into typed objects."""
    flat, ink, det = detect(
        path, is_screenshot, corners
    )
    H, W = ink.shape
    objects = []

    def add(item):
        item["id"] = f"o{len(objects) + 1}"
        objects.append(item)
        return item["id"]

    # Ramp
    ramp = det.get("ramp_points")

    if ramp:
        width = line_width(ink, *ramp)
        add({
            "type": "segment",
            "role": "ramp",
            "thick": width >= 1.75 if width else True,
            "p1": ramp[0],
            "p2": ramp[1],
            "stroke_width_px": width
        })

    # Ground
    segs = find_line_segments(ink)
    ground = [
        s for s in segs
        if segment_angle(s) < 3.5
        and (s[1] + s[3]) / 2 > H * .45
        and math.hypot(
            s[2] - s[0], s[3] - s[1]
        ) >= max(90, .14 * W)
    ]

    if ground:
        s = max(
            ground,
            key=lambda z: math.hypot(
                z[2] - z[0], z[3] - z[1]
            )
        )
        width = line_width(
            ink, s[:2], s[2:]
        )
        add({
            "type": "segment",
            "role": "ground",
            "thick": width >= 1.75 if width else True,
            "p1": [s[0], s[1]],
            "p2": [s[2], s[3]],
            "stroke_width_px": width
        })

    # Balls
    for x0, y0, x1, y1 in det["ball"]:
        add({
            "type": "circle",
            "center": [
                (x0 + x1) / 2,
                (y0 + y1) / 2
            ],
            "radius": (
                (x1 - x0) + (y1 - y0)
            ) / 4
        })

    # Rotated blocks
    for x0, y0, x1, y1 in det["block"]:
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(W, int(x1))
        y1 = min(H, int(y1))

        roi = ink[y0:y1, x0:x1]
        cnts, _ = cv2.findContours(
            roi,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if cnts:
            pts = cv2.boxPoints(
                cv2.minAreaRect(
                    max(cnts, key=cv2.contourArea)
                )
            )
            corners2 = [
                [
                    round(float(a + x0), 2),
                    round(float(b + y0), 2)
                ]
                for a, b in pts
            ]
        else:
            corners2 = [
                [x0, y0], [x1, y0],
                [x1, y1], [x0, y1]
            ]

        add({
            "type": "rect",
            "corners": corners2
        })

    # Arrows and pendulums
    arrows, pivots = detect_extra_shapes(
        ink, det
    )

    for arrow in arrows:
        add({
            "type": "arrow",
            **arrow
        })

    for p in pivots:
        dot_id = add({
            "type": "dot",
            "center": p["pivot"]
        })
        add({
            "type": "segment",
            "role": "string",
            "thick": False,
            "p1": p["pivot"],
            "p2": p["string_end"],
            "pivot_id": dot_id
        })

    # Text regions
    for box in det["text"]:
        add({
            "type": "label",
            "text": None,
            "bbox": [
                float(v) for v in box
            ],
            "near": None
        })

    return {
        "image_id": image_id,
        "paper_px": [W, H],
        "objects": objects,
        "warnings": [
            "Arrow and pendulum detection are heuristic",
            "Stroke width is estimated",
            "Handwriting OCR requires Gemini"
        ]
    }


# Gemini semantic interpretation (NEW)
def enrich_with_gemini(flat, result):
    from google import genai
    from google.genai import types

    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError(
            "Set GEMINI_API_KEY in the environment"
        )

    annotated = flat.copy()

    # Add object IDs for Gemini to reference
    for obj in result["objects"]:
        typ = obj["type"]

        if typ in ("circle", "dot"):
            x, y = map(int, obj["center"])
        elif typ == "segment":
            x, y = map(int, obj["p1"])
        elif typ == "rect":
            x, y = map(int, obj["corners"][0])
        elif typ == "label":
            x, y = map(int, obj["bbox"][:2])
        elif typ == "arrow":
            x, y = map(int, obj["tail"])
        else:
            continue

        cv2.putText(
            annotated,
            obj["id"],
            (x, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2
        )

    ok1, original_png = cv2.imencode(
        ".png", flat
    )
    ok2, annotated_png = cv2.imencode(
        ".png", annotated
    )

    if not (ok1 and ok2):
        raise ValueError(
            "Could not encode images for Gemini"
        )

    schema = {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "object_id": {"type": "string"},
                        "text": {"type": "string"},
                        "near": {"type": "string"}
                    },
                    "required": [
                        "object_id", "text", "near"
                    ]
                }
            },
            "roles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "object_id": {"type": "string"},
                        "role": {"type": "string"}
                    },
                    "required": [
                        "object_id", "role"
                    ]
                }
            }
        },
        "required": [
            "labels", "roles"
        ]
    }

    prompt = (
        "Interpret this hand-drawn physics sketch. "
        "The first image is original. "
        "The second image has OpenCV object IDs. "
        "Read handwriting in existing label regions, "
        "associate labels with an existing nearby "
        "object if clear, and assign semantic roles "
        "to existing objects. Use only provided IDs. "
        "Do not infer exact geometry or invent objects. "
        "Return empty arrays for uncertain answers. "
        "Set 'near' to empty string if unclear. "
        "Objects: " + json.dumps(result["objects"])
    )

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            types.Part.from_bytes(
                data=original_png.tobytes(),
                mime_type="image/png"
            ),
            types.Part.from_bytes(
                data=annotated_png.tobytes(),
                mime_type="image/png"
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema
        )
    )

    interpretation = json.loads(response.text)
    by_id = {
        obj["id"]: obj
        for obj in result["objects"]
    }

    # Read labels but do not change OpenCV geometry
    for item in interpretation.get("labels", []):
        obj = by_id.get(item.get("object_id"))

        if obj is not None and obj["type"] == "label":
            obj["text"] = item.get("text") or None
            near = item.get("near")
            obj["near"] = (
                near
                if near in by_id and near != obj["id"]
                else None
            )

    allowed = {
        "ramp", "ground", "wall",
        "ball", "block", "string", "pendulum"
    }

    compatible = {
        "segment": {
            "ramp", "ground", "wall", "string"
        },
        "circle": {
            "ball", "pendulum"
        },
        "rect": {
            "block"
        },
        "dot": {
            "pendulum"
        }
    }

    # Semantic role kept separate from geometry role
    for item in interpretation.get("roles", []):
        obj = by_id.get(item.get("object_id"))
        role = str(
            item.get("role", "")
        ).strip().lower()

        if (
            obj is not None
            and role in allowed
            and role in compatible.get(
                obj["type"], set()
            )
        ):
            obj["semantic_role"] = role

    result["gemini_used"] = True
    return result


# save visual test outputs and JSON
def save_debug(path, out_dir, image_id,
               is_screenshot=True, corners=None,
               use_gemini=False):

    os.makedirs(out_dir, exist_ok=True)

    flat, ink, det = detect(
        path, is_screenshot, corners
    )

    cv2.imwrite(
        os.path.join(out_dir, "flattened.png"),
        flat
    )
    cv2.imwrite(
        os.path.join(out_dir, "ink.png"),
        ink
    )
    cv2.imwrite(
        os.path.join(out_dir, "detections.png"),
        draw_debug(flat, det)
    )

    a = process(
        path, image_id, is_screenshot, corners,
        out_dir=os.path.join(out_dir, "masks")
    )
    b = recognize(
        path, image_id, is_screenshot, corners
    )

    # NEW: optional Gemini interpretation
    if use_gemini:
        try:
            b = enrich_with_gemini(flat, b)
        except Exception as e:
            print(
                f"Gemini failed; using OpenCV only: {e}"
            )
            b["gemini_used"] = False
            b["warnings"].append(
                "Gemini unavailable; OpenCV-only output saved"
            )

    overlay = flat.copy()

    for obj in b["objects"]:
        typ = obj["type"]

        if typ == "segment":
            cv2.line(
                overlay,
                tuple(map(int, obj["p1"])),
                tuple(map(int, obj["p2"])),
                (0, 255, 0), 2
            )

        elif typ == "circle":
            cv2.circle(
                overlay,
                tuple(map(int, obj["center"])),
                int(obj["radius"]),
                (0, 0, 255), 2
            )

        elif typ == "rect":
            pts = np.array(
                obj["corners"], dtype=np.int32
            )
            cv2.polylines(
                overlay, [pts], True,
                (255, 0, 0), 2
            )

        elif typ == "arrow":
            cv2.arrowedLine(
                overlay,
                tuple(map(int, obj["tail"])),
                tuple(map(int, obj["head"])),
                (255, 0, 255), 2
            )

        elif typ == "dot":
            cv2.circle(
                overlay,
                tuple(map(int, obj["center"])),
                5, (0, 255, 255), 2
            )

    cv2.imwrite(
        os.path.join(out_dir, "typed_overlay.png"),
        overlay
    )

    with open(
        os.path.join(out_dir, "a_to_b.json"), "w"
    ) as f:
        json.dump(a, f, indent=2)

    with open(
        os.path.join(out_dir, "b_to_c.json"), "w"
    ) as f:
        json.dump(b, f, indent=2)

    return a, b


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PaperPhysics vision pipeline"
    )

    parser.add_argument(
        "image", help="Path to image"
    )
    parser.add_argument(
        "--image-id", default="demo"
    )
    parser.add_argument(
        "--photo", action="store_true",
        help="Detect and flatten photographed paper"
    )
    parser.add_argument(
        "--out", default="fixtures/demo"
    )
    parser.add_argument(
        "--corners", type=float, nargs=8,
        help="Manual TL TR BR BL corner coordinates"
    )

    # NEW: optional Gemini mode
    parser.add_argument(
        "--gemini", action="store_true",
        help="Use Gemini for labels and semantic roles"
    )

    args = parser.parse_args()

    corners = (
        np.array(
            args.corners, dtype=np.float32
        ).reshape(4, 2)
        if args.corners else None
    )

    _, result = save_debug(
        args.image,
        args.out,
        args.image_id,
        is_screenshot=not args.photo,
        corners=corners,
        use_gemini=args.gemini
    )

    print(json.dumps(result, indent=2))
