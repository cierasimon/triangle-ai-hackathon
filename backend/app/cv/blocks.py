"""Extract a closed convex quadrilateral from actual ink near a semantic proposal."""
import math
import cv2
import numpy as np
from app.contracts import Block


def block_from_box(box, mask, stroke=3.):
    """The one ink-supported quadrilateral for a proposed block box. A proposal box can sit a little
    off the drawn block (it may cover only a corner, or include a label), so if nothing fits in the
    box's own neighbourhood, look further out for a quadrilateral that still overlaps the box."""
    return _block_from_box(box, mask, stroke, .45, False) or _block_from_box(box, mask, stroke, 1.3, True)


def _block_from_box(box, mask, stroke, pad_scale, must_overlap):
    x1, y1, x2, y2 = box
    h, w = mask.shape
    size = max(x2-x1, y2-y1)
    pad = max(8, int(size*pad_scale))
    left, top = max(0, int(x1)-pad), max(0, int(y1)-pad)
    crop = mask[top:min(h, int(y2)+pad), left:min(w, int(x2)+pad)]
    if not crop.size:
        return None
    contours, _ = cv2.findContours(crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    fits = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        quad = cv2.approxPolyDP(contour, .025*perimeter, True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        pts = quad[:, 0, :].astype(float) + [left, top]
        area = abs(cv2.contourArea(pts.astype(np.float32)))
        center = pts.mean(axis=0)
        sides = np.roll(pts, -1, axis=0)-pts
        lengths = np.linalg.norm(sides, axis=1)
        if (min(lengths) < max(10, 3*stroke) or area < .03*(x2-x1)*(y2-y1)
                or area > 2.5*(x2-x1)*(y2-y1) or max(lengths) > .4*min(h,w)
                or not must_overlap and not (x1-pad*.3 <= center[0] <= x2+pad*.3
                                             and y1-pad*.3 <= center[1] <= y2+pad*.3)):
            continue
        if must_overlap:  # the quadrilateral's extent must share at least 5% of its area with the box
            qx1, qy1, qx2, qy2 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
            shared = max(0, min(qx2, x2)-max(qx1, x1)) * max(0, min(qy2, y2)-max(qy1, y1))
            if shared < .05*(qx2-qx1)*(qy2-qy1):
                continue
        cosines = [abs(np.dot(sides[i], sides[(i+1)%4])/(lengths[i]*lengths[(i+1)%4])) for i in range(4)]
        if max(cosines) > .55:  # Reject triangles, arrows and strongly skewed marks.
            continue
        samples = np.concatenate([a[None,:]+np.linspace(0,1,25)[:,None]*(b-a)[None,:]
                                  for a,b in zip(pts,np.roll(pts,-1,axis=0))])
        radius = max(2, int(stroke*1.5))
        support = np.mean([mask[max(0,int(y)-radius):min(h,int(y)+radius+1),
                               max(0,int(x)-radius):min(w,int(x)+radius+1)].any() for x,y in samples])
        if support < .85:
            continue
        score = support - .1*math.dist(center, ((x1+x2)/2,(y1+y2)/2))/max(size,1)
        fits.append((score, Block(vertices=[tuple(p) for p in pts],center=tuple(center),confidence=float(support))))
    unique = []
    for _, block in sorted(fits, key=lambda pair: pair[0], reverse=True):
        side = min(math.dist(a,b) for a,b in zip(block.vertices, block.vertices[1:]+block.vertices[:1]))
        if not any(math.dist(block.center, other.center) < .3*side for other in unique):
            unique.append(block)
    # Nested contours of one stroke are duplicates; distinct shapes in one box are ambiguous.
    return unique[0] if len(unique) == 1 else None


def _squares(mask, trace, stroke):
    """Every drawn square in the ink mask, found without any model: closed, convex, four near-right
    corners, roughly equal sides, ink along all four edges. The ramp (a triangle), the ground, a
    right-angle mark (too small) and handwriting do not pass. This is what makes blocks independent
    of a vision model's boxes, which are noisy and sometimes missing."""
    h, w = mask.shape
    short = min(h, w)
    contours, _ = cv2.findContours(trace, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    fits = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if len(contour) < 8 or perimeter < 16 * max(5, stroke):
            continue
        quad = cv2.approxPolyDP(contour, .025*perimeter, True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        pts = quad[:, 0, :].astype(float)
        sides = np.roll(pts, -1, axis=0) - pts
        lengths = np.linalg.norm(sides, axis=1)
        if min(lengths) < max(.035*short, 3*stroke) or max(lengths) > .4*short or max(lengths) > 1.8*min(lengths):
            continue
        cosines = [abs(np.dot(sides[i], sides[(i+1) % 4])/(lengths[i]*lengths[(i+1) % 4])) for i in range(4)]
        if max(cosines) > .55:
            continue
        samples = np.concatenate([a[None, :]+np.linspace(0, 1, 25)[:, None]*(b-a)[None, :]
                                  for a, b in zip(pts, np.roll(pts, -1, axis=0))])
        radius = max(2, int(stroke*1.5))
        support = float(np.mean([mask[max(0, int(y)-radius):min(h, int(y)+radius+1),
                                      max(0, int(x)-radius):min(w, int(x)+radius+1)].any() for x, y in samples]))
        if support < .85:
            continue
        center = pts.mean(axis=0)
        fits.append((support, float(np.mean(lengths)), Block(vertices=[tuple(p) for p in pts], center=tuple(center),
                                                                  confidence=support)))
    return sorted(fits, key=lambda f: (-f[0], -f[1]))


def detect_blocks(mask, stroke=3.):
    """Every drawn square in the ink mask, found without any model: closed, convex, four near-right
    corners, roughly equal sides, ink along all four edges. The ramp (a triangle), the ground, a
    right-angle mark (too small) and handwriting do not pass. This is what makes blocks independent
    of a vision model's boxes, which are noisy and sometimes missing.

    Looked for twice: in the mask as it is, and with 1-2 px gaps closed (thin pen lines often leave a
    gap where a block's corner meets the ramp, so the square is not a closed shape)."""
    size = max(3, int(round(2.5*stroke)) | 1)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    unique = []  # the outer and inner outline of one stroke, or the same square twice, are one square
    for fit in _squares(mask, mask, stroke) + _squares(mask, closed, stroke):
        _, side, block = fit
        if not any(math.dist(block.center, other.center) < .5*side for _, _, other in unique):
            unique.append(fit)
    return [block for _, _, block in unique]
