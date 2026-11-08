"""Extract a closed convex quadrilateral from actual ink near a semantic proposal."""
import math
import cv2
import numpy as np
from app.contracts import Block


def block_from_box(box, mask, stroke=3.):
    x1, y1, x2, y2 = box
    h, w = mask.shape
    size = max(x2-x1, y2-y1)
    pad = max(8, int(size*.45))
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
        if (min(lengths) < max(10, 3*stroke) or area < .12*(x2-x1)*(y2-y1)
                or area > 2.5*(x2-x1)*(y2-y1) or max(lengths) > .4*min(h,w)
                or not x1-pad*.3 <= center[0] <= x2+pad*.3
                or not y1-pad*.3 <= center[1] <= y2+pad*.3):
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
