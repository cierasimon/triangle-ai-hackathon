"""Precision / recall / F1 for the PaperPhysics v3 dataset.

Usage:
    python eval_dataset.py --root paperphysics_v3_3000 --split val
    python eval_dataset.py --root paperphysics_v3_3000 --split test --iou 0.5

Tune on val, report on test (never tune on test).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

import paperphysics_capture as pc

CLASSES = ["ball", "block", "ramp", "spring", "text"]


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(preds, gts, thr):
    """Greedy one-to-one matching by IoU. Returns (tp, fp, fn)."""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)), reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for s, i, j in pairs:
        if s < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="paperphysics_v3_3000")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ann = {}
    for line in open(os.path.join(a.root, "annotations.jsonl")):
        d = json.loads(line)
        ann[d["image"]] = d
    names = [l.strip() for l in open(os.path.join(a.root, f"{a.split}.txt")) if l.strip()]
    if a.limit:
        names = names[:a.limit]

    tot = defaultdict(lambda: [0, 0, 0])
    ang_err, ramp_missing = [], 0
    for k, name in enumerate(names):
        d = ann[name]
        _, _, det = pc.detect(os.path.join(a.root, name), is_screenshot=True)
        gt = defaultdict(list)
        for o in d["objects"]:
            gt[o["class"]].append(o["bbox_xyxy"])
        for t in d["text_annotations"]:
            gt["text"].append(t["bbox_xyxy"])
        for c in CLASSES:
            tp, fp, fn = match(det[c], gt[c], a.iou)
            tot[c][0] += tp; tot[c][1] += fp; tot[c][2] += fn
        gt_ang = d["physics"]["angle_degrees"]["value"] if d["physics"]["angle_degrees"] else None
        ramp_ang = next((o["angle_degrees"] for o in d["objects"] if o["class"] == "ramp"), gt_ang)
        if det["ramp_angle_deg"] is None:
            ramp_missing += 1
        elif ramp_ang is not None:
            ang_err.append(abs(det["ramp_angle_deg"] - ramp_ang))
        if (k + 1) % 50 == 0:
            print(f"  ...{k + 1}/{len(names)}", file=sys.stderr)

    print(f"\nsplit={a.split}  images={len(names)}  IoU>={a.iou}")
    print(f"{'class':8s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'precision':>10s} {'recall':>8s} {'F1':>6s}")
    mtp = mfp = mfn = 0
    for c in CLASSES:
        tp, fp, fn = tot[c]
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f = 2 * p * r / (p + r) if p + r else 0
        print(f"{c:8s} {tp:6d} {fp:6d} {fn:6d} {p:10.3f} {r:8.3f} {f:6.3f}")
        mtp += tp; mfp += fp; mfn += fn
    p = mtp / (mtp + mfp) if mtp + mfp else 0
    r = mtp / (mtp + mfn) if mtp + mfn else 0
    print(f"{'ALL':8s} {mtp:6d} {mfp:6d} {mfn:6d} {p:10.3f} {r:8.3f} {2 * p * r / (p + r + 1e-9):6.3f}")
    if ang_err:
        e = np.array(ang_err)
        print(f"\nramp angle error: mean {e.mean():.2f} deg, median {np.median(e):.2f}, "
              f"within 2 deg {np.mean(e <= 2):.1%}, ramp not found in {ramp_missing} images")


if __name__ == "__main__":
    main()
