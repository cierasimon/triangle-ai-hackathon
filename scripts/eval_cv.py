"""Recognition accuracy (P1.2 / P1.8).

    .venv/bin/python scripts/eval_cv.py --set synthetic|real|all [--dir path] [--overlays out/]

Synthetic ground truth comes from each image's annotation JSON. Checked: the drawn ramp slope
equals `physics.angle_degrees` exactly (the generator draws the high point at
low + L·(cos θ, −sin θ)), so angle is scored against `angle_degrees`.
  ball ok   : centre within 0.5 r and radius within 35% of the annotation
  ramp ok   : angle error ≤ 3°, low end within 5% and high end within 10% of the short side
  ground ok : ground height within 2% of the short side
  all ok    : ball, ramp and ground ok

Real photos (fixtures/real/labels.json, drafted by agents, angles unconfirmed):
  ball ok   : found a ball iff the label says there is one (expected warning otherwise)
  ground ok : ramp and ground found, in an accepted upright rotation (`upright_rotation_k`)
  angle     : vs `angle_deg`; while `angle_confirmed` is false those are agent-measured
              from the overlays and the MAE is labelled unconfirmed.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.cv import capture as cap  # noqa: E402
from app.cv.recognize import overlay_png, recognize  # noqa: E402
from app.scene.builder import drawn_angle_deg  # noqa: E402


def eval_synthetic(folder: Path, overlays: Path | None = None) -> dict:
    rows = []
    for path in sorted(folder.glob("*.png")):
        ann = json.loads(path.with_suffix(".json").read_text())
        res = recognize(cap.capture(cap.load_image(path.read_bytes())), path.stem)
        rec = res.recognition
        if overlays:
            (overlays / f"{path.stem}.png").write_bytes(overlay_png(res.image, rec))
        w, h = rec.warped_size
        s = w / ann["width"] if res.rotation_k % 2 == 0 else h / ann["width"]
        short = min(w, h)
        ramp_gt = next(o for o in ann["objects"] if o["class"] == "ramp")
        ball_gt = next(o for o in ann["objects"] if o["class"] == "ball" and o["role"] == "primary")
        row = {"file": path.name, "ball": False, "ramp": False, "ground": False,
               "angle_err": None, "warnings": rec.warnings}
        if rec.ball and res.rotation_k == 0:
            cx, cy = (v * s for v in ball_gt["center_xy"])
            r = ball_gt["radius_px"] * s
            row["ball"] = (math.dist(rec.ball.center, (cx, cy)) < 0.5 * r
                           and abs(rec.ball.radius - r) < 0.35 * r)
        ramp = next((l for l in rec.lines if l.role == "ramp"), None)
        ground = next((l for l in rec.lines if l.role == "ground"), None)
        if ramp and res.rotation_k == 0:
            err = abs(drawn_angle_deg(rec) - ann["physics"]["angle_degrees"]["value"])
            row["angle_err"] = err
            low = [v * s for v in ramp_gt["low_point_xy"]]
            high = [v * s for v in ramp_gt["high_point_xy"]]
            row["ramp"] = (err <= 3 and math.dist(ramp.p2, low) < 0.05 * short
                           and math.dist(ramp.p1, high) < 0.10 * short)
        if ground and res.rotation_k == 0:
            gy = (ground.p1[1] + ground.p2[1]) / 2
            row["ground"] = abs(gy - ann["ground_y_px"] * s) < 0.02 * short
        row["all"] = row["ball"] and row["ramp"] and row["ground"]
        rows.append(row)
    return _summary("synthetic", rows)


def eval_real(folder: Path, overlays: Path | None = None) -> dict:
    labels = json.loads((folder / "labels.json").read_text())
    rows = []
    for item in labels:
        path = folder / item["file"]
        res = recognize(cap.capture(cap.load_image(path.read_bytes())), path.stem)
        rec = res.recognition
        if overlays:
            (overlays / f"{path.stem}.png").write_bytes(overlay_png(res.image, rec))
        has_lines = {l.role for l in rec.lines} == {"ramp", "ground"}
        accepted_k = item.get("upright_rotation_k", [0])
        row = {
            "file": item["file"],
            "ball": (rec.ball is not None) == item["ball"]
                    and (item["ball"] or any("no ball" in w for w in rec.warnings)),
            "ramp": has_lines,
            "ground": has_lines and res.rotation_k in accepted_k,
            "rotation_k": res.rotation_k,
            "angle_err": None,
            "angle": round(drawn_angle_deg(rec), 1) if has_lines else None,
            "angle_approx_unconfirmed": item.get("drawn_angle_approx_deg"),
            "warnings": rec.warnings,
        }
        if isinstance(item.get("angle_deg"), (int, float)) and has_lines:
            row["angle_err"] = abs(row["angle"] - item["angle_deg"])
        row["all"] = row["ball"] and row["ground"]
        rows.append(row)
    summary = _summary("real", rows)
    summary["angles_unconfirmed"] = any(not i.get("angle_confirmed", False) for i in labels)
    return summary


def _summary(name: str, rows: list[dict]) -> dict:
    n = len(rows) or 1
    errs = [r["angle_err"] for r in rows if r["angle_err"] is not None]
    return {
        "set": name,
        "n": len(rows),
        "ball_pct": 100 * sum(r["ball"] for r in rows) / n,
        "ramp_pct": 100 * sum(r["ramp"] for r in rows) / n,
        "ground_pct": 100 * sum(r["ground"] for r in rows) / n,
        "all_pct": 100 * sum(r["all"] for r in rows) / n,
        "angle_mae": sum(errs) / len(errs) if errs else None,
        "rows": rows,
    }


def print_summary(s: dict, verbose: bool) -> None:
    mae = "n/a" if s["angle_mae"] is None else f"{s['angle_mae']:.2f}°"
    if s.get("angles_unconfirmed"):
        mae += " (vs agent-measured angles, unconfirmed)"
    print(f"[{s['set']}] n={s['n']}  ball {s['ball_pct']:.0f}%  ramp {s['ramp_pct']:.0f}%  "
          f"ground {s['ground_pct']:.0f}%  all-objects-correct {s['all_pct']:.0f}%  "
          f"angle MAE {mae}")
    for r in s["rows"]:
        if verbose or not r["all"]:
            extra = {k: v for k, v in r.items() if k not in ("file", "all")}
            print(f"   {'ok ' if r['all'] else 'BAD'} {r['file']}: {extra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["synthetic", "real", "all"], default="all")
    ap.add_argument("--dir", type=Path, help="override the folder for a single --set")
    ap.add_argument("--overlays", type=Path, help="write overlay PNGs here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.overlays:
        args.overlays.mkdir(parents=True, exist_ok=True)
    sets = ["synthetic", "real"] if args.set == "all" else [args.set]
    for name in sets:
        folder = args.dir or ROOT / "fixtures" / name
        if name == "real" and not (folder / "labels.json").exists():
            print(f"[real] no labels.json in {folder}, skipped")
            continue
        fn = eval_synthetic if name == "synthetic" else eval_real
        print_summary(fn(folder, args.overlays), args.verbose)


if __name__ == "__main__":
    main()
