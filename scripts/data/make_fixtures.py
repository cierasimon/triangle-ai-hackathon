"""Copy N ball_plain images + their expected JSON into fixtures/synthetic/.

Source is a generated dataset folder (images/ + annotations.jsonl), e.g. the release zip
unpacked by fetch_dataset.sh, or a fresh run of paperphysics_generate_v3.py. Images are
deterministic per index (seed + i*10103), so a small --count run gives the same scenes as
the 3,000-image release.

    python scripts/data/make_fixtures.py --source data/generated/paperphysics_v3 --count 20
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--dest", type=Path, default=ROOT / "fixtures" / "synthetic")
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    copied = []
    with (args.source / "annotations.jsonl").open(encoding="utf-8") as f:
        for raw in f:
            record = json.loads(raw)
            if (record["primary_type"], record["scene_type"]) != ("ball", "plain"):
                continue
            image = args.source / record["image"]
            name = Path(record["image"]).name
            shutil.copyfile(image, args.dest / name)
            (args.dest / name.replace(".png", ".json")).write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            copied.append(name)
            if len(copied) == args.count:
                break
    print(f"copied {len(copied)} ball_plain fixtures to {args.dest}")


if __name__ == "__main__":
    main()
