#!/usr/bin/env bash
# Download the PaperPhysics V3 synthetic dataset (3,000 images, ~31 MB) into data/generated/.
# No-download alternative:
#   python scripts/data/paperphysics_generate_v3.py --count 3000 --output data/generated/paperphysics_v3
set -euo pipefail

URL="https://github.com/cierasimon/triangle-ai-hackathon/releases/download/dataset-v3/paperphysics_v3_release.zip"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/data/generated"
ZIP="$DEST/paperphysics_v3_release.zip"

mkdir -p "$DEST"
if [ ! -f "$ZIP" ]; then
  echo "downloading $URL"
  curl -fL --retry 3 -o "$ZIP.part" "$URL"
  mv "$ZIP.part" "$ZIP"
fi
unzip -oq "$ZIP" -d "$DEST"
echo "dataset unpacked into $DEST:"
find "$DEST" -name annotations.jsonl -print
