"""Inspect local fixtures with OpenCV or hybrid recognition without claiming accuracy.

Examples:
  python scripts/inspect_recognition.py
  python scripts/inspect_recognition.py --mode both --allow-cloud

Hybrid requests are opt-in, transmit images to Google Cloud, and may incur charges.
No datasets, labels, splits or checkpoints are modified. Reports are Git-ignored.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

from app.cv.capture import capture, load_image  # noqa: E402
from app.cv.recognize import recognize, overlay_png  # noqa: E402
from app.cv.hybrid import assist  # noqa: E402


def inspect(path: Path, mode: str) -> tuple[dict, bytes]:
    start = time.perf_counter()
    result = recognize(capture(load_image(path.read_bytes())), path.stem)
    if mode == 'hybrid':
        result = assist(result)
    rec = result.recognition
    return {
        'file': path.name, 'mode': mode, 'seconds': round(time.perf_counter()-start,3),
        'ball_count': len(rec.balls), 'block_count': len(rec.blocks),
        'gemini_status': rec.gemini_status, 'recognition': rec.model_dump(),
        'limitations': 'Diagnostic output only; no verified complete real-photo ground truth or accuracy claim.',
    }, overlay_png(result.image, rec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, default=ROOT/'fixtures/real')
    parser.add_argument('--output', type=Path, default=ROOT/'data/generated/recognition-reports')
    parser.add_argument('--mode', choices=['opencv','hybrid','both'], default='opencv')
    parser.add_argument('--allow-cloud', action='store_true', help='Allow image transmission and possible API charges')
    args = parser.parse_args()
    if args.mode != 'opencv' and not args.allow_cloud:
        parser.error('Hybrid mode requires --allow-cloud because it sends images to Google Cloud.')
    # Keep diagnostics out of source-controlled paths, regardless of caller working directory.
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/'data/generated').resolve()):
        parser.error('--output must be inside the Git-ignored data/generated directory')
    if not args.dir.is_dir():
        parser.error('--dir must be an existing image directory')
    paths = sorted(p for p in args.dir.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'})
    if not paths:
        parser.error('No supported images found in --dir')
    output.mkdir(parents=True,exist_ok=True)
    reports = []
    for path in paths:
        for mode in (['opencv','hybrid'] if args.mode == 'both' else [args.mode]):
            report, overlay = inspect(path,mode)
            reports.append(report)
            (output/f'{path.name}-{mode}.png').write_bytes(overlay)
            print(f"{path.name}: {mode}, {report['ball_count']} balls, {report['block_count']} blocks, "
                  f"Gemini={report['gemini_status']}, {report['seconds']} s")
    (output/'report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(f'Diagnostics saved to {output}; no accuracy score is calculated.')


if __name__ == '__main__':
    main()
