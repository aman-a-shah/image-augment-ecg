#!/usr/bin/env python3
"""Render a clean ECG printout from an ECG file (Phase 2 gate).

    python scripts/render_ecg.py data/dummy-ecg.xml
    python scripts/render_ecg.py data/dummy-ecg.xml --speed 25 --gain 10 --bbox

Writes <stem>_clean.png (and <stem>_clean_bbox.png with --bbox) to artifacts/render/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import ImageDraw  # noqa: E402

from physiorender.ingest import load_ecg  # noqa: E402
from physiorender.render import ECGRenderer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Render a clean ECG printout.")
    ap.add_argument("path", help="Path to ECG file (XML or WFDB record).")
    ap.add_argument("--speed", type=int, default=25, help="Paper speed mm/s (25/50).")
    ap.add_argument("--gain", type=int, default=10, help="Gain mm/mV (10/5).")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-rhythm", action="store_true", help="Omit rhythm strip.")
    ap.add_argument("--bbox", action="store_true", help="Also write a bbox overlay.")
    ap.add_argument("--out", default="artifacts/render")
    args = ap.parse_args()

    record = load_ecg(args.path, validate=True)
    renderer = ECGRenderer(dpi=args.dpi, paper_speed_mm_s=args.speed, gain_mm_mv=args.gain)

    from physiorender.render import build_standard_12lead
    layout = build_standard_12lead(rhythm=not args.no_rhythm)
    result = renderer.render(record, layout)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.path).stem
    clean_png = out / f"{stem}_clean.png"
    result.save(clean_png)

    print(f"Rendered        : {clean_png}")
    print(f"Image size      : {result.image.size[0]}x{result.image.size[1]} px @ {result.dpi} DPI")
    print(f"Template        : {result.template}")
    print(f"Speed / gain    : {result.paper_speed_mm_s} mm/s, {result.gain_mm_mv} mm/mV")
    print(f"Lead bboxes     : {len(result.lead_bboxes)}")

    if args.bbox:
        overlay = result.image.copy()
        d = ImageDraw.Draw(overlay)
        for key, (x1, y1, x2, y2) in result.lead_bboxes.items():
            d.rectangle([x1, y1, x2, y2], outline=(0, 120, 220), width=2)
            d.text((x1 + 3, y1 + 3), key, fill=(0, 120, 220))
        bbox_png = out / f"{stem}_clean_bbox.png"
        overlay.save(bbox_png)
        print(f"Bbox overlay    : {bbox_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
