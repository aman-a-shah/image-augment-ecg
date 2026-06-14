#!/usr/bin/env python3
"""Full pipeline: ECG file -> augmented smartphone-photo JPEG + metadata.json + warp.npy.

Phase 4 integration / Phase 7 CLI groundwork (plan §8, §12). Usage:

    python scripts/augment.py data/dummy-ecg.xml --n 3 --out artifacts/dataset
    python scripts/augment.py data/dummy-ecg.xml --bbox        # also dump a bbox overlay

A lightweight per-image parameter sampler stands in until the Phase 5
ParameterSampler lands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import ImageDraw  # noqa: E402

from physiorender.assemble import build_metadata  # noqa: E402
from physiorender.degrade import DegradationEngine  # noqa: E402
from physiorender.ingest import load_ecg  # noqa: E402
from physiorender.render import ECGRenderer  # noqa: E402
from physiorender.sampling import ParameterSampler  # noqa: E402

_SAMPLER = ParameterSampler()


def sample_params(rng: np.random.Generator):
    return _SAMPLER.sample(rng)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate augmented ECG photos + metadata.")
    ap.add_argument("path", help="Path to ECG file (XML or WFDB record).")
    ap.add_argument("--n", type=int, default=1, help="Number of augmented variants.")
    ap.add_argument("--seed", type=int, default=0, help="Base random seed.")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--speed", type=int, default=25)
    ap.add_argument("--gain", type=int, default=10)
    ap.add_argument("--out", default="artifacts/dataset")
    ap.add_argument("--bbox", action="store_true", help="Also write a bbox overlay PNG.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    record = load_ecg(args.path, validate=True)
    render = ECGRenderer(dpi=args.dpi, paper_speed_mm_s=args.speed,
                         gain_mm_mv=args.gain).render(record)
    engine = DegradationEngine(dpi=args.dpi)
    stem = Path(args.path).stem

    for i in range(args.n):
        seed = args.seed + i
        rng = np.random.default_rng(seed)
        params = sample_params(rng)
        result = engine.augment(render.image, params, seed=seed,
                                lead_bboxes=render.lead_bboxes)

        image_id = f"{stem}_aug_{seed:05d}"
        jpg = out / f"{image_id}.jpg"
        warp_name = f"warp_{image_id}.npy"
        result.image.save(jpg, quality=92)
        np.save(out / warp_name, result.warp_field_inverse())

        meta = build_metadata(record, render, result,
                              image_id=image_id, warp_field_filename=warp_name)
        report_ok = meta.is_valid()
        meta.save(str(out / f"{image_id}.json"))

        print(f"[{i+1}/{args.n}] {jpg.name}  "
              f"({result.image.size[0]}x{result.image.size[1]})  "
              f"meta_valid={report_ok}  effects={len(result.applied)}")

        if args.bbox:
            overlay = result.image.copy()
            d = ImageDraw.Draw(overlay)
            for key, (x1, y1, x2, y2) in result.lead_bboxes.items():
                d.rectangle([x1, y1, x2, y2], outline=(0, 200, 90), width=3)
            overlay.save(out / f"{image_id}_bbox.png")

    print(f"\nOutputs in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
