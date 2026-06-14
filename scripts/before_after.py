#!/usr/bin/env python3
"""Side-by-side before/after for the augmentation pipeline.

Renders the clean ECG (before) and the augmented smartphone photo (after) and
saves them stitched together for easy comparison.

    python scripts/before_after.py data/dummy-ecg.xml --n 3
    python scripts/before_after.py data/dummy-ecg.xml --seed 5 --out artifacts/compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from physiorender.degrade import DegradationEngine  # noqa: E402
from physiorender.ingest import load_ecg  # noqa: E402
from physiorender.render import ECGRenderer  # noqa: E402
from physiorender.sampling import ParameterSampler  # noqa: E402


def _label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 260, 46], fill=(0, 0, 0))
    d.text((12, 12), text, fill=(255, 255, 255))
    return img


def _side_by_side(before: Image.Image, after: Image.Image) -> Image.Image:
    h = min(before.height, after.height, 900)
    bw = int(before.width * h / before.height)
    aw = int(after.width * h / after.height)
    b = _label(before.resize((bw, h)).convert("RGB"), "BEFORE  (clean render)")
    a = _label(after.resize((aw, h)).convert("RGB"), "AFTER  (augmented photo)")
    gap = 16
    canvas = Image.new("RGB", (bw + aw + gap, h), (255, 255, 255))
    canvas.paste(b, (0, 0))
    canvas.paste(a, (bw + gap, 0))
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description="Before/after augmentation comparison.")
    ap.add_argument("path")
    ap.add_argument("--n", type=int, default=1, help="Number of after-variants.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="artifacts/compare")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    record = load_ecg(args.path, validate=True)
    render = ECGRenderer(dpi=args.dpi).render(record)
    engine = DegradationEngine(dpi=args.dpi)
    sampler = ParameterSampler()
    stem = Path(args.path).stem

    render.image.save(out / f"{stem}_before.png")
    for i in range(args.n):
        seed = args.seed + i
        params = sampler.sample(np.random.default_rng(seed))
        result = engine.augment(render.image, params, seed=seed,
                                lead_bboxes=render.lead_bboxes)
        result.image.save(out / f"{stem}_after_{seed:03d}.jpg", quality=92)
        combo = _side_by_side(render.image, result.image)
        combo_path = out / f"{stem}_compare_{seed:03d}.png"
        combo.save(combo_path)
        print(f"[{i+1}/{args.n}] {combo_path.name}   effects: {', '.join(result.applied)}")

    print(f"\nOpen the *_compare_*.png files in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
