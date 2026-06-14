#!/usr/bin/env python3
"""Turn ANY existing ECG image into a realistic smartphone photo.

Unlike scripts/augment.py (which starts from XML/WFDB signal and produces full
ground-truth metadata), this takes a clean ECG *image* you already have and runs
only the Physical Degradation Engine on it. No signal/bbox metadata is produced
(we don't know the underlying waveform), but the visual smartphone-photo effects
all apply.

    python scripts/augment_image.py my_ecg.png --n 3
    python scripts/augment_image.py my_ecg.png --dpi 150 --compare

--dpi controls the physical scale of artifacts (wrinkle width, blur, grid bleed).
If omitted, it's estimated from the image width assuming a ~250mm-wide printout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from physiorender.degrade import DegradationEngine  # noqa: E402
from physiorender.sampling import ParameterSampler  # noqa: E402

_PAGE_W_MM = 250.0  # standard 10s @ 25mm/s printout width


def _estimate_dpi(width_px: int) -> int:
    # dpi = px / inches; inches = page_mm / 25.4
    return max(72, int(round(width_px / (_PAGE_W_MM / 25.4))))


def _side_by_side(before: Image.Image, after: Image.Image) -> Image.Image:
    h = min(before.height, after.height, 900)
    bw, aw = int(before.width * h / before.height), int(after.width * h / after.height)
    canvas = Image.new("RGB", (bw + aw + 16, h), (255, 255, 255))
    canvas.paste(before.resize((bw, h)).convert("RGB"), (0, 0))
    canvas.paste(after.resize((aw, h)).convert("RGB"), (bw + 16, 0))
    d = ImageDraw.Draw(canvas)
    for x, t in ((0, "BEFORE (input image)"), (bw + 16, "AFTER (smartphone photo)")):
        d.rectangle([x, 0, x + 280, 44], fill=(0, 0, 0))
        d.text((x + 12, 12), t, fill=(255, 255, 255))
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description="Augment an existing ECG image.")
    ap.add_argument("image", help="Path to a clean ECG image (PNG/JPG).")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=None, help="Override artifact scale.")
    ap.add_argument("--compare", action="store_true", help="Also write before/after.")
    ap.add_argument("--out", default="artifacts/image_aug")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clean = Image.open(args.image).convert("RGB")
    dpi = args.dpi or _estimate_dpi(clean.width)
    print(f"input {clean.width}x{clean.height}, using dpi={dpi} for artifact scale")

    engine = DegradationEngine(dpi=dpi)
    sampler = ParameterSampler()
    stem = Path(args.image).stem

    for i in range(args.n):
        seed = args.seed + i
        params = sampler.sample(np.random.default_rng(seed))
        # No lead_bboxes: this is an arbitrary image, so no bbox/signal metadata.
        result = engine.augment(clean, params, seed=seed, lead_bboxes=None)
        jpg = out / f"{stem}_photo_{seed:03d}.jpg"
        result.image.save(jpg, quality=92)
        print(f"[{i+1}/{args.n}] {jpg.name}   effects: {', '.join(result.applied)}")
        if args.compare:
            _side_by_side(clean, result.image).save(out / f"{stem}_compare_{seed:03d}.png")

    print(f"\nOutputs in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
