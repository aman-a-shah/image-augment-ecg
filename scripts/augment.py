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
from physiorender.params import AugmentationParams  # noqa: E402
from physiorender.render import ECGRenderer  # noqa: E402


def sample_params(rng: np.random.Generator) -> AugmentationParams:
    """Stand-in sampler (Phase 5 will replace with correlation-aware sampling)."""
    blur = str(rng.choice(["motion", "defocus", "handshake", "none"]))
    return AugmentationParams(
        yellowing_intensity=float(rng.uniform(0.05, 0.3)),
        ink_density_variation=float(rng.uniform(0.0, 0.08)),
        n_wrinkles=int(rng.integers(0, 7)),
        wrinkle_intensity=float(rng.uniform(0.3, 1.0)),
        n_folds=int(rng.integers(0, 3)),
        has_stain=bool(rng.random() < 0.3),
        stain_opacity=float(rng.uniform(0.1, 0.4)),
        has_pen_marks=bool(rng.random() < 0.2),
        tilt_x_deg=float(np.clip(rng.normal(0, 8), -15, 15)),
        tilt_y_deg=float(np.clip(rng.normal(0, 10), -18, 18)),
        rotation_deg=float(np.clip(rng.normal(0, 4), -6, 6)),
        blur_type=blur,
        blur_strength=float(rng.uniform(0.2, 0.8)) if blur != "none" else 0.0,
        lens_k1=float(rng.uniform(0.0, 0.08)),
        has_lens_dirt=bool(rng.random() < 0.2),
        crop_margin=float(rng.uniform(0.02, 0.15)),
        light_angle_deg=float(rng.uniform(0, 360)),
        light_elevation_deg=float(rng.uniform(20, 70)),
        has_specular=bool(rng.random() < 0.35),
        specular_intensity=float(rng.uniform(0.3, 0.9)),
        has_fl_banding=bool(rng.random() < 0.3),
        shadow_width_fraction=float(rng.uniform(0.0, 0.25)),
        jpeg_quality=int(rng.integers(65, 89)),
        noise_iso_equiv=int(rng.integers(100, 1601)),
        colour_temp_delta_k=int(rng.integers(-300, 301)),
    )


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
