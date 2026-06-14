#!/usr/bin/env python3
"""Dump each PDE Layer 1-2 augmentation in isolation + a full composite.

Phase 3 gate tool (plan §6 gate): "each augmentation toggleable in isolation and
looks physically plausible." Writes PNGs to artifacts/visual_tests/.

    python scripts/visual_test_degrade.py data/dummy-ecg.xml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physiorender import config  # noqa: E402
from physiorender.degrade import DegradationEngine  # noqa: E402
from physiorender.degrade import imageutil as iu  # noqa: E402
from physiorender.degrade import layer1_paper as l1  # noqa: E402
from physiorender.degrade import layer2_handling as l2  # noqa: E402
from physiorender.degrade.light import LightSource  # noqa: E402
from physiorender.degrade.warp import DisplacementField  # noqa: E402
from physiorender.ingest import load_ecg  # noqa: E402
from physiorender.params import AugmentationParams  # noqa: E402
from physiorender.render import ECGRenderer  # noqa: E402

OUT = Path("artifacts/visual_tests")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    record = load_ecg(args.path, validate=False)
    clean = ECGRenderer(dpi=300).render(record).image
    clean.save(OUT / "00_clean.png")

    ppm = config.mm_to_px(1.0, 300)
    light = LightSource(angle_deg=35.0, elevation_deg=40.0)

    def isolated(name: str, fn):
        rng = np.random.default_rng(args.seed)
        arr = iu.pil_to_float(clean)
        disp = DisplacementField.zeros(*arr.shape[:2])
        arr = fn(arr, rng, disp)
        if not disp.is_identity():
            arr = disp.apply(arr)
        iu.float_to_pil(arr).save(OUT / name)
        print(f"  wrote {name}")

    print("Isolated augmentations:")
    isolated("01_yellowing.png",
             lambda a, r, d: l1.apply_yellowing(a, r, 0.3))
    isolated("02_ink_density.png",
             lambda a, r, d: l1.apply_ink_density(a, r, 0.08))
    isolated("03_ink_skip.png",
             lambda a, r, d: l1.apply_ink_skip(a, r, 30))
    isolated("04_wrinkles.png",
             lambda a, r, d: l2.add_wrinkles(a, d, r, n=6, intensity=1.0,
                                             light=light, ppm=ppm))
    isolated("05_folds.png",
             lambda a, r, d: l2.add_folds(a, d, r, n=2, intensity=1.0,
                                          light=light, ppm=ppm))
    isolated("06_edge_curl.png",
             lambda a, r, d: l2.apply_edge_curl(a, d, r, strength=1.0, light=light))
    isolated("07_stain.png",
             lambda a, r, d: l2.add_stain(a, r, opacity=0.4))
    isolated("08_pen_marks.png",
             lambda a, r, d: l2.add_pen_marks(a, r, n=3))
    isolated("09_fingerprint.png",
             lambda a, r, d: l2.add_fingerprint(a, r, opacity=0.2))

    # Full composite via the engine.
    params = AugmentationParams(
        yellowing_intensity=0.22, ink_density_variation=0.06,
        n_wrinkles=5, wrinkle_intensity=0.8, n_folds=1,
        has_stain=True, stain_opacity=0.3, has_pen_marks=True,
        light_angle_deg=35.0, light_elevation_deg=40.0,
    )
    result = DegradationEngine(dpi=300).apply(clean, params, seed=args.seed)
    result.image.save(OUT / "10_composite.png")
    print(f"\nComposite applied: {result.applied}")

    # Invertibility check on the composite warp.
    inv = result.inverse_warp()
    from physiorender.degrade.warp import apply_displacement
    arr = iu.pil_to_float(clean)
    warped = result.displacement.apply(arr)
    recovered = apply_displacement(warped, inv[..., 0], inv[..., 1])
    m = _interior(np.abs(recovered - arr))
    print(f"Warp round-trip interior MAE: {m:.5f} (lower is better)")
    iu.float_to_pil(np.abs(recovered - arr) * 5).save(OUT / "11_warp_residual_x5.png")

    print(f"\nAll outputs in {OUT}/")
    return 0


def _interior(arr: np.ndarray, margin: int = 40) -> float:
    return float(arr[margin:-margin, margin:-margin].mean())


if __name__ == "__main__":
    raise SystemExit(main())
