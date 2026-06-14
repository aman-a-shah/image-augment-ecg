#!/usr/bin/env python3
"""Train PhotoTrace (Stages 1-3) on a generated dataset (plan §9, §12 Phase 5-6).

    # 1) make data
    python scripts/generate_dataset.py data/dummy-ecg.xml --n 200 --out artifacts/train
    # 2) train geometry + digitizer, save checkpoints
    python scripts/train_phototrace.py --data artifacts/train --out artifacts/ckpt

Reports val metrics for each stage. Defaults are CPU-friendly; scale epochs/data
up with a GPU for real performance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from physiorender.ingest import load_ecg  # noqa: E402
from physiorender.render import ECGRenderer  # noqa: E402

from phototrace.digitize_data import StripDataset, build_strip_examples  # noqa: E402
from phototrace.models import CornerRegressor, LeadDetector  # noqa: E402
from phototrace.train_digitize import train_digitizer  # noqa: E402
from phototrace.train_geometry import split_dataset, train_regressor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Train PhotoTrace stages.")
    ap.add_argument("--data", required=True, help="Generated dataset dir (manifest.jsonl).")
    ap.add_argument("--out", default="artifacts/ckpt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--digitizer-source", default=None,
                    help="ECG file for Stage-3 strip training (defaults to skipping).")
    ap.add_argument("--digitizer-dpi", type=int, default=150)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- Stage 1 & 2 ---
    train_ds, val_ds = split_dataset(args.data, image_size=args.image_size, seed=0)
    print(f"geometry: {len(train_ds)} train / {len(val_ds)} val")

    s1 = train_regressor(CornerRegressor(), train_ds, val_ds, target_key="corners",
                         epochs=args.epochs, seed=1)
    print(f"Stage1 corners: val_mae={s1.val_mae:.4f} baseline={s1.baseline_mae:.4f} "
          f"beats={s1.beats_baseline}")
    torch.save(s1.model.state_dict(), out / "stage1_corners.pt")

    s2 = train_regressor(LeadDetector(), train_ds, val_ds, target_key="boxes",
                         epochs=args.epochs, seed=2)
    print(f"Stage2 boxes:   val_mae={s2.val_mae:.4f} baseline={s2.baseline_mae:.4f} "
          f"beats={s2.beats_baseline}")
    torch.save(s2.model.state_dict(), out / "stage2_boxes.pt")

    # --- Stage 3 (needs strip examples from a source ECG) ---
    if args.digitizer_source:
        rec = load_ecg(args.digitizer_source, validate=False)
        render = ECGRenderer(dpi=args.digitizer_dpi).render(rec)
        examples = build_strip_examples(rec, render)
        tr = StripDataset(examples, n_variants=10, seed=0)
        va = StripDataset(examples, n_variants=2, seed=7, augment=False)
        s3 = train_digitizer(tr, va, epochs=args.epochs, seed=0)
        print(f"Stage3 digitize: corr={s3.val_corr:.3f} baseline={s3.baseline_corr:.3f} "
              f"dtw={s3.val_dtw:.4f} peakF1={s3.val_peak_f1:.3f}")
        torch.save(s3.model.state_dict(), out / "stage3_digitizer.pt")

    print(f"\ncheckpoints in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
