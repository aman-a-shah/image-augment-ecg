#!/usr/bin/env python3
"""Generate a training dataset of augmented ECG photos (plan §10).

    python scripts/generate_dataset.py data/*.xml --n 20 --out artifacts/train
    python scripts/generate_dataset.py data/dummy-ecg.xml --n 50 --no-signals

Writes images + metadata + warp fields + manifest.jsonl, then runs an integrity
pass over the batch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physiorender.dataset_gen import generate_dataset, validate_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate an augmented ECG dataset.")
    ap.add_argument("sources", nargs="+", help="Source ECG files (XML or WFDB).")
    ap.add_argument("--n", type=int, default=10, help="Variants per source.")
    ap.add_argument("--out", default="artifacts/train")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-warp", action="store_true", help="Skip warp-field export.")
    ap.add_argument("--no-signals", action="store_true",
                    help="Drop waveform arrays from metadata (smaller JSON).")
    args = ap.parse_args()

    report = generate_dataset(
        args.sources, out_dir=args.out, n_per_source=args.n, dpi=args.dpi,
        base_seed=args.seed, save_warp=not args.no_warp,
        write_signals=not args.no_signals,
    )
    print(f"\nGenerated {report.n_images} images from {report.n_sources} sources")
    print(f"  valid={report.n_valid}  failed={report.n_failed}")

    integrity = validate_dataset(args.out)
    print(f"Integrity pass: {integrity.n_valid}/{integrity.n_images} valid")
    for f in integrity.failures[:10]:
        print(f"  FAIL {f}")
    return 0 if integrity.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
