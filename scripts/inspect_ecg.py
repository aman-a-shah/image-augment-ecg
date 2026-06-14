#!/usr/bin/env python3
"""Inspect an ECG file: detect format, extract, validate, and plot an overlay.

Phase 1 gate tool (plan §0). Usage:

    python scripts/inspect_ecg.py data/dummy-ecg.xml
    python scripts/inspect_ecg.py data/dummy-ecg.xml --no-plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (no install needed).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physiorender.ingest import detect_ecg_xml_format, load_ecg, validate_record  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect an ECG file.")
    ap.add_argument("path", help="Path to ECG file (XML or WFDB record).")
    ap.add_argument("--no-plot", action="store_true", help="Skip the overlay plot.")
    ap.add_argument("--out", default="artifacts/inspect", help="Output dir for the plot.")
    args = ap.parse_args()

    fmt = detect_ecg_xml_format(args.path)
    print(f"Detected format : {fmt.value}")

    record = load_ecg(args.path, validate=False)
    print(f"Source format   : {record.source_format}")
    print(f"Leads ({record.n_leads}) : {', '.join(record.lead_order)}")
    print(f"Sample rate     : {record.sample_rate_hz} Hz")
    print(f"Duration        : {record.duration_s:.2f} s")
    if record.measurements:
        print(f"Measurements    : {record.measurements}")

    report = validate_record(record)
    print(f"\nValidation      : {'OK' if report.ok else 'FAILED'}")
    for e in report.errors:
        print(f"  ERROR   {e}")
    for w in report.warnings:
        print(f"  WARNING {w}")

    if not args.no_plot:
        _plot(record, args.out)

    return 0 if report.ok else 1


def _plot(record, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = record.lead_order
    fig, axes = plt.subplots(len(order), 1, figsize=(12, 1.1 * len(order)), sharex=True)
    if len(order) == 1:
        axes = [axes]
    for ax, name in zip(axes, order):
        lead = record.leads[name]
        t = [i / lead.sample_rate_hz for i in range(lead.n_samples)]
        ax.plot(t, lead.signal_mv, lw=0.5, color="#1A1A1A")
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
        ax.margins(x=0)
        ax.grid(True, color="#F0C8C8", lw=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{Path(record.source_path).name} — {record.source_format}")
    fig.tight_layout()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{Path(record.source_path).stem}_overlay.png"
    fig.savefig(png, dpi=120)
    print(f"\nOverlay plot    : {png}")


if __name__ == "__main__":
    raise SystemExit(main())
