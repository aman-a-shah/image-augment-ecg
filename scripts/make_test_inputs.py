#!/usr/bin/env python3
"""Fill an input folder with fake ECG files (XMLs + images) for testing batch convert.

Derives a few varied 12-lead records from the bundled sample (time-shifted /
amplitude-scaled), writes some as GE-CardioSoft-style XML and some as clean
rendered PNG/JPG "scans".

    python scripts/make_test_inputs.py --out input_ecgs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physiorender.config import STANDARD_LEADS  # noqa: E402
from physiorender.ingest import load_ecg  # noqa: E402
from physiorender.ingest.record import ECGRecord, LeadSignal  # noqa: E402
from physiorender.render import ECGRenderer  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "dummy-ecg.xml"
_UV_PER_LSB = 5.0


def write_cardiology_xml(record: ECGRecord, path: Path) -> None:
    """Serialize a record as a minimal GE CardioSoft <CardiologyXML> file."""
    leads_xml = []
    n = next(iter(record.leads.values())).n_samples
    for name in record.lead_order:
        counts = np.round(record.leads[name].signal_mv * 1000.0 / _UV_PER_LSB).astype(int)
        leads_xml.append(
            f'      <WaveformData lead="{name}">' + ",".join(map(str, counts.tolist()))
            + "</WaveformData>"
        )
    xml = (
        '<?xml version="1.0" encoding="ISO8859-1" ?>\n'
        "<CardiologyXML>\n"
        "  <ObservationType>RestECG</ObservationType>\n"
        f"  <LeadOrder>{', '.join(record.lead_order)}</LeadOrder>\n"
        "  <StripData>\n"
        "    <NumberOfLeads>12</NumberOfLeads>\n"
        f"    <SampleRate units=\"Hz\">{record.sample_rate_hz}</SampleRate>\n"
        f"    <ChannelSampleCountTotal>{n}</ChannelSampleCountTotal>\n"
        f"    <Resolution units=\"uVperLsb\">{int(_UV_PER_LSB)}</Resolution>\n"
        + "\n".join(leads_xml) + "\n"
        "  </StripData>\n"
        "</CardiologyXML>\n"
    )
    path.write_text(xml, encoding="ISO8859-1")


def _variant(base: ECGRecord, *, roll: int, scale: float) -> ECGRecord:
    leads = {}
    for name in base.lead_order:
        sig = np.roll(base.leads[name].signal_mv, roll) * scale
        leads[name] = LeadSignal(name=name, signal_mv=sig.astype(np.float32),
                                 sample_rate_hz=base.sample_rate_hz)
    return ECGRecord(leads=leads, sample_rate_hz=base.sample_rate_hz,
                     source_path="synthetic", source_format="CARDIOLOGY_XML",
                     lead_order=list(base.lead_order))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="input_ecgs")
    args = ap.parse_args()
    if not SAMPLE.exists():
        print(f"need sample at {SAMPLE}")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = load_ecg(SAMPLE, validate=False)

    variants = {
        "patient_A": _variant(base, roll=0, scale=1.0),
        "patient_B": _variant(base, roll=700, scale=1.2),
        "patient_C": _variant(base, roll=1800, scale=0.8),
    }

    # Three as XML.
    for name, rec in variants.items():
        write_cardiology_xml(rec, out / f"{name}.xml")
        print(f"wrote {name}.xml")

    # Two as clean rendered images (simulating scanned ECGs).
    for name, scan_name in (("patient_A", "scan_001.png"), ("patient_B", "scan_002.jpg")):
        img = ECGRenderer(dpi=200).render(variants[name]).image
        img.save(out / scan_name)
        print(f"wrote {scan_name}")

    print(f"\nTest inputs in {out}/  ({len(list(out.iterdir()))} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
