"""Post-extraction validation (plan §4.2).

Fail loud on bad data: a malformed extraction must be caught here, not silently
rendered. Returns a structured report so batch generation can log-and-skip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import STANDARD_LEADS
from ..logging_setup import get_logger
from .record import ECGRecord

log = get_logger(__name__)

# Plausible physiological amplitude envelope, in mV (generous bounds).
_MAX_ABS_MV = 12.0
_MIN_SPAN_MV = 0.05  # a real lead spans at least this much (else likely flatline/parse error)
_FLATLINE_MIN_S = 1.0


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)   # disqualifying
    warnings: list[str] = field(default_factory=list)  # suspicious but usable

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_record(record: ECGRecord, *, expected_leads: int = 12) -> ValidationReport:
    """Validate an extracted ECG (plan §4.2 checks):

    - all expected leads present
    - sample-count consistency across leads
    - amplitude within a physiological mV envelope
    - flag flatline segments > 1s (genuine or parse error)
    """
    report = ValidationReport()

    # Lead presence
    present = set(record.leads)
    missing = [name for name in STANDARD_LEADS if name not in present]
    if len(record.leads) < expected_leads:
        report.add_error(
            f"expected >= {expected_leads} leads, got {len(record.leads)}; "
            f"missing standard leads: {missing}"
        )
    elif missing:
        report.add_warning(f"missing standard leads (extras present): {missing}")

    # Sample-count consistency
    lengths = {name: lead.n_samples for name, lead in record.leads.items()}
    if len(set(lengths.values())) > 1:
        report.add_error(f"inconsistent sample counts across leads: {lengths}")

    # Sample-rate consistency
    rates = {lead.sample_rate_hz for lead in record.leads.values()}
    if len(rates) > 1:
        report.add_error(f"inconsistent sample rates across leads: {rates}")

    # Per-lead amplitude / flatline checks
    for name, lead in record.leads.items():
        sig = lead.signal_mv
        if sig.size == 0:
            report.add_error(f"lead {name!r} is empty")
            continue
        peak = float(np.nanmax(np.abs(sig)))
        if peak > _MAX_ABS_MV:
            report.add_warning(
                f"lead {name!r} peak {peak:.2f} mV exceeds {_MAX_ABS_MV} mV "
                f"(check resolution/scaling)"
            )
        span = float(np.nanmax(sig) - np.nanmin(sig))
        if span < _MIN_SPAN_MV:
            report.add_warning(f"lead {name!r} span {span:.4f} mV — possible flatline")
        else:
            flat = _longest_flatline_seconds(sig, lead.sample_rate_hz)
            if flat > _FLATLINE_MIN_S:
                report.add_warning(
                    f"lead {name!r} has a {flat:.1f}s flatline segment"
                )

    level = log.error if not report.ok else (log.warning if report.warnings else log.info)
    level("validation %s: %d errors, %d warnings",
          "FAILED" if not report.ok else "ok",
          len(report.errors), len(report.warnings))
    return report


def _longest_flatline_seconds(sig: np.ndarray, fs: int) -> float:
    """Length (s) of the longest run where the signal is ~constant."""
    if sig.size < 2:
        return 0.0
    is_flat = np.abs(np.diff(sig)) < 1e-4
    best = run = 0
    for flat in is_flat:
        run = run + 1 if flat else 0
        best = max(best, run)
    return best / fs
