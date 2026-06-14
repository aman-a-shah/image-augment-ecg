"""WFDB reader — unlocks PTB-XL / CPSC / Georgia datasets (plan §10).

WFDB is the de-facto PhysioNet format; supporting it makes the whole pipeline
useful on large public corpora without needing XML at all. ``wfdb`` is imported
lazily so the rest of ingestion works even if it isn't installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..logging_setup import get_logger
from .base import SignalExtractor
from .record import ECGRecord, LeadSignal

log = get_logger(__name__)

# Normalize common WFDB lead spellings to our canonical names.
_LEAD_ALIASES = {
    "MLI": "I", "MLII": "II", "MLIII": "III",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
}


class WFDBExtractor(SignalExtractor):
    format_name = "WFDB"

    def extract(self, filepath: str | Path) -> ECGRecord:
        try:
            import wfdb
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "WFDB support requires the 'wfdb' package: pip install wfdb"
            ) from exc

        path = Path(filepath)
        record_name = str(path.with_suffix(""))  # wfdb wants the path w/o extension
        rec = wfdb.rdrecord(record_name)

        # rec.p_signal is [n_samples × n_leads] already in physical units (mV for ECG).
        signals = np.asarray(rec.p_signal, dtype=np.float32)
        sample_rate = int(rec.fs)
        raw_names = list(rec.sig_name)

        leads: dict[str, LeadSignal] = {}
        lead_order: list[str] = []
        for i, raw in enumerate(raw_names):
            name = self._canonical(raw)
            col = signals[:, i]
            # Units are usually mV; convert if the header says otherwise.
            units = (rec.units[i] if rec.units else "mV").lower()
            if units in ("uv", "µv"):
                col = col / 1000.0
            elif units in ("v",):
                col = col * 1000.0
            leads[name] = LeadSignal(name=name, signal_mv=col,
                                     sample_rate_hz=sample_rate)
            lead_order.append(name)

        log.info("WFDB: %d leads, %d Hz, %.2fs from %s",
                 len(leads), sample_rate,
                 next(iter(leads.values())).duration_s, path.name)

        return ECGRecord(
            leads=leads,
            sample_rate_hz=sample_rate,
            source_path=str(path),
            source_format=self.format_name,
            lead_order=lead_order,
            meta={"comments": getattr(rec, "comments", [])},
        )

    @staticmethod
    def _canonical(name: str) -> str:
        key = name.strip().upper()
        if key in _LEAD_ALIASES:
            return _LEAD_ALIASES[key]
        # Vn leads: normalize V1..V6 casing; pass others through.
        if key.startswith("V") and key[1:].isdigit():
            return "V" + key[1:]
        return name.strip()
