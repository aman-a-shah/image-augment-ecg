"""In-memory representation of an ingested ECG (plan §4).

This is the hand-off between ingestion (Phase 1) and the renderer (Phase 2):
a clean ``[leads × samples]`` set of waveforms in millivolts, plus the metadata
the renderer needs (sample rate, lead order, optional interval annotations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LeadSignal:
    """One lead's waveform, in millivolts."""

    name: str
    signal_mv: np.ndarray          # 1-D float array, millivolts
    sample_rate_hz: int

    @property
    def n_samples(self) -> int:
        return int(self.signal_mv.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate_hz


@dataclass
class ECGRecord:
    """A full multi-lead ECG ready for rendering.

    ``leads`` is keyed by lead name (I, II, ... V6). ``lead_order`` preserves the
    acquisition/display order; all leads share ``sample_rate_hz``.
    """

    leads: dict[str, LeadSignal]
    sample_rate_hz: int
    source_path: str
    source_format: str
    lead_order: list[str] = field(default_factory=list)
    # Optional fiducial annotations (sample indices) and free-form extras.
    measurements: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.lead_order:
            self.lead_order = list(self.leads.keys())

    @property
    def n_leads(self) -> int:
        return len(self.leads)

    @property
    def duration_s(self) -> float:
        if not self.leads:
            return 0.0
        return max(lead.duration_s for lead in self.leads.values())

    def matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return (signals, lead_order) as a [n_leads × n_samples] array.

        Leads are zero-padded to the longest length so the result is rectangular.
        """
        order = self.lead_order
        n = max((self.leads[name].n_samples for name in order), default=0)
        out = np.zeros((len(order), n), dtype=np.float32)
        for i, name in enumerate(order):
            sig = self.leads[name].signal_mv
            out[i, : sig.shape[0]] = sig
        return out, list(order)
