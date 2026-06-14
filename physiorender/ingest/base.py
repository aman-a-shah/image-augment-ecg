"""SignalExtractor ABC — one concrete subclass per format (plan §4.2)."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .record import ECGRecord


class SignalExtractor(ABC):
    """Extract a normalized :class:`ECGRecord` from a source file.

    Subclasses implement :meth:`extract`. Shared decoding helpers live here so
    every format converts raw counts to millivolts the same way.
    """

    #: Human-readable format label, stored on the produced record.
    format_name: str = "UNKNOWN"

    @abstractmethod
    def extract(self, filepath: str | Path) -> ECGRecord:
        ...

    # ------------------------------------------------------------------ #
    # Shared decoding helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def counts_to_mv(counts: np.ndarray, uv_per_lsb: float) -> np.ndarray:
        """Convert raw integer counts to millivolts.

        ``uv_per_lsb`` is the device resolution (microvolts per least-significant
        bit). mV = counts * uV_per_LSB / 1000.
        """
        return counts.astype(np.float32) * (uv_per_lsb / 1000.0)

    @staticmethod
    def parse_csv_ints(text: str) -> np.ndarray:
        """Parse comma/space-separated integers (CardioSoft-style waveforms)."""
        cleaned = text.replace("\n", " ").replace("\t", " ").strip()
        if not cleaned:
            return np.zeros(0, dtype=np.int64)
        parts = (p for p in cleaned.replace(",", " ").split() if p)
        return np.fromiter((int(float(p)) for p in parts), dtype=np.int64)

    @staticmethod
    def decode_base64_int16(text: str, *, little_endian: bool = True) -> np.ndarray:
        """Decode a base64 block of int16 samples (GE MUSE / Philips style)."""
        raw = base64.b64decode(text.strip())
        dtype = "<i2" if little_endian else ">i2"
        return np.frombuffer(raw, dtype=dtype).astype(np.int64)
