"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from physiorender.config import STANDARD_LEADS
from physiorender.ingest.record import ECGRecord, LeadSignal

DUMMY_XML = Path(__file__).resolve().parent.parent / "data" / "dummy-ecg.xml"


@pytest.fixture
def dummy_xml_path() -> Path:
    if not DUMMY_XML.exists():
        pytest.skip(f"sample file not present: {DUMMY_XML}")
    return DUMMY_XML


def make_synthetic_record(
    *, fs: int = 500, seconds: float = 10.0, amplitude_mv: float = 1.0
) -> ECGRecord:
    """A 12-lead record with a simple sine on every lead (no file needed)."""
    n = int(fs * seconds)
    t = np.arange(n) / fs
    wave = (amplitude_mv * np.sin(2 * np.pi * 1.0 * t)).astype(np.float32)
    leads = {
        name: LeadSignal(name=name, signal_mv=wave.copy(), sample_rate_hz=fs)
        for name in STANDARD_LEADS
    }
    return ECGRecord(
        leads=leads,
        sample_rate_hz=fs,
        source_path="synthetic.xml",
        source_format="SYNTHETIC",
        lead_order=list(STANDARD_LEADS),
    )


def make_flatline_record(*, fs: int = 500, seconds: float = 10.0) -> ECGRecord:
    """A 12-lead record of pure 0 mV — used to isolate calibration pulses."""
    n = int(fs * seconds)
    zeros = np.zeros(n, dtype=np.float32)
    leads = {
        name: LeadSignal(name=name, signal_mv=zeros.copy(), sample_rate_hz=fs)
        for name in STANDARD_LEADS
    }
    return ECGRecord(
        leads=leads,
        sample_rate_hz=fs,
        source_path="flat.xml",
        source_format="SYNTHETIC",
        lead_order=list(STANDARD_LEADS),
    )
