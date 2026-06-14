"""Phase 1 gate: ingestion correctness."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from physiorender.config import STANDARD_LEADS
from physiorender.ingest import (
    CardiologyXMLExtractor,
    ECGFormat,
    GenericXMLExtractor,
    detect_ecg_xml_format,
    load_ecg,
    validate_record,
)
from physiorender.ingest.base import SignalExtractor


# --- decoding helpers -------------------------------------------------------
def test_counts_to_mv():
    counts = np.array([0, 200, -200], dtype=np.int64)
    mv = SignalExtractor.counts_to_mv(counts, uv_per_lsb=5.0)
    # 200 counts * 5 uV = 1000 uV = 1.0 mV
    assert np.allclose(mv, [0.0, 1.0, -1.0])


def test_parse_csv_ints_handles_newlines_and_spaces():
    out = SignalExtractor.parse_csv_ints("1,2,\n3, 4,\t5")
    assert out.tolist() == [1, 2, 3, 4, 5]


def test_parse_csv_ints_empty():
    assert SignalExtractor.parse_csv_ints("   ").size == 0


def test_decode_base64_int16_round_trip():
    arr = np.array([0, 1, -1, 1000, -1000], dtype="<i2")
    text = base64.b64encode(arr.tobytes()).decode()
    out = SignalExtractor.decode_base64_int16(text)
    assert out.tolist() == arr.tolist()


# --- format detection -------------------------------------------------------
def test_detect_cardiology_xml(dummy_xml_path):
    assert detect_ecg_xml_format(dummy_xml_path) == ECGFormat.CARDIOLOGY_XML


def test_detect_wfdb_by_extension(tmp_path):
    hea = tmp_path / "rec.hea"
    hea.write_text("rec 12 500 5000\n")
    assert detect_ecg_xml_format(hea) == ECGFormat.WFDB


def test_detect_unknown_on_garbage(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<<<not xml")
    assert detect_ecg_xml_format(bad) == ECGFormat.UNKNOWN


# --- end-to-end on the real sample -----------------------------------------
def test_load_dummy(dummy_xml_path):
    rec = load_ecg(dummy_xml_path, validate=False)
    assert rec.source_format == "CARDIOLOGY_XML"
    assert rec.n_leads == 12
    assert set(rec.leads) == set(STANDARD_LEADS)
    assert rec.sample_rate_hz == 500
    assert rec.lead_order[:3] == ["I", "II", "III"]
    # 5000 samples @ 500 Hz = 10s (StripData preferred over MedianSamples)
    assert abs(rec.duration_s - 10.0) < 1e-6
    assert rec.leads["I"].n_samples == 5000


def test_dummy_amplitudes_physiological(dummy_xml_path):
    rec = load_ecg(dummy_xml_path, validate=False)
    for lead in rec.leads.values():
        peak = float(np.max(np.abs(lead.signal_mv)))
        assert 0.05 < peak < 12.0, f"{lead.name} peak {peak} mV implausible"


def test_dummy_validates_clean(dummy_xml_path):
    rec = load_ecg(dummy_xml_path, validate=False)
    report = validate_record(rec)
    assert report.ok, report.errors


def test_matrix_shape(dummy_xml_path):
    rec = load_ecg(dummy_xml_path, validate=False)
    mat, order = rec.matrix()
    assert mat.shape == (12, 5000)
    assert order == rec.lead_order


# --- generic fallback should also parse the CardioSoft CSV ------------------
def test_generic_extractor_parses_dummy(dummy_xml_path):
    rec = GenericXMLExtractor().extract(dummy_xml_path)
    assert rec.n_leads == 12
    assert set(rec.leads) == set(STANDARD_LEADS)


# --- validation catches bad data -------------------------------------------
def test_validation_flags_missing_leads(dummy_xml_path):
    rec = CardiologyXMLExtractor().extract(dummy_xml_path)
    del rec.leads["V6"]
    rec.lead_order.remove("V6")
    report = validate_record(rec)
    assert not report.ok
    assert any("expected" in e for e in report.errors)


def test_validation_flags_inconsistent_lengths(dummy_xml_path):
    rec = CardiologyXMLExtractor().extract(dummy_xml_path)
    rec.leads["II"].signal_mv = rec.leads["II"].signal_mv[:100]
    report = validate_record(rec)
    assert not report.ok
    assert any("sample counts" in e for e in report.errors)
