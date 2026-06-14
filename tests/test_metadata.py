"""Phase 0 gate: ECGMetadata round-trips and validates."""

from __future__ import annotations

import pytest

from physiorender.config import STANDARD_LEADS
from physiorender.metadata import ECGMetadata, Layout, LeadData
from physiorender.params import AugmentationParams

IDENTITY_H = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _make_metadata(*, leads=STANDARD_LEADS) -> ECGMetadata:
    lead_data = {
        name: LeadData(signal_mv=[0.0, 0.1, -0.1], sample_rate_hz=500)
        for name in leads
    }
    bboxes = {name: [0, i * 10, 100, i * 10 + 10] for i, name in enumerate(leads)}
    layout = Layout(
        template="standard_12lead_rhythm",
        paper_speed_mm_s=25,
        gain_mm_mv=10,
        lead_bboxes=bboxes,
    )
    return ECGMetadata(
        image_id="ecg_aug_00142",
        source_xml="patient_0042.xml",
        format_detected="GE_MUSE",
        leads=lead_data,
        layout=layout,
        augmentation=AugmentationParams(n_wrinkles=2, has_specular=True),
        homography_inv=IDENTITY_H,
        warp_field="warp_00142.npy",
    )


def test_valid_record_passes():
    _make_metadata().validate()
    assert _make_metadata().is_valid()


def test_json_round_trip_preserves_everything():
    meta = _make_metadata()
    restored = ECGMetadata.from_json(meta.to_json())
    assert restored.to_dict() == meta.to_dict()
    restored.validate()


def test_save_load_round_trip(tmp_path):
    meta = _make_metadata()
    path = tmp_path / "meta.json"
    meta.save(str(path))
    restored = ECGMetadata.load(str(path))
    assert restored.to_dict() == meta.to_dict()


def test_wrong_lead_count_fails():
    meta = _make_metadata(leads=STANDARD_LEADS[:8])
    with pytest.raises(ValueError, match="expected 12 leads"):
        meta.validate()


def test_inconsistent_sample_rates_fail():
    meta = _make_metadata()
    meta.leads["II"].sample_rate_hz = 250
    with pytest.raises(ValueError, match="inconsistent sample rates"):
        meta.validate()


def test_missing_bbox_fails():
    meta = _make_metadata()
    del meta.layout.lead_bboxes["V6"]
    with pytest.raises(ValueError, match="missing bboxes"):
        meta.validate()


def test_bad_bbox_shape_fails():
    meta = _make_metadata()
    meta.layout.lead_bboxes["I"] = [0, 0, 100]  # only 3 coords
    with pytest.raises(ValueError, match="x1,y1,x2,y2"):
        meta.validate()


def test_bad_homography_shape_fails():
    meta = _make_metadata()
    meta.homography_inv = [[1.0, 0.0], [0.0, 1.0]]  # 2x2
    with pytest.raises(ValueError, match="3x3"):
        meta.validate()


def test_nested_augmentation_validation_propagates():
    meta = _make_metadata()
    meta.augmentation.jpeg_quality = 5  # out of range
    with pytest.raises(ValueError, match="jpeg_quality"):
        meta.validate()


def test_warp_field_optional():
    meta = _make_metadata()
    meta.warp_field = None
    meta.validate()
    restored = ECGMetadata.from_json(meta.to_json())
    assert restored.warp_field is None
