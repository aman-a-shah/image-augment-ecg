"""Phase 0 gate: AugmentationParams round-trips and validates."""

from __future__ import annotations

import json

import pytest

from physiorender.params import FIELD_RANGES, AugmentationParams


def test_defaults_are_valid():
    AugmentationParams().validate()  # should not raise
    assert AugmentationParams().is_valid()


def test_dict_round_trip():
    p = AugmentationParams(n_wrinkles=3, blur_type="motion", blur_strength=0.7)
    restored = AugmentationParams.from_dict(p.to_dict())
    assert restored == p


def test_json_round_trip():
    p = AugmentationParams(
        yellowing_intensity=0.25,
        has_specular=True,
        jpeg_quality=70,
        colour_temp_delta_k=-150,
    )
    text = json.dumps(p.to_dict())
    restored = AugmentationParams.from_dict(json.loads(text))
    assert restored == p


def test_from_dict_ignores_unknown_keys():
    data = AugmentationParams().to_dict()
    data["future_param_not_yet_defined"] = 123
    restored = AugmentationParams.from_dict(data)
    assert restored == AugmentationParams()


def test_out_of_range_raises():
    p = AugmentationParams(tilt_x_deg=999.0)
    with pytest.raises(ValueError, match="tilt_x_deg"):
        p.validate()
    assert not p.is_valid()


def test_invalid_blur_type_raises():
    p = AugmentationParams(blur_type="warp_drive")
    with pytest.raises(ValueError, match="blur_type"):
        p.validate()


def test_validate_collects_multiple_errors():
    p = AugmentationParams(tilt_x_deg=999.0, jpeg_quality=10, blur_type="nope")
    with pytest.raises(ValueError) as exc:
        p.validate()
    msg = str(exc.value)
    assert "tilt_x_deg" in msg
    assert "jpeg_quality" in msg
    assert "blur_type" in msg


def test_every_numeric_field_default_within_declared_range():
    p = AugmentationParams()
    for name, (lo, hi) in FIELD_RANGES.items():
        value = getattr(p, name)
        assert lo <= value <= hi, f"{name}={value} default outside [{lo}, {hi}]"
