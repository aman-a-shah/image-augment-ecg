"""Phase 2 gate: renderer correctness, incl. amplitude calibration (1mV = 10mm)."""

from __future__ import annotations

import numpy as np
import pytest

from physiorender import config
from physiorender.config import STANDARD_LEADS
from physiorender.render import ECGRenderer, build_standard_12lead
from physiorender.render.renderer import _CAL_PULSE_MV

from .conftest import make_flatline_record, make_synthetic_record


def test_render_produces_expected_geometry():
    rec = make_synthetic_record()
    result = ECGRenderer(dpi=300).render(rec)
    # 12 grid leads + rhythm strip = 13 bboxes; all standard leads covered.
    assert len(result.lead_bboxes) == 13
    for lead in STANDARD_LEADS:
        assert lead in result.lead_bboxes
    assert "II_rhythm" in result.lead_bboxes
    assert result.template == "standard_12lead_rhythm"
    w, h = result.image.size
    assert w > 2000 and h > 1500  # ~A4 landscape at 300 DPI


def test_bboxes_are_well_formed_and_within_image():
    rec = make_synthetic_record()
    result = ECGRenderer(dpi=300).render(rec)
    W, H = result.image.size
    for key, (x1, y1, x2, y2) in result.lead_bboxes.items():
        assert 0 <= x1 < x2 <= W, key
        assert 0 <= y1 < y2 <= H, key


def test_no_rhythm_layout_has_12_bboxes():
    rec = make_synthetic_record()
    layout = build_standard_12lead(rhythm=False)
    result = ECGRenderer(dpi=300).render(rec, layout)
    assert len(result.lead_bboxes) == 12
    assert result.template == "standard_12lead"


def test_y_transform_is_linear_in_gain():
    """1 mV must map to gain_mm_mv * (px per mm) pixels of vertical displacement."""
    rec = make_synthetic_record()
    r = ECGRenderer(dpi=300, gain_mm_mv=10, supersample=1)
    layout = build_standard_12lead()
    panel = layout.panels[0]
    y0 = r._y_for_mv(panel, 0.0)
    y1 = r._y_for_mv(panel, 1.0)
    expected = 10 * config.mm_to_px(1.0, 300)  # 10mm at 300 DPI
    assert abs((y0 - y1) - expected) < 1e-6


def test_amplitude_calibration_end_to_end():
    """Render a flatline ECG; the only excursions are 1mV cal pulses.

    Measure a calibration pulse's pixel height and confirm it equals ~10mm.
    """
    rec = make_flatline_record()
    gain = 10
    result = ECGRenderer(dpi=300, paper_speed_mm_s=25, gain_mm_mv=gain,
                         supersample=2).render(rec)
    arr = np.asarray(result.image.convert("L"))

    # Calibration pulse sits in the left margin, well before the first panel.
    # x band ~3-10mm (pulse spans ~4-9mm); restrict to the top row region.
    ppm = config.mm_to_px(1.0, 300)
    x_lo, x_hi = int(2 * ppm), int(11 * ppm)
    y_lo, y_hi = 0, int(config.mm_to_px(60, 300))  # first row area
    band = arr[y_lo:y_hi, x_lo:x_hi]

    dark_rows = np.where((band < 100).any(axis=1))[0]
    assert dark_rows.size > 0, "no calibration pulse found"
    pulse_h_px = dark_rows.max() - dark_rows.min()

    expected = _CAL_PULSE_MV * gain * ppm  # 1mV * 10mm/mV -> 10mm in px
    # Allow ~1.5mm tolerance for anti-aliasing / line width.
    assert abs(pulse_h_px - expected) < config.mm_to_px(1.5, 300), (
        f"cal pulse {pulse_h_px}px vs expected {expected:.1f}px"
    )


def test_trace_pixels_present():
    """A sine-wave render must actually put dark trace pixels on the page."""
    rec = make_synthetic_record(amplitude_mv=1.0)
    result = ECGRenderer(dpi=300).render(rec)
    arr = np.asarray(result.image.convert("L"))
    dark_fraction = float((arr < 80).mean())
    assert dark_fraction > 0.001  # non-trivial amount of ink
