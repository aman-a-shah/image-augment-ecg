"""Phase 4 gate: camera/lighting/capture (Layers 3-5) + full-pipeline integration."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from physiorender import config
from physiorender.assemble import build_metadata
from physiorender.degrade import DegradationEngine
from physiorender.degrade import imageutil as iu
from physiorender.degrade import layer3_camera as l3
from physiorender.degrade import layer4_lighting as l4
from physiorender.degrade import layer5_capture as l5
from physiorender.degrade.light import LightSource
from physiorender.ingest.record import ECGRecord, LeadSignal
from physiorender.config import STANDARD_LEADS
from physiorender.params import AugmentationParams
from physiorender.render import ECGRenderer

from .conftest import make_synthetic_record

PPM = config.mm_to_px(1.0, 100)


def _smooth(h=200, w=300):
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    r = 0.5 + 0.3 * np.sin(3 * xx)
    g = 0.5 + 0.3 * np.cos(2 * yy) + 0 * xx
    b = 0.5 + 0.2 * (xx + yy) / 2
    return np.clip(np.stack([r + 0 * yy, g, b + 0 * yy], -1), 0, 1).astype(np.float32)


# --- lens -------------------------------------------------------------------
def test_barrel_noop_when_zero():
    img = _smooth()
    lens = l3.make_lens(300, 200, 0.0)
    assert np.array_equal(l3.apply_barrel(img, lens), img)


def test_barrel_changes_image():
    img = _smooth()
    lens = l3.make_lens(300, 200, 0.06)
    assert not np.allclose(l3.apply_barrel(img, lens), img)


def test_distort_point_identity_when_zero():
    lens = l3.make_lens(300, 200, 0.0)
    assert l3.distort_point((10, 20), lens) == (10, 20)


def test_distort_point_moves_corner_toward_center():
    lens = l3.make_lens(300, 200, 0.06)
    cx, cy = 150, 100
    dx, dy = l3.distort_point((0, 0), lens)
    # Feature near a corner appears closer to the center under barrel distortion.
    assert math.hypot(dx - cx, dy - cy) < math.hypot(0 - cx, 0 - cy)


# --- perspective ------------------------------------------------------------
def test_zero_tilt_corners_are_rectangle():
    dst = l3.project_corners(300, 200, 0, 0, 0)
    # Top edge horizontal, left edge vertical.
    assert abs(dst[0, 1] - dst[1, 1]) < 1e-6
    assert abs(dst[0, 0] - dst[3, 0]) < 1e-6


def test_homography_inverse_consistent():
    img = _smooth()
    rng = np.random.default_rng(0)
    p = l3.frame_and_warp(img, rng, tilt_x_deg=8, tilt_y_deg=-6, rotation_deg=3,
                          crop_margin=0.08, out_size=(300, 200))
    assert np.allclose(p.H @ p.H_inv, np.eye(3), atol=1e-4)
    assert p.image.shape[:2] == (200, 300)
    assert 0.2 < float(p.mask.mean()) < 1.0


def test_perspective_recovers_near_clean():
    """Phase 4 gate: H_inv unwarps the perspective back to near-clean."""
    img = _smooth(240, 320)
    rng = np.random.default_rng(3)
    p = l3.frame_and_warp(img, rng, tilt_x_deg=10, tilt_y_deg=8, rotation_deg=4,
                          crop_margin=0.05, out_size=(320, 240))
    recovered = cv2.warpPerspective(p.image, p.H_inv, (320, 240))
    interior = np.abs(recovered - img)[30:-30, 30:-30]
    assert interior.mean() < 0.02, interior.mean()


def test_transform_bbox_within_image():
    rng = np.random.default_rng(1)
    lens = l3.make_lens(320, 240, 0.05)
    p = l3.frame_and_warp(_smooth(240, 320), rng, tilt_x_deg=6, tilt_y_deg=5,
                          rotation_deg=2, crop_margin=0.1, out_size=(320, 240))
    bb = l3.transform_bbox([50, 40, 180, 120], lens, p.H, (320, 240))
    x1, y1, x2, y2 = bb
    assert 0 <= x1 < x2 <= 320 and 0 <= y1 < y2 <= 240


# --- blur -------------------------------------------------------------------
def test_blur_none_is_noop():
    img = _smooth()
    out = l3.apply_blur(img, np.random.default_rng(0), blur_type="none",
                        strength=0.5, ppm=PPM)
    assert np.array_equal(out, img)


@pytest.mark.parametrize("kind", ["motion", "defocus", "handshake"])
def test_blur_modes_change_image(kind):
    img = _smooth()
    out = l3.apply_blur(img, np.random.default_rng(0), blur_type=kind,
                        strength=0.8, ppm=config.mm_to_px(1.0, 300))
    assert not np.allclose(out, img)


# --- lighting ---------------------------------------------------------------
def test_ambient_gradient_brightens_toward_light():
    img = np.full((100, 200, 3), 0.5, np.float32)
    out = l4.apply_ambient_gradient(img, LightSource(0, 45), strength=0.3)  # light +x
    assert iu.luminance(out)[:, -10:].mean() > iu.luminance(out)[:, :10].mean()


def test_specular_adds_brightness():
    img = np.full((120, 160, 3), 0.4, np.float32)
    out = l4.apply_specular(img, np.random.default_rng(0), LightSource(0, 45),
                            intensity=0.8)
    assert out.max() > img.max()


def test_fl_banding_changes_image():
    img = np.full((120, 160, 3), 0.5, np.float32)
    out = l4.apply_fluorescent_banding(img, np.random.default_rng(0))
    assert not np.allclose(out, img)


def test_hand_shadow_darkens():
    img = np.full((120, 160, 3), 0.6, np.float32)
    out = l4.apply_hand_shadow(img, np.random.default_rng(0), width_fraction=0.25)
    assert out.mean() < img.mean()


# --- capture ----------------------------------------------------------------
def test_sensor_noise_changes_and_reproducible():
    img = np.full((80, 100, 3), 0.5, np.float32)
    a = l5.apply_sensor_noise(img, np.random.default_rng(7), iso_equiv=800)
    b = l5.apply_sensor_noise(img, np.random.default_rng(7), iso_equiv=800)
    assert not np.allclose(a, img)
    assert np.array_equal(a, b)


def test_color_temp_zero_noop():
    img = _smooth()
    assert np.array_equal(l5.apply_color_temperature(img, delta_k=0), img)


def test_jpeg_preserves_shape_and_compresses():
    img = _smooth(128, 128)
    out = l5.apply_jpeg(img, quality=65)
    assert out.shape == img.shape
    assert not np.allclose(out, img)


# --- full pipeline integration ---------------------------------------------
def _small_render():
    rec = make_synthetic_record(seconds=10.0)
    return rec, ECGRenderer(dpi=100).render(rec)


def test_augment_end_to_end():
    rec, render = _small_render()
    params = AugmentationParams(
        yellowing_intensity=0.2, n_wrinkles=2, wrinkle_intensity=0.6, n_folds=1,
        tilt_x_deg=7, tilt_y_deg=-5, rotation_deg=3, blur_type="defocus",
        blur_strength=0.4, lens_k1=0.04, has_specular=True, jpeg_quality=80,
    )
    eng = DegradationEngine(dpi=100)
    result = eng.augment(render.image, params, seed=5, lead_bboxes=render.lead_bboxes)

    assert result.image.size == render.image.size
    assert np.asarray(result.homography_inv).shape == (3, 3)
    W, H = result.image.size
    for lead in STANDARD_LEADS:
        x1, y1, x2, y2 = result.lead_bboxes[lead]
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
    assert result.warp_field_inverse().shape == (H, W, 2)
    assert "perspective" in result.applied and "jpeg" in result.applied


def test_augment_reproducible():
    rec, render = _small_render()
    params = AugmentationParams(n_wrinkles=3, wrinkle_intensity=0.7, tilt_x_deg=6,
                               blur_type="motion", blur_strength=0.5)
    eng = DegradationEngine(dpi=100)
    a = np.asarray(eng.augment(render.image, params, seed=9,
                               lead_bboxes=render.lead_bboxes).image)
    b = np.asarray(eng.augment(render.image, params, seed=9,
                               lead_bboxes=render.lead_bboxes).image)
    assert np.array_equal(a, b)


def test_document_fully_in_frame_no_crop():
    """No part of the ECG is ever cut off: the whole document quad stays in frame."""
    from physiorender.sampling import ParameterSampler
    rec, render = _small_render()
    w, h = render.image.size
    eng = DegradationEngine(dpi=100)
    samp = ParameterSampler()
    rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(1, -1, 2)
    for seed in range(20):
        params = samp.sample(np.random.default_rng(seed))
        result = eng.augment(render.image, params, seed=seed,
                             lead_bboxes=render.lead_bboxes)
        Wp, Hp = result.image.size
        quad = cv2.perspectiveTransform(rect, result.homography).reshape(-1, 2)
        assert quad[:, 0].min() >= -1.0, (seed, quad)
        assert quad[:, 1].min() >= -1.0, (seed, quad)
        assert quad[:, 0].max() <= Wp + 1.0, (seed, quad)
        assert quad[:, 1].max() <= Hp + 1.0, (seed, quad)


def test_all_lead_bboxes_inside_frame():
    """Every lead's post-warp bbox is strictly inside the image (not clamped to an edge)."""
    from physiorender.sampling import ParameterSampler
    rec, render = _small_render()
    eng = DegradationEngine(dpi=100)
    samp = ParameterSampler()
    for seed in range(10):
        params = samp.sample(np.random.default_rng(seed))
        result = eng.augment(render.image, params, seed=seed,
                             lead_bboxes=render.lead_bboxes)
        Wp, Hp = result.image.size
        for lead in STANDARD_LEADS:
            x1, y1, x2, y2 = result.lead_bboxes[lead]
            assert 0 < x1 < x2 < Wp, (seed, lead, (x1, y1, x2, y2))
            assert 0 < y1 < y2 < Hp, (seed, lead, (x1, y1, x2, y2))


def test_displacement_replicates_not_mirrors():
    """Warp border must extend (replicate) the edge, never mirror page content back in."""
    from physiorender.degrade.warp import apply_displacement
    w = 80
    ramp = np.tile(np.linspace(0, 1, w, dtype=np.float32), (40, 1))[..., None].repeat(3, 2)
    dx = np.full((40, w), 6.0, np.float32)      # sample 6px to the right
    dy = np.zeros((40, w), np.float32)
    out = apply_displacement(ramp, dx, dy)
    # Right edge samples beyond -> replicate keeps it ~1.0; a mirror would dip below.
    assert out[:, -1].mean() > 0.95


def test_sampler_never_emits_pen_marks():
    from physiorender.sampling import ParameterSampler
    samp = ParameterSampler()
    for seed in range(100):
        p = samp.sample(np.random.default_rng(seed))
        assert not p.has_pen_marks
        assert p.n_pen_marks == 0
        assert p.pen_count == 0


def test_build_metadata_valid_and_round_trips():
    rec, render = _small_render()
    params = AugmentationParams(n_wrinkles=1, tilt_x_deg=5, has_specular=True)
    result = DegradationEngine(dpi=100).augment(render.image, params, seed=2,
                                                lead_bboxes=render.lead_bboxes)
    meta = build_metadata(rec, render, result, image_id="t_00001",
                          warp_field_filename="warp_t_00001.npy")
    meta.validate()  # should not raise
    from physiorender.metadata import ECGMetadata
    restored = ECGMetadata.from_json(meta.to_json())
    assert restored.to_dict() == meta.to_dict()
    assert restored.warp_field == "warp_t_00001.npy"
    assert len(restored.leads) == 12
