"""Phase 3 gate: PDE Layers 1-2 — isolation, plausibility, invertibility."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from physiorender import config
from physiorender.degrade import DegradationEngine, DisplacementField
from physiorender.degrade import imageutil as iu
from physiorender.degrade import layer1_paper as l1
from physiorender.degrade import layer2_handling as l2
from physiorender.degrade.light import LightSource
from physiorender.degrade.noise import edge_weight, fractal_noise
from physiorender.degrade.warp import apply_displacement, invert_displacement
from physiorender.params import AugmentationParams

PPM = config.mm_to_px(1.0, 300)


# --- fixtures ---------------------------------------------------------------
def _smooth_image(h: int = 240, w: int = 320) -> np.ndarray:
    """A smooth low-frequency RGB image (no high freq -> clean warp round-trips)."""
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    r = 0.5 + 0.3 * np.sin(3 * xx + 1)
    g = 0.5 + 0.3 * np.cos(2 * yy)
    b = 0.5 + 0.2 * np.sin(2 * xx + 2 * yy)
    return np.clip(np.stack([r + 0 * yy, g + 0 * xx, b], axis=-1), 0, 1).astype(np.float32)


def _clean_pil(h: int = 240, w: int = 320) -> Image.Image:
    return iu.float_to_pil(_smooth_image(h, w))


# --- noise ------------------------------------------------------------------
def test_fractal_noise_range_and_repro():
    a = fractal_noise(64, 80, np.random.default_rng(1))
    b = fractal_noise(64, 80, np.random.default_rng(1))
    assert a.shape == (64, 80)
    assert 0.0 <= a.min() and a.max() <= 1.0
    assert np.array_equal(a, b)  # seeded -> reproducible


def test_fractal_noise_signed():
    a = fractal_noise(64, 64, np.random.default_rng(2), normalize="signed")
    assert a.min() < 0 < a.max()
    assert a.min() >= -1.0001 and a.max() <= 1.0001


def test_edge_weight_higher_at_edges():
    ew = edge_weight(100, 100)
    assert ew[0, 0] > ew[50, 50]
    assert ew[50, 50] < 0.2


# --- light ------------------------------------------------------------------
def test_light_direction_unit():
    L = LightSource(0.0, 45.0)
    dx, dy = L.direction
    assert abs(dx - 1.0) < 1e-6 and abs(dy) < 1e-6


def test_grazing_strength_monotonic():
    assert LightSource(0, 20).grazing_strength > LightSource(0, 70).grazing_strength


# --- warp -------------------------------------------------------------------
def test_identity_displacement_is_noop():
    img = _smooth_image()
    out = apply_displacement(img, np.zeros(img.shape[:2], np.float32),
                             np.zeros(img.shape[:2], np.float32))
    assert np.allclose(out, img, atol=1e-4)


def test_displacement_invertible_on_smooth_image():
    """The Phase 3 gate: composite warp applied + inverted recovers the original."""
    img = _smooth_image()
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = 5.0 * np.sin(yy / 40.0)
    dy = 5.0 * np.cos(xx / 40.0)
    warped = apply_displacement(img, dx, dy)
    idx, idy = invert_displacement(dx, dy, iters=15)
    recovered = apply_displacement(warped, idx, idy)
    interior = np.abs(recovered - img)[30:-30, 30:-30]
    assert interior.mean() < 0.01, interior.mean()


def test_displacement_field_accumulates():
    d = DisplacementField.zeros(20, 20)
    assert d.is_identity()
    d.add(np.ones((20, 20), np.float32), np.zeros((20, 20), np.float32))
    assert not d.is_identity()
    assert d.dx.mean() == 1.0


# --- layer 1 ----------------------------------------------------------------
def test_yellowing_noop_at_zero():
    img = _smooth_image()
    assert np.array_equal(l1.apply_yellowing(img, np.random.default_rng(0), 0.0), img)


def test_yellowing_warms_image():
    img = _smooth_image()
    out = l1.apply_yellowing(img, np.random.default_rng(0), 0.3)
    # Warmer => mean red rises relative to blue.
    assert (out[..., 0] - out[..., 2]).mean() > (img[..., 0] - img[..., 2]).mean()


def test_ink_density_noop_at_zero():
    img = _smooth_image()
    assert np.array_equal(l1.apply_ink_density(img, np.random.default_rng(0), 0.0), img)


def test_ink_skip_lightens_dark_pixels():
    img = np.zeros((50, 50, 3), np.float32)  # all dark
    out = l1.apply_ink_skip(img, np.random.default_rng(0), 20)
    assert out.max() > 0.5  # some pixels lightened toward paper


# --- layer 2 ----------------------------------------------------------------
def test_wrinkles_change_image_and_accumulate_displacement():
    img = _smooth_image()
    disp = DisplacementField.zeros(*img.shape[:2])
    out = l2.add_wrinkles(img, disp, np.random.default_rng(3), n=4, intensity=1.0,
                          light=LightSource(35, 40), ppm=PPM)
    assert not disp.is_identity()
    assert not np.allclose(out, img)


def test_folds_displace_more_than_wrinkles():
    img = _smooth_image()
    rng_args = dict(light=LightSource(35, 40), ppm=PPM)
    dw = DisplacementField.zeros(*img.shape[:2])
    l2.add_wrinkles(img, dw, np.random.default_rng(5), n=1, intensity=1.0, **rng_args)
    df = DisplacementField.zeros(*img.shape[:2])
    l2.add_folds(img, df, np.random.default_rng(5), n=1, intensity=1.0, **rng_args)
    assert df.magnitude().max() > dw.magnitude().max()


def test_stain_alters_a_region():
    img = _smooth_image()
    out = l2.add_stain(img, np.random.default_rng(0), opacity=0.4)
    assert not np.allclose(out, img)


def test_wrinkle_shading_follows_light_direction():
    """Flipping the light to the opposite side flips which side is brighter."""
    img = np.full((200, 200, 3), 0.6, np.float32)
    d1 = DisplacementField.zeros(200, 200)
    a = l2.add_wrinkles(img, d1, np.random.default_rng(9), n=1, intensity=1.0,
                        light=LightSource(0, 30), ppm=PPM)
    d2 = DisplacementField.zeros(200, 200)
    b = l2.add_wrinkles(img, d2, np.random.default_rng(9), n=1, intensity=1.0,
                        light=LightSource(180, 30), ppm=PPM)
    # Same geometry, opposite light -> shading is (approximately) mirrored in sign.
    da = iu.luminance(a) - iu.luminance(img)
    db = iu.luminance(b) - iu.luminance(img)
    assert np.corrcoef(da.ravel(), db.ravel())[0, 1] < 0


# --- engine -----------------------------------------------------------------
def test_engine_runs_and_is_reproducible():
    clean = _clean_pil()
    params = AugmentationParams(yellowing_intensity=0.2, ink_density_variation=0.05,
                               n_wrinkles=3, wrinkle_intensity=0.7, n_folds=1,
                               has_stain=True, has_pen_marks=True)
    eng = DegradationEngine(dpi=300)
    r1 = eng.apply(clean, params, seed=42)
    r2 = eng.apply(clean, params, seed=42)
    assert r1.image.size == clean.size
    assert np.array_equal(np.asarray(r1.image), np.asarray(r2.image))
    assert r1.applied


def test_engine_different_seeds_differ():
    clean = _clean_pil()
    params = AugmentationParams(n_wrinkles=4, wrinkle_intensity=0.8)
    eng = DegradationEngine(dpi=300)
    a = np.asarray(eng.apply(clean, params, seed=1).image)
    b = np.asarray(eng.apply(clean, params, seed=2).image)
    assert not np.array_equal(a, b)


def test_engine_inverse_warp_shape():
    clean = _clean_pil()
    params = AugmentationParams(n_wrinkles=2, wrinkle_intensity=0.6, n_folds=1)
    result = DegradationEngine(dpi=300).apply(clean, params, seed=0)
    inv = result.inverse_warp()
    assert inv.shape == (clean.size[1], clean.size[0], 2)


def test_engine_invertibility_on_smooth_image():
    """End-to-end: engine's composite warp recovers a smooth image within tolerance."""
    clean = _clean_pil(360, 480)
    params = AugmentationParams(yellowing_intensity=0.0, ink_density_variation=0.0,
                               n_wrinkles=3, wrinkle_intensity=0.7, n_folds=1)
    result = DegradationEngine(dpi=300).apply(clean, params, seed=11)
    base = _smooth_image(360, 480)
    warped = result.displacement.apply(base)
    inv = result.inverse_warp(iters=15)
    recovered = apply_displacement(warped, inv[..., 0], inv[..., 1])
    interior = np.abs(recovered - base)[40:-40, 40:-40]
    assert interior.mean() < 0.01, interior.mean()
