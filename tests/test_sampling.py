"""Phase 5 gate: correlation-aware sampling, dataset generation, calibration."""

from __future__ import annotations

import numpy as np
import pytest

from physiorender.calibration import compare, image_stats
from physiorender.dataset_gen import generate_dataset, validate_dataset
from physiorender.sampling import ParameterSampler, SamplerConfig

from .conftest import DUMMY_XML


# --- sampler ----------------------------------------------------------------
def test_sampled_params_always_valid():
    s = ParameterSampler()
    for seed in range(50):
        s.sample(np.random.default_rng(seed)).validate()  # must not raise


def test_sampler_reproducible():
    s = ParameterSampler()
    a = s.sample(np.random.default_rng(3))
    b = s.sample(np.random.default_rng(3))
    assert a == b


def test_blur_suppresses_specular_on_average():
    """High-blur draws should have specular set less often than low-blur draws."""
    s = ParameterSampler(SamplerConfig(blur_suppresses_specular=1.0, p_blur=1.0))
    high_blur_spec = low_blur_spec = 0
    high_n = low_n = 0
    for seed in range(800):
        p = s.sample(np.random.default_rng(seed))
        if p.blur_strength > 0.6:
            high_n += 1
            high_blur_spec += int(p.has_specular)
        elif 0 < p.blur_strength < 0.4:
            low_n += 1
            low_blur_spec += int(p.has_specular)
    high_rate = high_blur_spec / max(1, high_n)
    low_rate = low_blur_spec / max(1, low_n)
    assert high_rate < low_rate


def test_folds_increase_yellowing_on_average():
    s = ParameterSampler(SamplerConfig(folds_increase_yellowing=1.0))
    y0, n0, y2, n2 = 0.0, 0, 0.0, 0
    for seed in range(800):
        p = s.sample(np.random.default_rng(seed))
        if p.n_folds == 0:
            y0 += p.yellowing_intensity; n0 += 1
        elif p.n_folds == 2:
            y2 += p.yellowing_intensity; n2 += 1
    assert (y2 / n2) > (y0 / n0)


# --- dataset generation -----------------------------------------------------
@pytest.mark.skipif(not DUMMY_XML.exists(), reason="sample file absent")
def test_generate_small_dataset(tmp_path):
    report = generate_dataset([DUMMY_XML], out_dir=tmp_path, n_per_source=3,
                              dpi=100, save_warp=False, write_signals=False)
    assert report.n_images == 3
    assert report.ok
    assert (tmp_path / "manifest.jsonl").exists()
    jpgs = list(tmp_path.glob("*.jpg"))
    assert len(jpgs) == 3

    integrity = validate_dataset(tmp_path)
    assert integrity.n_images == 3
    assert integrity.ok


@pytest.mark.skipif(not DUMMY_XML.exists(), reason="sample file absent")
def test_generation_is_deterministic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_dataset([DUMMY_XML], out_dir=a, n_per_source=2, dpi=100,
                     save_warp=False, write_signals=False, base_seed=7)
    generate_dataset([DUMMY_XML], out_dir=b, n_per_source=2, dpi=100,
                     save_warp=False, write_signals=False, base_seed=7)
    import numpy as np
    from PIL import Image
    for f in a.glob("*.jpg"):
        ia = np.asarray(Image.open(f))
        ib = np.asarray(Image.open(b / f.name))
        assert np.array_equal(ia, ib)


# --- calibration ------------------------------------------------------------
def test_image_stats_keys():
    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    s = image_stats(img)
    assert set(s) == {"brightness", "contrast", "sharpness", "colorfulness", "edge_density"}


def test_compare_flags_brightness_gap():
    bright = [image_stats(np.full((32, 32, 3), 220, np.uint8)) for _ in range(5)]
    dark = [image_stats(np.full((32, 32, 3), 40, np.uint8)) for _ in range(5)]
    report = compare(bright, dark)
    assert "brightness" in report.worst_offenders(5)
    assert report.z_divergence["brightness"] > 1.0
