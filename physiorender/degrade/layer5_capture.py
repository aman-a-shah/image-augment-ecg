"""PDE Layer 5 — sensor noise & compression (plan §6 L5).

Applied last, in capture order: sensor noise in linear light, white-balance shift,
then JPEG re-encode/re-decode so blocking artifacts are baked into the pixels.
"""

from __future__ import annotations

import cv2
import numpy as np

_GAMMA = 2.2


def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 1) ** _GAMMA


def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 1) ** (1.0 / _GAMMA)


def apply_sensor_noise(img: np.ndarray, rng: np.random.Generator,
                       *, iso_equiv: int) -> np.ndarray:
    """Signal-dependent Poisson shot noise + Gaussian read noise (plan §6 L5).

    Done in linear light: brighter regions get proportionally less noise.
    """
    lin = _srgb_to_linear(img)
    # Higher ISO -> fewer effective photons -> more shot noise.
    full_well = max(200.0, 60000.0 / max(50, iso_equiv))
    shot = rng.poisson(np.clip(lin, 0, 1) * full_well) / full_well
    read_sigma = (0.5 + iso_equiv / 1600.0 * 1.5) / 255.0
    read = rng.normal(0.0, read_sigma, lin.shape).astype(np.float32)
    noisy = np.clip(shot + read, 0, 1).astype(np.float32)
    return _linear_to_srgb(noisy)


def apply_color_temperature(img: np.ndarray, *, delta_k: int) -> np.ndarray:
    """Inconsistent auto white balance: warm/cool shift (plan §6 L5)."""
    if delta_k == 0:
        return img
    f = delta_k / 300.0  # [-1, 1] over the sampled range
    r_gain = 1.0 + 0.06 * f
    b_gain = 1.0 - 0.06 * f
    out = img.copy()
    out[..., 0] = np.clip(out[..., 0] * r_gain, 0, 1)
    out[..., 2] = np.clip(out[..., 2] * b_gain, 0, 1)
    return out


def apply_jpeg(img: np.ndarray, *, quality: int) -> np.ndarray:
    """Encode then decode as JPEG so blocking artifacts enter the pixels (plan §6 L5)."""
    bgr = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb
