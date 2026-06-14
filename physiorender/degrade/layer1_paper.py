"""PDE Layer 1 — paper and print artifacts (plan §6 L1).

Photometric only (no geometry): thermal yellowing, ink-density variation, grid
bleed, and ink-skip dropouts. All operate on float RGB [0,1] and are individually
toggleable for visual unit testing (plan §6 gate).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import imageutil as iu
from .noise import edge_weight, fractal_noise


def apply_yellowing(img: np.ndarray, rng: np.random.Generator,
                    intensity: float) -> np.ndarray:
    """Spatially-varying thermal-paper yellow/brown tint (plan §6 L1).

    HSV shift H+5deg, S+10%, V-5% at full strength, modulated by a smooth noise
    field and biased toward the edges (paper yellows at folds/edges first).
    """
    if intensity <= 0:
        return img
    h, w = img.shape[:2]
    field = fractal_noise(h, w, rng, cell_px=max(40, w / 6))      # smooth blotches
    bias = 0.5 + 0.5 * edge_weight(h, w)                          # heavier at edges
    amount = (intensity * field * bias).astype(np.float32)        # [H,W] in ~[0,intensity]

    hsv = iu.rgb_to_hsv(img)
    hsv[..., 0] = (hsv[..., 0] + 5.0 * amount) % 360.0            # hue toward yellow
    hsv[..., 1] = np.clip(hsv[..., 1] + 0.10 * amount, 0, 1)      # more saturation
    hsv[..., 2] = np.clip(hsv[..., 2] - 0.05 * amount, 0, 1)      # slightly darker
    return iu.hsv_to_rgb(hsv)


def apply_ink_density(img: np.ndarray, rng: np.random.Generator,
                      variation: float) -> np.ndarray:
    """Low-frequency multiplicative ink-density variation (plan §6 L1).

    rendered *= (1 + variation * perlin) — some areas print slightly lighter/darker.
    """
    if variation <= 0:
        return img
    h, w = img.shape[:2]
    field = fractal_noise(h, w, rng, cell_px=max(50, w / 5), normalize="signed")
    gain = (1.0 + variation * field).astype(np.float32)
    return iu.apply_brightness(img, gain)


def apply_grid_bleed(img: np.ndarray, sigma: float = 0.4) -> np.ndarray:
    """Mild blur emulating ink bleed into adjacent pixels (plan §6 L1)."""
    if sigma <= 0:
        return img
    k = max(1, int(round(sigma * 3)) * 2 + 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_ink_skip(img: np.ndarray, rng: np.random.Generator,
                   n_skips: int, *, bg: tuple[float, float, float] = (1.0, 0.99, 0.97)
                   ) -> np.ndarray:
    """Randomly drop tiny 1-3px segments of the dark trace (plan §6 L1).

    Thermal heads occasionally skip; we lighten short runs of trace pixels toward
    the paper color.
    """
    if n_skips <= 0:
        return img
    out = img.copy()
    lum = iu.luminance(out)
    ys, xs = np.where(lum < 0.35)            # dark trace pixels
    if xs.size == 0:
        return out
    bg_arr = np.array(bg, dtype=np.float32)
    for _ in range(n_skips):
        i = int(rng.integers(0, xs.size))
        cx, cy = int(xs[i]), int(ys[i])
        seg = int(rng.integers(1, 4))        # 1-3 px
        x0, x1 = max(0, cx - seg), min(out.shape[1], cx + seg + 1)
        y0, y1 = max(0, cy - 1), min(out.shape[0], cy + 1)
        out[y0:y1, x0:x1] = bg_arr
    return out
