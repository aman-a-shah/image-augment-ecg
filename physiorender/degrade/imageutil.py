"""Image conversion helpers for the degradation engine.

Internally the PDE works on **float32 RGB in [0, 1]** so photometric math
(HSV shifts, multiplicative fields, brightness) is clean and lossless. We convert
to/from PIL/uint8 only at the boundaries.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def pil_to_float(img: Image.Image) -> np.ndarray:
    """PIL RGB image -> float32 array [H, W, 3] in [0, 1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def float_to_pil(arr: np.ndarray) -> Image.Image:
    """float32 [0, 1] array -> 8-bit PIL RGB image."""
    out = np.clip(arr, 0.0, 1.0) * 255.0
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    """float RGB [0,1] -> HSV with H in [0,360], S,V in [0,1]."""
    return cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)


def hsv_to_rgb(arr: np.ndarray) -> np.ndarray:
    """float HSV (H[0,360], S,V[0,1]) -> RGB [0,1]."""
    return cv2.cvtColor(arr, cv2.COLOR_HSV2RGB)


def luminance(arr: np.ndarray) -> np.ndarray:
    """Perceptual luminance [H, W] from float RGB."""
    return arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def apply_brightness(arr: np.ndarray, gain: np.ndarray | float) -> np.ndarray:
    """Multiply RGB by a per-pixel (or scalar) brightness gain, clipped to [0,1]."""
    if np.ndim(gain) == 2:
        gain = gain[:, :, None]
    return np.clip(arr * gain, 0.0, 1.0)
