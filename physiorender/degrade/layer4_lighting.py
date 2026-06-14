"""PDE Layer 4 — lighting & environment (plan §6 L4).

Applied in the captured-photo frame, after perspective. Shares the same global
:class:`LightSource` as the Layer-2 crease shading so the scene lighting is
consistent (plan §6 L2/L4).
"""

from __future__ import annotations

import math

import numpy as np

from . import imageutil as iu
from .light import LightSource


def apply_ambient_gradient(img: np.ndarray, light: LightSource,
                           *, strength: float = 0.2) -> np.ndarray:
    """Smooth directional brightness gradient across the frame (plan §6 L4)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w / 2) / (w / 2)
    ny = (yy - h / 2) / (h / 2)
    lx, ly = light.direction
    proj = nx * lx + ny * ly                      # ~[-1, 1] along light direction
    gain = 1.0 + strength * proj
    return iu.apply_brightness(img, gain.astype(np.float32))


def apply_specular(img: np.ndarray, rng: np.random.Generator, light: LightSource,
                   *, intensity: float) -> np.ndarray:
    """Soft elliptical specular hot-spot that washes out the trace (plan §6 L4)."""
    h, w = img.shape[:2]
    # Place the hot-spot toward the light direction from center.
    lx, ly = light.direction
    cx = w / 2 + lx * rng.uniform(0.1, 0.35) * w
    cy = h / 2 + ly * rng.uniform(0.1, 0.35) * h
    rx = rng.uniform(0.12, 0.28) * w
    ry = rx * rng.uniform(0.6, 1.4)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    spot = np.exp(-0.5 * d).astype(np.float32)
    a = (intensity * spot)[..., None]
    # Blend toward white -> locally washes out detail.
    return np.clip(img * (1 - a) + a, 0, 1).astype(np.float32)


def apply_fluorescent_banding(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Subtle horizontal AC-flicker bands (plan §6 L4)."""
    h, w = img.shape[:2]
    period = rng.uniform(30, 60) * (h / 700.0)   # scale with resolution
    phase = rng.uniform(0, 2 * math.pi)
    y = np.arange(h, dtype=np.float32)
    band = 1.0 + 0.03 * np.sin(2 * math.pi * y / max(2.0, period) + phase)
    return iu.apply_brightness(img, band[:, None].astype(np.float32))


def apply_hand_shadow(img: np.ndarray, rng: np.random.Generator,
                      *, width_fraction: float) -> np.ndarray:
    """Soft shadow cast by the hand, encroaching from one edge (plan §6 L4)."""
    if width_fraction <= 0:
        return img
    h, w = img.shape[:2]
    edge = rng.integers(0, 4)  # 0 left, 1 right, 2 top, 3 bottom
    if edge in (0, 1):
        coord = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        coord = coord if edge == 0 else coord[:, ::-1]
        ramp = np.repeat(coord, h, axis=0)
    else:
        coord = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        coord = coord if edge == 2 else coord[::-1, :]
        ramp = np.repeat(coord, w, axis=1)
    width = max(1e-3, width_fraction)
    darken = 1.0 - 0.45 * np.clip(1.0 - ramp / width, 0, 1)
    return iu.apply_brightness(img, darken.astype(np.float32))
