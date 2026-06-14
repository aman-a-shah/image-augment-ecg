"""Smooth multi-octave noise fields (plan §6 — "smooth Perlin noise field").

We generate value/fractal noise with numpy + cubic upsampling rather than taking
a fragile native Perlin dependency: fully seedable, reproducible, and gives the
spatially-correlated low-frequency structure the paper artifacts need (yellowing,
ink density). plan.md §2: degradation must be *spatially correlated*, not random.
"""

from __future__ import annotations

import cv2
import numpy as np


def fractal_noise(
    h: int,
    w: int,
    rng: np.random.Generator,
    *,
    cell_px: float = 80.0,
    octaves: int = 4,
    persistence: float = 0.5,
    normalize: str = "unit",  # "unit" -> [0,1], "signed" -> [-1,1]
) -> np.ndarray:
    """Return an [h, w] smooth noise field.

    ``cell_px`` is the approximate feature size of the lowest-frequency octave;
    smaller = finer texture. Each octave halves the feature size and scales the
    amplitude by ``persistence``.
    """
    field = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total = 0.0
    for o in range(octaves):
        cell = max(2.0, cell_px / (2 ** o))
        gh = max(2, int(round(h / cell)) + 2)
        gw = max(2, int(round(w / cell)) + 2)
        grid = rng.standard_normal((gh, gw)).astype(np.float32)
        up = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        field += amp * up
        total += amp
        amp *= persistence
    field /= total

    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-8:
        field = np.zeros_like(field)
    else:
        field = (field - lo) / (hi - lo)  # -> [0,1]

    if normalize == "signed":
        field = field * 2.0 - 1.0
    return field.astype(np.float32)


def edge_weight(h: int, w: int, *, falloff: float = 0.35) -> np.ndarray:
    """Field that is ~1 near the page edges and ~0 in the center.

    Used to bias thermal yellowing toward edges/folds (plan §6 L1: "heavier
    yellowing toward edges"). ``falloff`` controls how quickly it ramps inward.
    """
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    # Distance from center toward edges (max-norm gives a rectangular ramp).
    d = np.maximum(np.abs(yy), np.abs(xx))
    w_edge = np.clip((d - (1.0 - falloff)) / falloff, 0.0, 1.0)
    return (w_edge ** 1.5).astype(np.float32)
