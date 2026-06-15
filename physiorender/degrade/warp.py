"""Displacement-field warping with invertibility (plan §6 L2, §8 warp_field).

Convention: a *backward* displacement field ``(dx, dy)`` defines
``dst(p) = src(p + d(p))`` — for each destination pixel we sample the source at an
offset. Individual layer displacements (wrinkles, folds, edge curl) are summed
into one composite field and applied once via a single remap (plan §6 L2:
"Composite displacement fields → apply to image via cv2.remap()").

The composite field and its numerically-inverted counterpart are exported as the
``warp_field`` supervision signal for the undistortion model (plan §8, §9.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    return np.meshgrid(xs, ys)  # X, Y


def apply_displacement(
    img: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    """Warp ``img`` by backward displacement ``(dx, dy)``.

    Uses BORDER_REPLICATE (not REFLECT) so wrinkle/fold displacement never
    *mirrors* a strip of the page content back into view — it only extends the
    outermost (margin) pixels. Keeps the ECG content from being duplicated.
    """
    h, w = img.shape[:2]
    X, Y = _grid(h, w)
    map_x = (X + dx).astype(np.float32)
    map_y = (Y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation, borderMode=border)


def _sample_field(field: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(field, map_x.astype(np.float32), map_y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def invert_displacement(
    dx: np.ndarray, dy: np.ndarray, *, iters: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically invert a backward displacement field.

    Returns ``(idx, idy)`` such that warping by ``(dx, dy)`` and then by
    ``(idx, idy)`` recovers the original image. Solves the fixed point
    ``i(p) = -d(p + i(p))`` by iteration.
    """
    h, w = dx.shape
    X, Y = _grid(h, w)
    idx = -dx.copy()
    idy = -dy.copy()
    for _ in range(iters):
        sx = _sample_field(dx, X + idx, Y + idy)
        sy = _sample_field(dy, X + idx, Y + idy)
        idx = -sx
        idy = -sy
    return idx.astype(np.float32), idy.astype(np.float32)


@dataclass
class DisplacementField:
    """Accumulates per-layer backward displacements into one composite field."""

    dx: np.ndarray
    dy: np.ndarray

    @classmethod
    def zeros(cls, h: int, w: int) -> "DisplacementField":
        return cls(np.zeros((h, w), np.float32), np.zeros((h, w), np.float32))

    @property
    def shape(self) -> tuple[int, int]:
        return self.dx.shape

    def add(self, dx: np.ndarray, dy: np.ndarray) -> None:
        self.dx += dx
        self.dy += dy

    def is_identity(self, *, eps: float = 1e-6) -> bool:
        return bool(np.all(np.abs(self.dx) < eps) and np.all(np.abs(self.dy) < eps))

    def apply(self, img: np.ndarray) -> np.ndarray:
        return apply_displacement(img, self.dx, self.dy)

    def invert(self, *, iters: int = 12) -> tuple[np.ndarray, np.ndarray]:
        return invert_displacement(self.dx, self.dy, iters=iters)

    def magnitude(self) -> np.ndarray:
        return np.sqrt(self.dx ** 2 + self.dy ** 2)
