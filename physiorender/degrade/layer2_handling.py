"""PDE Layer 2 — handling degradation (plan §6 L2).

Where most of the realism comes from: wrinkles, folds, edge curl, and stains.
Geometric effects (wrinkles/folds/curl) accumulate into a shared
:class:`DisplacementField` so the whole page is remapped once; their brightness
shading is tied to a single global :class:`LightSource` so it reads as real
(plan §6 L2). Overlays (stain/pen/fingerprint) are painted on the flat paper
*before* warping, so they deform with it.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import imageutil as iu
from .light import LightSource
from .noise import fractal_noise
from .warp import DisplacementField


def _grid(h: int, w: int):
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    return np.meshgrid(xs, ys)


def _odd_profile(s: np.ndarray, width: float) -> np.ndarray:
    """Odd bump: zero at the crease and far away, peaks at +/- width.

    profile(s) = (s/width) * exp(-0.5 (s/width)^2). Gives lateral pinch
    (displacement) and a lit/shadow pair (brightness) across a crease.
    """
    u = s / max(1e-3, width)
    return (u * np.exp(-0.5 * u * u)).astype(np.float32)


def _signed_distance_field(h: int, w: int, rng: np.random.Generator,
                           *, wobble_px: float) -> tuple[np.ndarray, float, float]:
    """A random crease line across the page; return (signed-distance, perp_x, perp_y)."""
    X, Y = _grid(h, w)
    # Random point near center and a random axis angle.
    px = rng.uniform(0.25 * w, 0.75 * w)
    py = rng.uniform(0.25 * h, 0.75 * h)
    ang = rng.uniform(0, np.pi)
    ax, ay = np.cos(ang), np.sin(ang)
    perp_x, perp_y = -ay, ax
    s = (X - px) * perp_x + (Y - py) * perp_y
    if wobble_px > 0:
        s = s + wobble_px * fractal_noise(h, w, rng, cell_px=max(80, w / 4),
                                          normalize="signed")
    return s.astype(np.float32), float(perp_x), float(perp_y)


def _add_crease(
    img: np.ndarray,
    disp: DisplacementField,
    rng: np.random.Generator,
    light: LightSource,
    *,
    width_px: float,
    disp_amp_px: float,
    shade_amp: float,
    wobble_px: float,
    dark_line: float = 0.0,
) -> np.ndarray:
    """Add one crease (wrinkle or fold) to the image and displacement field."""
    h, w = img.shape[:2]
    s, perp_x, perp_y = _signed_distance_field(h, w, rng, wobble_px=wobble_px)
    profile = _odd_profile(s, width_px)

    # Lateral displacement (backward) perpendicular to the crease axis.
    disp.add(disp_amp_px * profile * perp_x, disp_amp_px * profile * perp_y)

    # Light-consistent shading: bright side faces the light.
    sign = light.shading_sign(perp_x, perp_y)
    gain = 1.0 + sign * shade_amp * light.grazing_strength * profile
    out = iu.apply_brightness(img, gain.astype(np.float32))

    # Optional dark crease line (a real fold leaves a thin printed seam).
    if dark_line > 0:
        seam = np.exp(-0.5 * (s / max(1.0, width_px * 0.15)) ** 2).astype(np.float32)
        out = iu.apply_brightness(out, (1.0 - dark_line * seam))
    return out


def add_wrinkles(img: np.ndarray, disp: DisplacementField, rng: np.random.Generator,
                 *, n: int, intensity: float, light: LightSource, ppm: float) -> np.ndarray:
    """3-8 soft wrinkles with light-consistent shading (plan §6 L2)."""
    out = img
    for _ in range(max(0, n)):
        width_px = rng.uniform(3.0, 8.0) * ppm
        out = _add_crease(
            out, disp, rng, light,
            width_px=width_px,
            disp_amp_px=intensity * rng.uniform(0.6, 1.4) * ppm,   # ~1mm lateral
            shade_amp=0.22 * intensity,
            wobble_px=rng.uniform(0.5, 1.5) * ppm,
        )
    return out


def add_folds(img: np.ndarray, disp: DisplacementField, rng: np.random.Generator,
              *, n: int, intensity: float, light: LightSource, ppm: float) -> np.ndarray:
    """0-2 hard folds: larger displacement + a thin dark crease (plan §6 L2)."""
    out = img
    for _ in range(max(0, n)):
        width_px = rng.uniform(6.0, 12.0) * ppm
        out = _add_crease(
            out, disp, rng, light,
            width_px=width_px,
            disp_amp_px=(1.0 + 1.3 * intensity) * rng.uniform(0.8, 1.2) * ppm,  # ~2-3x wrinkle
            shade_amp=0.22,
            wobble_px=rng.uniform(0.3, 1.0) * ppm,
            dark_line=0.16,
        )
    return out


def apply_edge_curl(img: np.ndarray, disp: DisplacementField, rng: np.random.Generator,
                    *, strength: float, light: LightSource) -> np.ndarray:
    """Document not lying flat: warp corners inward + radial brightness (plan §6 L2)."""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    X, Y = _grid(h, w)
    cx, cy = w / 2.0, h / 2.0
    nx = (X - cx) / cx
    ny = (Y - cy) / cy
    r2 = nx * nx + ny * ny                       # 0 center -> ~2 corners
    pull = (strength * 0.04 * r2).astype(np.float32)   # inward fraction near edges
    disp.add(pull * (X - cx), pull * (Y - cy))   # sample outward -> visually curls in

    # Radial brightness: edges catch more light or fall into shadow.
    sign = -1.0 if rng.random() < 0.5 else 1.0
    gain = 1.0 + sign * 0.12 * strength * light.grazing_strength * (r2 / 2.0)
    return iu.apply_brightness(img, gain.astype(np.float32))


def add_stain(img: np.ndarray, rng: np.random.Generator, *, opacity: float) -> np.ndarray:
    """Coffee-ring / watermark: feathered brownish ellipse with a darker rim."""
    h, w = img.shape[:2]
    X, Y = _grid(h, w)
    cx = rng.uniform(0.15 * w, 0.85 * w)
    cy = rng.uniform(0.15 * h, 0.85 * h)
    rx = rng.uniform(0.06, 0.16) * w
    ry = rx * rng.uniform(0.7, 1.3)
    d = np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2)
    fill = np.clip(1.0 - d, 0.0, 1.0) ** 2
    ring = np.exp(-0.5 * ((d - 1.0) / 0.12) ** 2)        # darker rim near edge
    alpha = (opacity * (0.5 * fill + ring)).astype(np.float32)
    alpha = np.clip(alpha, 0.0, opacity)
    stain_color = np.array([0.55, 0.40, 0.25], dtype=np.float32)  # brown
    out = img * (1.0 - alpha[..., None]) + stain_color[None, None, :] * alpha[..., None]
    return np.clip(out, 0.0, 1.0)


def add_pen_marks(img: np.ndarray, rng: np.random.Generator, *, n: int = 0) -> np.ndarray:
    """A few thin annotation strokes with slight pressure (width) variation."""
    if n <= 0:
        n = int(rng.integers(1, 4))
    out = np.ascontiguousarray(img)
    h, w = out.shape[:2]
    color = (0.10, 0.15, 0.45) if rng.random() < 0.6 else (0.05, 0.05, 0.05)  # blue/black
    for _ in range(n):
        x0, y0 = rng.uniform(0.1 * w, 0.9 * w), rng.uniform(0.1 * h, 0.9 * h)
        npts = int(rng.integers(2, 5))
        pts = [(x0, y0)]
        for _ in range(npts):
            x0 += rng.uniform(-0.12 * w, 0.12 * w)
            y0 += rng.uniform(-0.06 * h, 0.06 * h)
            pts.append((np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1)))
        poly = np.array(pts, dtype=np.int32)
        thick = int(rng.integers(2, 5))
        cv2.polylines(out, [poly], False, color, thickness=thick, lineType=cv2.LINE_AA)
    return out


def add_fingerprint(img: np.ndarray, rng: np.random.Generator, *, opacity: float = 0.12
                    ) -> np.ndarray:
    """Synthetic translucent fingerprint smudge (plan §6 L2)."""
    h, w = img.shape[:2]
    X, Y = _grid(h, w)
    cx = rng.uniform(0.2 * w, 0.8 * w)
    cy = rng.uniform(0.2 * h, 0.8 * h)
    rad = rng.uniform(0.05, 0.10) * w
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = np.clip(1.0 - d / rad, 0.0, 1.0)
    ridges = 0.5 + 0.5 * np.sin(d / max(2.0, rad * 0.08) + rng.uniform(0, 6.28))
    alpha = (opacity * mask * ridges).astype(np.float32)
    smudge = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.5)
    out = iu.apply_brightness(img, (1.0 - 0.5 * smudge))
    return out
