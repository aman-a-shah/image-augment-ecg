"""PDE Layer 3 — camera & capture artifacts (plan §6 L3).

Capture physics order (the thing that makes the data useful, plan §6 L3 / §14):
**lens distortion → perspective warp → blur**. Lens distortion is pre-warp; blur
is post-warp. Perspective is the highest-impact artifact and exports ``H_inv`` as
the supervision signal for PhotoTrace Stage 1 (plan §9.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import imageutil as iu


# ====================================================================== #
# Lens distortion (barrel) — applied BEFORE perspective
# ====================================================================== #
@dataclass(frozen=True)
class LensModel:
    k1: float
    k2: float
    cx: float
    cy: float
    norm: float  # half-diagonal, so radius is ~1 at the corners

    def _gain(self, r2: np.ndarray | float):
        return 1.0 + self.k1 * r2 + self.k2 * r2 * r2


def make_lens(w: int, h: int, k1: float, k2: float = 0.0) -> LensModel:
    return LensModel(k1=k1, k2=k2, cx=w / 2.0, cy=h / 2.0,
                     norm=0.5 * math.hypot(w, h))


def apply_barrel(img: np.ndarray, lens: LensModel) -> np.ndarray:
    """Barrel-distort an image: r' = r(1 + k1 r^2 + k2 r^4) (plan §6 L3)."""
    if lens.k1 == 0 and lens.k2 == 0:
        return img
    h, w = img.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    r2 = ((X - lens.cx) ** 2 + (Y - lens.cy) ** 2) / (lens.norm ** 2)
    g = lens._gain(r2)
    map_x = (lens.cx + (X - lens.cx) * g).astype(np.float32)
    map_y = (lens.cy + (Y - lens.cy) * g).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def distort_point(pt: tuple[float, float], lens: LensModel,
                  *, iters: int = 8) -> tuple[float, float]:
    """Forward-map an undistorted point to its location in the distorted image.

    Inverts the remap relation S(d)=p by fixed-point iteration.
    """
    if lens.k1 == 0 and lens.k2 == 0:
        return pt
    px, py = pt
    dx, dy = px, py
    for _ in range(iters):
        r2 = ((dx - lens.cx) ** 2 + (dy - lens.cy) ** 2) / (lens.norm ** 2)
        g = lens._gain(r2)
        dx = lens.cx + (px - lens.cx) / g
        dy = lens.cy + (py - lens.cy) / g
    return float(dx), float(dy)


# ====================================================================== #
# Perspective (homography from 3D tilt) — exports H_inv
# ====================================================================== #
def _rot_matrix(ax: float, ay: float, az: float) -> np.ndarray:
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project_corners(w: int, h: int, tilt_x_deg: float, tilt_y_deg: float,
                    rotation_deg: float, *, focal: float | None = None) -> np.ndarray:
    """Project the 4 doc corners through a 3D tilt; return centered dst corners."""
    if focal is None:
        focal = 1.2 * max(w, h)
    R = _rot_matrix(math.radians(tilt_y_deg), math.radians(tilt_x_deg),
                    math.radians(rotation_deg))
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    dst = []
    for x, y in src:
        P = R @ np.array([x - w / 2.0, y - h / 2.0, 0.0])
        Z = P[2] + focal
        dst.append([focal * P[0] / Z, focal * P[1] / Z])
    return np.array(dst, dtype=np.float64)


@dataclass
class PerspectiveResult:
    image: np.ndarray          # warped document on a transparent canvas
    mask: np.ndarray           # [H,W] coverage of the document (1 inside)
    H: np.ndarray              # 3x3 doc-px -> photo-px
    H_inv: np.ndarray          # 3x3 photo-px -> doc-px  (Stage-1 supervision)


def frame_and_warp(doc: np.ndarray, rng: np.random.Generator, *,
                   tilt_x_deg: float, tilt_y_deg: float, rotation_deg: float,
                   crop_margin: float, out_size: tuple[int, int]) -> PerspectiveResult:
    """Tilt, scale to occupy most of the frame, and place at a random offset.

    Returns the warped doc + coverage mask + the homography and its inverse.
    """
    h, w = doc.shape[:2]
    Wo, Ho = out_size
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = project_corners(w, h, tilt_x_deg, tilt_y_deg, rotation_deg)

    # Scale so the document occupies (1 - 2*crop_margin) of the frame.
    occupy = max(0.55, 1.0 - 2.0 * crop_margin)
    bw = dst[:, 0].max() - dst[:, 0].min()
    bh = dst[:, 1].max() - dst[:, 1].min()
    s = occupy * min(Wo / bw, Ho / bh)
    dst *= s

    # Random placement inside the canvas.
    bx0, by0 = dst[:, 0].min(), dst[:, 1].min()
    free_x = Wo - (dst[:, 0].max() - bx0)
    free_y = Ho - (dst[:, 1].max() - by0)
    ox = rng.uniform(0, max(0.0, free_x)) - bx0
    oy = rng.uniform(0, max(0.0, free_y)) - by0
    dst[:, 0] += ox
    dst[:, 1] += oy

    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    H_inv = np.linalg.inv(H)
    warped = cv2.warpPerspective(doc, H, (Wo, Ho), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), H, (Wo, Ho),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return PerspectiveResult(image=warped, mask=mask, H=H, H_inv=H_inv)


def transform_bbox(bbox: list[int], lens: LensModel, H: np.ndarray,
                   out_size: tuple[int, int]) -> list[int]:
    """Map a doc-space bbox through lens distortion then homography → photo aabb."""
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    distorted = np.array([distort_point(c, lens) for c in corners], dtype=np.float32)
    pts = cv2.perspectiveTransform(distorted.reshape(1, -1, 2), H).reshape(-1, 2)
    Wo, Ho = out_size
    nx1 = int(np.clip(pts[:, 0].min(), 0, Wo - 1))
    ny1 = int(np.clip(pts[:, 1].min(), 0, Ho - 1))
    nx2 = int(np.clip(pts[:, 0].max(), 0, Wo - 1))
    ny2 = int(np.clip(pts[:, 1].max(), 0, Ho - 1))
    return [nx1, ny1, max(nx1 + 1, nx2), max(ny1 + 1, ny2)]


# ====================================================================== #
# Background, blur, lens dirt
# ====================================================================== #
def make_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Procedural surface behind the document — many varied schemes."""
    from .noise import fractal_noise
    kind = rng.choice(["solid", "noisy", "wood", "cloth", "gradient", "desk", "dark"],
                      p=[0.14, 0.18, 0.18, 0.16, 0.14, 0.12, 0.08])
    # Random base colour across a wide gamut (woods, fabrics, desks, darks).
    hue = rng.uniform(0, 1)
    sat = rng.uniform(0.0, 0.5)
    val = rng.uniform(0.08, 0.7) if kind != "dark" else rng.uniform(0.02, 0.18)
    import colorsys
    base = np.array(colorsys.hsv_to_rgb(hue, sat, val), np.float32)
    bg = np.ones((h, w, 3), np.float32) * base

    if kind in ("noisy", "cloth", "desk"):
        cell = w / (10 if kind != "cloth" else 40)
        tex = fractal_noise(h, w, rng, cell_px=max(10, cell), normalize="signed")
        bg = np.clip(bg + rng.uniform(0.06, 0.16) * tex[..., None], 0, 1)
    if kind == "wood":
        streak = fractal_noise(h, w, rng, cell_px=max(6, w / 70), normalize="signed")
        grain = fractal_noise(h, w, rng, cell_px=max(30, w / 6), normalize="signed")
        bg = np.clip(bg + 0.09 * streak[..., None] * np.array([0.6, 0.4, 0.2])
                     + 0.06 * grain[..., None], 0, 1)
    if kind == "cloth":  # fine cross-weave
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        weave = (np.sin(xx / rng.uniform(2, 5)) + np.sin(yy / rng.uniform(2, 5)))
        bg = np.clip(bg + 0.03 * weave[..., None], 0, 1)
    if kind == "gradient":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ang = rng.uniform(0, 2 * np.pi)
        g = (xx * np.cos(ang) + yy * np.sin(ang))
        g = (g - g.min()) / (np.ptp(g) + 1e-6)
        bg = np.clip(bg + (rng.uniform(-0.2, 0.2)) * g[..., None], 0, 1)
    return bg.astype(np.float32)


def composite_on_background(doc_warped: np.ndarray, mask: np.ndarray,
                            bg: np.ndarray) -> np.ndarray:
    a = np.clip(mask, 0, 1)[..., None]
    return (doc_warped * a + bg * (1.0 - a)).astype(np.float32)


def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    length = max(3, length | 1)
    k = np.zeros((length, length), np.float32)
    c = length // 2
    a = math.radians(angle_deg)
    for t in range(length):
        x = int(round(c + (t - c) * math.cos(a)))
        y = int(round(c + (t - c) * math.sin(a)))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    s = k.sum()
    return k / s if s > 0 else k


def _handshake_kernel(radius: int, rng: np.random.Generator) -> np.ndarray:
    size = max(3, (radius * 2 + 1) | 1)
    k = np.zeros((size, size), np.float32)
    x = y = size // 2
    for _ in range(size * 2):
        x = int(np.clip(x + rng.integers(-1, 2), 0, size - 1))
        y = int(np.clip(y + rng.integers(-1, 2), 0, size - 1))
        k[y, x] += 1.0
    k = cv2.GaussianBlur(k, (3, 3), 0)
    s = k.sum()
    return k / s if s > 0 else k


def apply_blur(img: np.ndarray, rng: np.random.Generator, *,
               blur_type: str, strength: float, ppm: float) -> np.ndarray:
    """Apply one blur mode AFTER the perspective warp (plan §6 L3)."""
    if blur_type == "none" or strength <= 0:
        return img
    if blur_type == "motion":
        length = int((0.15 + 0.7 * strength) * ppm)
        k = _motion_kernel(length, rng.uniform(0, 180))
        return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT_101)
    if blur_type == "handshake":
        k = _handshake_kernel(int((0.12 + 0.5 * strength) * ppm), rng)
        return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT_101)
    # defocus: uniform isotropic Gaussian. The document and the table it lies on
    # are coplanar (same focal depth), so blur must be uniform across the whole
    # frame — no spatially-varying / shallow depth-of-field falloff.
    sigma = (0.08 + 0.3 * strength) * ppm
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def apply_lens_dirt(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Translucent out-of-focus smears from a dirty lens (plan §6 L3)."""
    h, w = img.shape[:2]
    blob = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(2, 6))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        rad = rng.uniform(0.05, 0.18) * w
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        blob += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad ** 2)))
    blob = cv2.GaussianBlur(blob, (0, 0), sigmaX=0.02 * w)
    blob = np.clip(blob, 0, 1)[..., None]
    haze = rng.uniform(0.06, 0.15)
    # Low-contrast translucent veil where dirt sits.
    return np.clip(img * (1 - haze * blob) + haze * blob * 0.85, 0, 1).astype(np.float32)
