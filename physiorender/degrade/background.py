"""Realistic table-surface backgrounds, perspective-locked to the document.

The paper lies *on* a surface, so the surface shares the document's plane. We
generate the surface (and any stray papers/clutter) top-down in world/document
coordinates, then warp it to the photo with the **same homography** as the ECG —
so wood grain, table edges and other sheets recede with exactly the same
perspective as the ECG (plan §6 L3/L4 cohesion).

Surfaces: white / black / wood / metal / marble / grey-laminate. A soft contact
shadow is dropped under the document and under clutter, offset by the global
:class:`LightSource`, so everything sits together and is lit consistently.
"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .light import LightSource
from .noise import fractal_noise

# Light paper/card colours for stray sheets peeking in at the edges.
_CLUTTER_COLORS = (
    (0.97, 0.96, 0.93), (0.93, 0.90, 0.84), (0.96, 0.93, 0.70),  # white, manila, legal-yellow
    (0.85, 0.90, 0.96), (0.96, 0.86, 0.88), (0.88, 0.95, 0.88),  # pale blue/pink/green sticky
    (0.80, 0.80, 0.82),                                          # grey card
)


# ====================================================================== #
# Surface generators (top-down, world/texture space) -> float RGB [0,1]
# ====================================================================== #
def _oriented_highlight(th: int, tw: int, light: LightSource, *, width: float,
                        rng: np.random.Generator) -> np.ndarray:
    """A soft highlight stripe perpendicular to the light direction (gloss/sheen)."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    nx = (xx / tw) * 2 - 1
    ny = (yy / th) * 2 - 1
    lx, ly = light.direction
    proj = nx * lx + ny * ly
    offset = rng.uniform(-0.4, 0.4)
    return np.exp(-((proj - offset) ** 2) / (2 * width ** 2)).astype(np.float32)


def _surface(kind: str, th: int, tw: int, rng: np.random.Generator,
             light: LightSource) -> np.ndarray:
    if kind == "white":
        base = rng.uniform(0.80, 0.95)
        warm = rng.uniform(-0.02, 0.03)
        img = np.ones((th, tw, 3), np.float32) * np.array([base + warm, base, base - warm * 0.5])
        # very soft, low-frequency blotchiness only — no directional lines/scuffs
        img += 0.02 * fractal_noise(th, tw, rng, cell_px=tw / 3, normalize="signed")[..., None]

    elif kind == "black":
        base = rng.uniform(0.04, 0.15)
        img = np.ones((th, tw, 3), np.float32) * base
        img += 0.02 * fractal_noise(th, tw, rng, cell_px=tw / 5, normalize="signed")[..., None]
        gloss = _oriented_highlight(th, tw, light, width=rng.uniform(0.18, 0.4), rng=rng)
        img += rng.uniform(0.10, 0.30) * gloss[..., None]   # specular reflection

    elif kind == "wood":
        n_planks = int(rng.integers(4, 9))
        plank = tw / n_planks
        u = np.arange(tw)
        board_id = (u // plank).astype(int)
        seam = np.abs((u % plank) / plank - 0.0)
        seam = np.minimum(seam, 1 - seam)
        seam_dark = np.exp(-(seam / 0.03) ** 2)              # dark line at plank seams
        grain = fractal_noise(th, tw, rng, cell_px=tw / 3, normalize="signed")
        grain = cv2.GaussianBlur(grain, (1, 41), 0)          # stretch grain along board
        hue = rng.uniform(0.05, 0.09)
        sat = rng.uniform(0.35, 0.6)
        val = rng.uniform(0.35, 0.62)
        base_rgb = np.array(colorsys.hsv_to_rgb(hue, sat, val), np.float32)
        # per-board brightness variation
        board_var = (rng.uniform(0.85, 1.15, size=n_planks + 1))[np.clip(board_id, 0, n_planks)]
        img = np.ones((th, tw, 3), np.float32) * base_rgb[None, None, :]
        img *= (board_var[None, :, None]).astype(np.float32)
        img *= (1.0 + 0.35 * grain)[..., None]
        img *= (1.0 - 0.5 * seam_dark)[None, :, None]

    elif kind == "metal":
        base = rng.uniform(0.45, 0.7)
        img = np.ones((th, tw, 3), np.float32) * np.array([base * 0.98, base, base * 1.02])
        # smooth low-frequency tonal variation (no brushed streak lines)
        img += 0.05 * fractal_noise(th, tw, rng, cell_px=tw / 3, normalize="signed")[..., None]
        sheen = _oriented_highlight(th, tw, light, width=rng.uniform(0.2, 0.45), rng=rng)
        img += rng.uniform(0.12, 0.28) * sheen[..., None]

    else:  # grey laminate / desk
        base = rng.uniform(0.35, 0.6)
        img = np.ones((th, tw, 3), np.float32) * base
        speck = fractal_noise(th, tw, rng, cell_px=max(6, tw / 60), normalize="signed")
        img += 0.05 * speck[..., None]

    return np.clip(img, 0, 1).astype(np.float32)


# ====================================================================== #
# Clutter (stray papers peeking in at the edges) + contact shadows
# ====================================================================== #
def _rect_corners(cx, cy, hw, hh, ang):
    c, s = np.cos(ang), np.sin(ang)
    pts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], np.float32)
    rot = np.array([[c, -s], [s, c]], np.float32)
    return (pts @ rot.T) + np.array([cx, cy], np.float32)


def _add_clutter(tex: np.ndarray, rng: np.random.Generator, light: LightSource,
                 *, n: int) -> None:
    """Draw stray sheets near the edges (drawn in texture space => peek in after warp)."""
    th, tw = tex.shape[:2]
    lx, ly = light.direction
    for _ in range(n):
        side = rng.integers(0, 4)
        if side == 0:    # left
            cx, cy = rng.uniform(-0.1, 0.15) * tw, rng.uniform(0.1, 0.9) * th
        elif side == 1:  # right
            cx, cy = rng.uniform(0.85, 1.1) * tw, rng.uniform(0.1, 0.9) * th
        elif side == 2:  # top
            cx, cy = rng.uniform(0.1, 0.9) * tw, rng.uniform(-0.1, 0.15) * th
        else:            # bottom
            cx, cy = rng.uniform(0.1, 0.9) * tw, rng.uniform(0.85, 1.1) * th
        hw = rng.uniform(0.18, 0.42) * tw
        hh = rng.uniform(0.18, 0.42) * th
        ang = rng.uniform(-0.5, 0.5)
        corners = _rect_corners(cx, cy, hw, hh, ang)

        # contact shadow (offset away from light), then the sheet.
        off = np.array([-lx, -ly], np.float32) * 0.03 * max(tw, th)
        shadow = np.zeros((th, tw), np.float32)
        cv2.fillConvexPoly(shadow, (corners + off).astype(np.int32), 1.0)
        shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=0.02 * max(tw, th))
        tex *= (1.0 - 0.35 * shadow[..., None])

        color = np.array(_CLUTTER_COLORS[int(rng.integers(len(_CLUTTER_COLORS)))], np.float32)
        color = np.clip(color + rng.normal(0, 0.03, 3), 0, 1)
        cv2.fillConvexPoly(tex, corners.astype(np.int32), color.tolist())
        # subtle edge line so sheets read as distinct objects
        cv2.polylines(tex, [corners.astype(np.int32)], True,
                      (color * 0.8).tolist(), thickness=max(1, tw // 400),
                      lineType=cv2.LINE_AA)
        # occasional faint ruling on a sheet
        if rng.random() < 0.4:
            for t in np.linspace(0.2, 0.8, int(rng.integers(3, 7))):
                p0 = corners[0] * (1 - t) + corners[3] * t
                p1 = corners[1] * (1 - t) + corners[2] * t
                cv2.line(tex, p0.astype(np.int32), p1.astype(np.int32),
                         (color * 0.78).tolist(), 1, cv2.LINE_AA)


def _add_doc_shadow(tex: np.ndarray, light: LightSource, *, doc_w: float, doc_h: float,
                    wx0: float, wy0: float, sx: float, sy: float) -> None:
    """Soft contact shadow under the document (offset by light), in texture space."""
    lx, ly = light.direction
    off_w = np.array([-lx, -ly], np.float32) * 0.035 * doc_w
    world = np.array([[0, 0], [doc_w, 0], [doc_w, doc_h], [0, doc_h]], np.float32) + off_w
    tx = (world[:, 0] - wx0) / sx
    ty = (world[:, 1] - wy0) / sy
    pts = np.stack([tx, ty], axis=1).astype(np.int32)
    th, tw = tex.shape[:2]
    shadow = np.zeros((th, tw), np.float32)
    cv2.fillConvexPoly(shadow, pts, 1.0)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=0.02 * max(tw, th))
    tex *= (1.0 - 0.4 * shadow[..., None])


# ====================================================================== #
# Orchestrator
# ====================================================================== #
_SURFACES = ("white", "black", "wood", "metal", "grey")
_SURFACE_P = (0.28, 0.18, 0.28, 0.14, 0.12)


def build_scene_background(out_w: int, out_h: int, rng: np.random.Generator, *,
                           H: np.ndarray, H_inv: np.ndarray,
                           doc_w: int, doc_h: int, light: LightSource) -> np.ndarray:
    """Return a perspective-locked table surface (with clutter + doc shadow).

    The document itself is composited on top by the caller; this provides the
    surface it sits on, already sharing the document's homography and lighting.
    """
    # World region that the photo frame back-projects to (document-plane coords).
    photo_corners = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]],
                             np.float32).reshape(1, -1, 2)
    wc = cv2.perspectiveTransform(photo_corners, H_inv.astype(np.float32)).reshape(-1, 2)
    wx0, wx1 = float(wc[:, 0].min()), float(wc[:, 0].max())
    wy0, wy1 = float(wc[:, 1].min()), float(wc[:, 1].max())
    # Expand a little, and clamp the span so the surface never degenerates.
    wx0 -= 0.35 * doc_w; wx1 += 0.35 * doc_w
    wy0 -= 0.35 * doc_h; wy1 += 0.35 * doc_h
    cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
    half_x = min(max(wx1 - cx, 0.7 * doc_w), 3.0 * doc_w)
    half_y = min(max(wy1 - cy, 0.7 * doc_h), 3.0 * doc_h)
    wx0, wx1 = cx - half_x, cx + half_x
    wy0, wy1 = cy - half_y, cy + half_y
    span_x, span_y = wx1 - wx0, wy1 - wy0

    # Top-down texture canvas.
    tw = 1000
    th = int(np.clip(round(tw * span_y / span_x), 240, 1500))
    sx, sy = span_x / tw, span_y / th

    kind = str(rng.choice(_SURFACES, p=_SURFACE_P))
    tex = _surface(kind, th, tw, rng, light)
    _add_clutter(tex, rng, light, n=int(rng.integers(1, 5)))
    _add_doc_shadow(tex, light, doc_w=doc_w, doc_h=doc_h,
                    wx0=wx0, wy0=wy0, sx=sx, sy=sy)

    # Warp texture -> photo using the document's homography composed with the
    # texture-pixel -> world mapping. Same H as the ECG => same perspective.
    S = np.array([[sx, 0, wx0], [0, sy, wy0], [0, 0, 1]], np.float64)
    M = (H.astype(np.float64) @ S).astype(np.float32)
    bg = cv2.warpPerspective(tex, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return np.clip(bg, 0, 1).astype(np.float32)
