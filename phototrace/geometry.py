"""Homography / corner geometry helpers for PhotoTrace Stage 1 (plan §9.1).

The document occupies the full canvas in PhysioRender (out_size == doc size), so
the document corners in photo space are ``H · doc_rect`` where ``H = inv(H_inv)``.
Stage 1 predicts those 4 corners; from them we recover the unwarping homography.
"""

from __future__ import annotations

import cv2
import numpy as np

# Corner order: top-left, top-right, bottom-right, bottom-left.
def doc_rect(w: int, h: int) -> np.ndarray:
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)


def corners_from_h_inv(h_inv: np.ndarray, w: int, h: int) -> np.ndarray:
    """Document corners in photo-pixel space, given the stored ``homography_inv``."""
    H = np.linalg.inv(np.asarray(h_inv, dtype=np.float64))
    pts = doc_rect(w, h).reshape(1, -1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, H.astype(np.float32)).reshape(-1, 2)
    return out  # (4, 2) in pixels


def homography_to_unwarp(corners_px: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Homography mapping the photo to a fronto-parallel doc of size (out_w, out_h)."""
    src = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
    dst = doc_rect(out_w, out_h)
    return cv2.getPerspectiveTransform(src, dst)


def unwarp_image(photo: np.ndarray, corners_px: np.ndarray,
                 out_size: tuple[int, int]) -> np.ndarray:
    """Warp the photo to a canonical fronto-parallel document view."""
    out_w, out_h = out_size
    H = homography_to_unwarp(corners_px, out_w, out_h)
    return cv2.warpPerspective(photo, H, (out_w, out_h))
