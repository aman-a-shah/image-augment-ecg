"""PhotoTrace inference: photo -> unwarped document -> per-lead strip crops.

Composes Stage 1 (corners) and Stage 2 (lead boxes). Boxes predicted in photo
space are mapped through the unwarp homography so the returned crops are clean,
fronto-parallel lead strips ready for Stage 3 digitization (plan §9.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from . import geometry as geo
from .data import BBOX_KEYS


@dataclass
class GeometryOutput:
    corners_px: np.ndarray              # (4,2) predicted document corners
    unwarped: np.ndarray               # canonical fronto-parallel doc (RGB float)
    lead_crops: dict[str, np.ndarray]  # lead name -> strip crop (RGB float)
    lead_boxes_unwarped: dict[str, list[int]]


class GeometryPipeline:
    def __init__(self, corner_model, lead_model, *, image_size: int = 128,
                 out_size: tuple[int, int] = (1024, 700), device: str = "cpu") -> None:
        self.corner_model = corner_model.to(device).eval()
        self.lead_model = lead_model.to(device).eval()
        self.image_size = image_size
        self.out_size = out_size
        self.device = device

    def _prep(self, pil: Image.Image) -> torch.Tensor:
        small = pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(small, np.float32) / 255.0)
        return x.permute(2, 0, 1).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def process(self, pil: Image.Image) -> GeometryOutput:
        w, h = pil.size
        x = self._prep(pil)

        corners_n = self.corner_model(x).cpu().numpy().reshape(4, 2)
        corners_px = corners_n * np.array([w, h], np.float32)

        boxes_n = self.lead_model(x).cpu().numpy().reshape(len(BBOX_KEYS), 4)

        photo = np.asarray(pil.convert("RGB"), np.float32) / 255.0
        out_w, out_h = self.out_size
        H = geo.homography_to_unwarp(corners_px, out_w, out_h)
        unwarped = cv2.warpPerspective(photo, H, (out_w, out_h))

        lead_crops: dict[str, np.ndarray] = {}
        lead_boxes_uw: dict[str, list[int]] = {}
        for i, key in enumerate(BBOX_KEYS):
            x1, y1, x2, y2 = boxes_n[i] * np.array([w, h, w, h], np.float32)
            # Map the photo-space box corners into unwarped space.
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                           np.float32).reshape(1, -1, 2)
            uw = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
            bx1 = int(np.clip(uw[:, 0].min(), 0, out_w - 1))
            by1 = int(np.clip(uw[:, 1].min(), 0, out_h - 1))
            bx2 = int(np.clip(uw[:, 0].max(), 0, out_w - 1))
            by2 = int(np.clip(uw[:, 1].max(), 0, out_h - 1))
            if bx2 <= bx1 or by2 <= by1:
                continue
            lead_boxes_uw[key] = [bx1, by1, bx2, by2]
            lead_crops[key] = unwarped[by1:by2, bx1:bx2].copy()

        return GeometryOutput(corners_px=corners_px, unwarped=unwarped,
                              lead_crops=lead_crops, lead_boxes_unwarped=lead_boxes_uw)
