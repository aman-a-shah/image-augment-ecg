"""End-to-end PhotoTrace digitization (plan §9, §12 Phase 6).

photo -> (Stage 1 unwarp) -> per-lead strips (canonical layout) -> (Stage 3
column digitizer) -> calibrated mV signals. The canonical layout gives exact
column->mV calibration once the page is fronto-parallel, so signal extraction
doesn't depend on Stage-2 box regression accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from physiorender.render import build_standard_12lead

from . import geometry as geo
from .digitize_data import y_to_mv
from .postprocess import remove_baseline_wander, resample_signal


@dataclass
class PanelGeom:
    lead: str
    key: str
    bbox: tuple[int, int, int, int]
    a: float
    b: float
    t_start_s: float
    t_dur_s: float


def panel_geometry(w: int, h: int, gain_mm_mv: int) -> list[PanelGeom]:
    """Crop boxes + column->mV calibration for a canonical doc of size (w, h)."""
    layout = build_standard_12lead(rhythm=True)
    ppm = w / layout.page_w_mm
    gain_ppm = gain_mm_mv * ppm
    out = []
    for p in layout.panels:
        x1 = int(round(p.x_mm * ppm)); y1 = int(round(p.y_mm * ppm))
        x2 = int(round((p.x_mm + p.w_mm) * ppm)); y2 = int(round((p.y_mm + p.h_mm) * ppm))
        baseline_px = p.baseline_y_mm * ppm
        a = (baseline_px - y1) / gain_ppm
        b = (y2 - y1) / gain_ppm
        out.append(PanelGeom(p.lead, p.bbox_key, (x1, y1, x2, y2), a, b,
                             p.t_start_s, p.t_dur_s))
    return out


class DigitizationPipeline:
    """Full photo -> signals pipeline."""

    def __init__(self, digitizer, *, corner_model=None, gain_mm_mv: int = 10,
                 target_fs: int = 500, strip_size: tuple[int, int] = (64, 512),
                 device: str = "cpu") -> None:
        self.digitizer = digitizer.to(device).eval()
        self.corner_model = corner_model.to(device).eval() if corner_model else None
        self.gain = gain_mm_mv
        self.target_fs = target_fs
        self.sh, self.sw = strip_size
        self.device = device

    def _canonical(self, pil: Image.Image) -> np.ndarray:
        """Return a fronto-parallel grayscale document in [0,1]."""
        if self.corner_model is None:
            return np.asarray(pil.convert("L"), np.float32) / 255.0
        w, h = pil.size
        small = pil.resize((128, 128), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(small, np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            corners = self.corner_model(x).cpu().numpy().reshape(4, 2) * [w, h]
        layout = build_standard_12lead()
        out_w = 1024
        out_h = int(round(out_w * layout.page_h_mm / layout.page_w_mm))
        gray = np.asarray(pil.convert("L"), np.float32) / 255.0
        return geo.unwarp_image(gray, corners.astype(np.float32), (out_w, out_h))

    @torch.no_grad()
    def digitize(self, pil: Image.Image, *, baseline_correct: bool = False
                 ) -> dict[str, np.ndarray]:
        """Return {bbox_key: mV signal} for all 12 grid leads + the rhythm strip."""
        canon = self._canonical(pil)
        h, w = canon.shape[:2]
        signals: dict[str, np.ndarray] = {}
        for g in panel_geometry(w, h, self.gain):
            x1, y1, x2, y2 = g.bbox
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            strip = canon[y1:y2, x1:x2]
            inp = cv2.resize(strip, (self.sw, self.sh), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().to(self.device)
            y_norm = self.digitizer(t).squeeze(0).cpu().numpy()
            mv = y_to_mv(y_norm, g.a, g.b)
            n = max(2, int(round(g.t_dur_s * self.target_fs)))
            mv = resample_signal(mv, len(mv), n)
            if baseline_correct:
                mv = remove_baseline_wander(mv, self.target_fs)
            signals[g.key] = mv
        return signals
