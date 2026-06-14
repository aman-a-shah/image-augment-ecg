"""Stage 3 training data: lead-strip crops -> per-column trace position (plan §9.1).

Builds perfectly-aligned supervision by cropping each lead strip from the *clean*
rendered ECG and computing the target trace y-position per column directly from
the rendering transform (signal mV -> pixel y). Light per-strip degradation gives
input variety. The deployment path feeds unwarped real strips through the same
model; the column->mV calibration (a, b) makes predictions physical.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from physiorender import config
from physiorender.ingest.record import ECGRecord
from physiorender.render import build_standard_12lead
from physiorender.render.renderer import RenderResult


@dataclass
class StripExample:
    image: np.ndarray      # grayscale crop [h, w] in [0,1]
    target_y: np.ndarray   # per-column normalized trace y in [0,1], length w
    a: float               # mV = a - b * y_norm   (column->mV calibration)
    b: float
    lead: str


def build_strip_examples(record: ECGRecord, render: RenderResult) -> list[StripExample]:
    """One StripExample per grid lead panel, cropped from the clean render."""
    clean = np.asarray(render.image.convert("L"), np.float32) / 255.0
    H_img = clean.shape[0]
    layout = build_standard_12lead(rhythm=True)
    ppm = config.mm_to_px(1.0, render.dpi)
    gain_ppm = render.gain_mm_mv * ppm

    examples: list[StripExample] = []
    for panel in layout.panels:
        if panel.lead not in record.leads or panel.bbox_key not in render.lead_bboxes:
            continue
        x1, y1, x2, y2 = render.lead_bboxes[panel.bbox_key]
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crop = clean[y1:y2, x1:x2]
        crop_h = y2 - y1
        width = x2 - x1
        baseline_px = panel.baseline_y_mm * ppm

        lead = record.leads[panel.lead]
        fs = lead.sample_rate_hz
        target = np.zeros(width, np.float32)
        for c in range(width):
            t = panel.t_start_s + (c / width) * panel.t_dur_s
            si = min(lead.n_samples - 1, int(round(t * fs)))
            mv = float(lead.signal_mv[si])
            y_px = baseline_px - mv * gain_ppm
            target[c] = np.clip((y_px - y1) / crop_h, 0.0, 1.0)

        # mV = (baseline - (y1 + y_norm*crop_h)) / gain_ppm = a - b*y_norm
        a = (baseline_px - y1) / gain_ppm
        b = crop_h / gain_ppm
        examples.append(StripExample(image=crop, target_y=target, a=a, b=b,
                                     lead=panel.lead))
    return examples


def y_to_mv(y_norm: np.ndarray, a: float, b: float) -> np.ndarray:
    """Convert normalized column-y predictions back to millivolts."""
    return a - b * np.asarray(y_norm, np.float32)


class StripDataset(Dataset):
    """Resized lead strips + per-column targets, with light input degradation."""

    def __init__(self, examples: list[StripExample], *, n_variants: int = 6,
                 out_h: int = 64, out_w: int = 512, seed: int = 0,
                 augment: bool = True) -> None:
        self.base = examples
        self.n_variants = n_variants
        self.out_h, self.out_w = out_h, out_w
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        # Materialize (example_idx, variant_seed) pairs for determinism.
        self.index = [(i, v) for i in range(len(examples)) for v in range(n_variants)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        i, v = self.index[k]
        ex = self.base[i]
        rng = np.random.default_rng((i + 1) * 1000 + v)

        img = cv2.resize(ex.image, (self.out_w, self.out_h),
                         interpolation=cv2.INTER_AREA)
        if self.augment:
            img = img * rng.uniform(0.85, 1.05) + rng.uniform(-0.05, 0.05)
            img = img + rng.normal(0, 0.03, img.shape).astype(np.float32)
            img = np.clip(img, 0, 1)

        # Resample target to out_w.
        xp = np.linspace(0, 1, ex.target_y.shape[0])
        xq = np.linspace(0, 1, self.out_w)
        target = np.interp(xq, xp, ex.target_y).astype(np.float32)

        x = torch.from_numpy(img).unsqueeze(0).float()       # [1, H, W]
        y = torch.from_numpy(target).float()                 # [W]
        cal = torch.tensor([ex.a, ex.b], dtype=torch.float32)
        return {"image": x, "target": y, "cal": cal}
