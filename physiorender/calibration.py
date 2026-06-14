"""Calibrate PhysioRender against real photos (plan §11).

The fully-automatable part of Phase 4 calibration: extract distribution-level
image statistics (brightness, contrast, sharpness, colorfulness, edge density)
from a set of synthetic outputs and a set of real photos, then report where the
distributions diverge so the :class:`ParameterSampler` ranges can be tuned.

The perceptual study (humans guessing real vs synthetic, plan §11) is run
manually; this module covers the measurable gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_STAT_KEYS = ("brightness", "contrast", "sharpness", "colorfulness", "edge_density")


def image_stats(img: np.ndarray) -> dict[str, float]:
    """Compute distribution-relevant statistics from a uint8/float RGB image."""
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    gray = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    rg = img[..., 0] - img[..., 1]
    yb = 0.5 * (img[..., 0] + img[..., 1]) - img[..., 2]
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                         + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(lap.var()),
        "colorfulness": colorfulness,
        "edge_density": float((edges > 0).mean()),
    }


def stats_for_dir(path: str | Path, *, pattern: str = "*.jpg",
                  limit: int | None = None) -> list[dict[str, float]]:
    """Compute stats for every image in a directory."""
    files = sorted(Path(path).glob(pattern))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        out.append(image_stats(np.asarray(Image.open(f).convert("RGB"))))
    return out


@dataclass
class CalibrationReport:
    synth_summary: dict[str, tuple[float, float]]   # key -> (mean, std)
    real_summary: dict[str, tuple[float, float]]
    z_divergence: dict[str, float]                  # |mean_synth - mean_real| / std_real

    def worst_offenders(self, k: int = 3) -> list[str]:
        return sorted(self.z_divergence, key=self.z_divergence.get, reverse=True)[:k]

    def __str__(self) -> str:
        lines = ["Calibration report (synthetic vs real):"]
        for key in _STAT_KEYS:
            sm, ss = self.synth_summary[key]
            rm, rs = self.real_summary[key]
            lines.append(f"  {key:13s} synth={sm:7.3f}±{ss:5.3f}  "
                         f"real={rm:7.3f}±{rs:5.3f}  z={self.z_divergence[key]:5.2f}")
        lines.append(f"  worst offenders: {', '.join(self.worst_offenders())}")
        return "\n".join(lines)


def _summary(stats: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    out = {}
    for key in _STAT_KEYS:
        vals = np.array([s[key] for s in stats], dtype=np.float64)
        out[key] = (float(vals.mean()), float(vals.std() + 1e-9))
    return out


def compare(synth_stats: list[dict[str, float]],
            real_stats: list[dict[str, float]]) -> CalibrationReport:
    """Compare synthetic vs real stat distributions; flag the largest gaps."""
    synth = _summary(synth_stats)
    real = _summary(real_stats)
    z = {k: abs(synth[k][0] - real[k][0]) / real[k][1] for k in _STAT_KEYS}
    return CalibrationReport(synth_summary=synth, real_summary=real, z_divergence=z)
