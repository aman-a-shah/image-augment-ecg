"""Domain-gap evaluation harness (plan §11, §14 — the headline result).

The project's headline claim is: an existing *digital* ECG model fails on phone
photos, and PhotoTrace closes that gap. That requires the external digital model,
which isn't available here — so this module provides the **harness** plus a
concrete proxy:

  - A pluggable ``digital_model`` callable: signals dict -> prediction.
  - A default proxy model (heart rate from lead II), since HR is recoverable
    from any correct digitization and sensitive to a failed one.
  - ``compare`` measures the proxy's error when fed (a) a naive non-learned
    digitization vs (b) the PhotoTrace digitization — the miniature domain gap.

Swap in the real digital model + real photos to produce the paper's number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np

from .digitize_data import y_to_mv
from .metrics import heart_rate_bpm, signal_correlation
from .pipeline import DigitizationPipeline, panel_geometry

# A digital model maps {lead: signal_mv} -> some prediction (label, measurement…).
DigitalModel = Callable[[dict[str, np.ndarray]], float]


def hr_digital_model(signals: dict[str, np.ndarray], *, fs: int = 500) -> float:
    """Proxy 'digital model': heart rate (bpm) from the rhythm strip / lead II."""
    sig = signals.get("II_rhythm")
    if sig is None:
        sig = signals.get("II")
    if sig is None or len(sig) < 4:
        return 0.0
    return heart_rate_bpm(sig, fs)


def naive_digitize(canon_gray: np.ndarray, gain_mm_mv: int = 10,
                   target_fs: int = 500) -> dict[str, np.ndarray]:
    """Non-learned baseline: trace y = darkest row per column (a 'filter stack')."""
    h, w = canon_gray.shape[:2]
    out: dict[str, np.ndarray] = {}
    for g in panel_geometry(w, h, gain_mm_mv):
        x1, y1, x2, y2 = g.bbox
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        strip = canon_gray[y1:y2, x1:x2]
        y_idx = strip.argmin(axis=0).astype(np.float32)      # darkest row
        y_norm = y_idx / max(1, strip.shape[0] - 1)
        out[g.key] = y_to_mv(y_norm, g.a, g.b)
    return out


@dataclass
class DomainGapResult:
    naive_error: float
    pipeline_error: float
    naive_corr: float
    pipeline_corr: float

    @property
    def improvement(self) -> float:
        return self.naive_error - self.pipeline_error


def compare(
    samples: Sequence[tuple[np.ndarray, dict[str, np.ndarray]]],
    pipeline: DigitizationPipeline,
    *,
    digital_model: DigitalModel = hr_digital_model,
    fs: int = 500,
) -> DomainGapResult:
    """Compare proxy-model error: naive digitization vs PhotoTrace digitization.

    ``samples``: list of (canonical_gray_image, ground_truth_signals). The GT
    digital-model output is computed from the ground-truth signals.
    """
    from PIL import Image

    naive_errs, pipe_errs, naive_corrs, pipe_corrs = [], [], [], []
    for gray, gt_signals in samples:
        gt_pred = digital_model(gt_signals)

        naive_sig = naive_digitize(gray, pipeline.gain, fs)
        pipe_sig = pipeline.digitize(Image.fromarray((gray * 255).astype(np.uint8)))

        naive_errs.append(abs(digital_model(naive_sig) - gt_pred))
        pipe_errs.append(abs(digital_model(pipe_sig) - gt_pred))

        # Track lead-II waveform fidelity too.
        for store, sig in ((naive_corrs, naive_sig), (pipe_corrs, pipe_sig)):
            key = "II" if "II" in gt_signals else next(iter(gt_signals))
            if key in sig and key in gt_signals:
                n = min(len(sig[key]), len(gt_signals[key]))
                store.append(signal_correlation(sig[key][:n], gt_signals[key][:n]))

    return DomainGapResult(
        naive_error=float(np.mean(naive_errs)),
        pipeline_error=float(np.mean(pipe_errs)),
        naive_corr=float(np.mean(naive_corrs)) if naive_corrs else 0.0,
        pipeline_corr=float(np.mean(pipe_corrs)) if pipe_corrs else 0.0,
    )
