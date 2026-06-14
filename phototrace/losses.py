"""Digitization losses (plan §9.2).

The headline idea (plan §9.2): MSE alone produces a model that's accurate on the
flat isoelectric baseline and wrong on the QRS complex — the opposite of clinical
priority. The morphology-weighted loss up-weights samples near QRS peaks.

    L = L_pos + 0.1 * L_freq + 0.5 * L_morph
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def qrs_peak_mask(gt_y: torch.Tensor, *, prominence: float = 0.08) -> torch.Tensor:
    """Soft mask (1 near QRS peaks, 0 elsewhere) from the ground-truth column-y.

    QRS R-waves are large upward deflections -> small y (trace near top). We find
    prominent local minima of y and dilate them. Operates per-row on [B, W].
    """
    # Amplitude where up = positive: invert y around its per-row mean.
    amp = gt_y.mean(dim=1, keepdim=True) - gt_y               # [B, W]
    # Local maxima of amplitude.
    left = F.pad(amp[:, :-1], (1, 0), value=-1e9)
    right = F.pad(amp[:, 1:], (0, 1), value=-1e9)
    is_peak = (amp >= left) & (amp >= right) & (amp > prominence)
    mask = is_peak.float()
    # Dilate by a small window so the QRS neighbourhood is weighted.
    mask = F.max_pool1d(mask.unsqueeze(1), kernel_size=9, stride=1, padding=4).squeeze(1)
    return mask


def morphology_weighted_loss(pred_y: torch.Tensor, gt_y: torch.Tensor, *,
                             w_freq: float = 0.1, w_morph: float = 0.5,
                             peak_weight: float = 5.0) -> torch.Tensor:
    """Combined position + frequency + morphology loss (plan §9.2)."""
    l_pos = F.smooth_l1_loss(pred_y, gt_y)

    # Frequency-domain loss preserves clinical rhythm features.
    pf = torch.fft.rfft(pred_y, dim=1).abs()
    gf = torch.fft.rfft(gt_y, dim=1).abs()
    l_freq = F.mse_loss(pf, gf) / (gf.pow(2).mean() + 1e-6)

    # Morphology: weight errors near QRS peaks.
    mask = qrs_peak_mask(gt_y)
    weight = 1.0 + peak_weight * mask
    l_morph = (weight * (pred_y - gt_y) ** 2).mean()

    return l_pos + w_freq * l_freq + w_morph * l_morph
