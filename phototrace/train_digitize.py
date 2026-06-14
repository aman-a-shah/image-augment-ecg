"""Training + evaluation for PhotoTrace Stage 3 (column-wise digitizer) (plan §9).

Trains :class:`ColumnDigitizer` with the morphology-weighted loss and evaluates
in the millivolt domain (correlation, DTW, R-peak F1) against a predict-the-mean
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from .digitize_data import StripDataset, y_to_mv
from .losses import morphology_weighted_loss
from .metrics import dtw_distance, peak_f1, signal_correlation
from .models import ColumnDigitizer


@dataclass
class DigitizeReport:
    train_losses: list[float] = field(default_factory=list)
    val_corr: float = 0.0          # mean Pearson corr (mV) pred vs gt
    baseline_corr: float = 0.0
    val_dtw: float = 0.0
    val_peak_f1: float = 0.0
    model: ColumnDigitizer | None = None

    @property
    def beats_baseline(self) -> bool:
        return self.val_corr > self.baseline_corr


def train_digitizer(
    train_ds: StripDataset,
    val_ds: StripDataset,
    *,
    epochs: int = 30,
    lr: float = 2e-3,
    batch_size: int = 16,
    device: str = "cpu",
    seed: int = 0,
    eval_fs: int = 500,
) -> DigitizeReport:
    torch.manual_seed(seed)
    model = ColumnDigitizer().to(device)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    report = DigitizeReport()

    # Baseline: predict the mean target column-y from training data.
    mean_y = torch.stack([train_ds[i]["target"] for i in range(len(train_ds))]).mean(0)

    for _ in range(epochs):
        model.train()
        running, nb = 0.0, 0
        for batch in tl:
            x = batch["image"].to(device)
            y = batch["target"].to(device)
            opt.zero_grad()
            loss = morphology_weighted_loss(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        report.train_losses.append(running / max(1, nb))

    # Evaluate in the mV domain.
    model.eval()
    corrs, base_corrs, dtws, f1s = [], [], [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            sample = val_ds[i]
            x = sample["image"].unsqueeze(0).to(device)
            a, b = sample["cal"].tolist()
            pred_y = model(x).squeeze(0).cpu().numpy()
            gt_y = sample["target"].numpy()
            pred_mv = y_to_mv(pred_y, a, b)
            gt_mv = y_to_mv(gt_y, a, b)
            base_mv = y_to_mv(mean_y.numpy(), a, b)
            corrs.append(signal_correlation(pred_mv, gt_mv))
            base_corrs.append(signal_correlation(base_mv, gt_mv))
            dtws.append(dtw_distance(pred_mv, gt_mv, band=64))
            f1s.append(peak_f1(pred_mv, gt_mv, eval_fs)["f1"])

    report.val_corr = float(np.mean(corrs))
    report.baseline_corr = float(np.mean(base_corrs))
    report.val_dtw = float(np.mean(dtws))
    report.val_peak_f1 = float(np.mean(f1s))
    report.model = model
    return report
