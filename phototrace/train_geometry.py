"""Training for PhotoTrace Stage 1 (corners) and Stage 2 (lead boxes) (plan §9.1).

Both are normalized-coordinate regressions trained with smooth-L1. We always
report a "predict-the-mean" baseline so it's clear the model actually learned
geometry rather than memorizing the average layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .data import ECGPhotoDataset, collate_geometry


@dataclass
class TrainReport:
    train_losses: list[float] = field(default_factory=list)
    val_mae: float = 0.0
    baseline_mae: float = 0.0
    model: nn.Module | None = None

    @property
    def beats_baseline(self) -> bool:
        return self.val_mae < self.baseline_mae


def split_dataset(root: str, *, image_size: int = 128, val_frac: float = 0.25,
                  seed: int = 0) -> tuple[Subset, Subset]:
    ds = ECGPhotoDataset(root, image_size=image_size)
    n = len(ds)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return Subset(ds, train_idx), Subset(ds, val_idx)


def _target_mean(loader: DataLoader, key: str) -> torch.Tensor:
    total, count = None, 0
    for batch in loader:
        t = batch[key]
        total = t.sum(0) if total is None else total + t.sum(0)
        count += t.shape[0]
    return total / max(1, count)


def train_regressor(
    model: nn.Module,
    train_ds: Subset,
    val_ds: Subset,
    *,
    target_key: str,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 8,
    device: str = "cpu",
    seed: int = 0,
) -> TrainReport:
    """Train a coordinate regressor; return losses + val/baseline MAE."""
    torch.manual_seed(seed)
    model = model.to(device)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    collate_fn=collate_geometry)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                    collate_fn=collate_geometry)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    report = TrainReport()

    for _ in range(epochs):
        model.train()
        running = 0.0
        nb = 0
        for batch in tl:
            x = batch["image"].to(device)
            y = batch[target_key].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        report.train_losses.append(running / max(1, nb))

    # Evaluation: model MAE vs predict-the-train-mean baseline.
    baseline = _target_mean(tl, target_key).to(device)
    model.eval()
    abs_err, base_err, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in vl:
            x = batch["image"].to(device)
            y = batch[target_key].to(device)
            pred = model(x)
            abs_err += (pred - y).abs().sum().item()
            base_err += (baseline.unsqueeze(0) - y).abs().sum().item()
            n += y.numel()
    report.val_mae = abs_err / max(1, n)
    report.baseline_mae = base_err / max(1, n)
    report.model = model
    return report
