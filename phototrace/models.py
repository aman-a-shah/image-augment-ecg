"""PhotoTrace neural models (plan §9.1).

Lightweight, train-from-scratch CNNs (no pretrained downloads):
  - :class:`ConvEncoder` — shared backbone
  - :class:`CornerRegressor` — Stage 1, predicts 4 document corners
  - :class:`LeadDetector` — Stage 2, regresses 13 lead boxes (uses the fixed-layout prior)
  - :class:`ColumnDigitizer` — Stage 3, per-column trace position (plan §9.1 Option A)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ConvEncoder(nn.Module):
    """Small strided conv backbone -> global feature vector."""

    def __init__(self, in_ch: int = 3, widths=(16, 32, 64, 128), feat: int = 128) -> None:
        super().__init__()
        layers = []
        c = in_ch
        for w in widths:
            layers.append(_block(c, w))
            c = w
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.pool(x).flatten(1)


class CornerRegressor(nn.Module):
    """Stage 1: predict 4 document corners (8 values) in normalized [0,1]."""

    def __init__(self, image_ch: int = 3) -> None:
        super().__init__()
        self.encoder = ConvEncoder(image_ch)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 8), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class LeadDetector(nn.Module):
    """Stage 2: regress 13 lead boxes (52 values) in normalized [0,1]."""

    def __init__(self, n_boxes: int = 13, image_ch: int = 3) -> None:
        super().__init__()
        self.encoder = ConvEncoder(image_ch)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, n_boxes * 4), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class ColumnDigitizer(nn.Module):
    """Stage 3 Option A: per-column trace y-position from a lead-strip crop.

    Input:  [B, 1, H, W] grayscale lead strip.
    Output: [B, W] normalized trace y-position per pixel column (plan §9.1).

    Predicts a per-column heatmap over rows and takes a **soft-argmax** over
    height — this preserves vertical localization (a height mean-pool would throw
    away exactly the y-position we want) and is sub-pixel differentiable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)       # per-pixel trace-presence logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heat = self.head(self.features(x)).squeeze(1)   # [B, H, W]
        prob = torch.softmax(heat, dim=1)               # distribution over rows
        H = heat.shape[1]
        rows = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype)
        y = (prob * rows[None, :, None]).sum(dim=1)     # [B, W] soft-argmax
        return y
