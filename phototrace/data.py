"""PhotoTrace dataset over PhysioRender output (plan §9, §10).

Reads a generated dataset (manifest.jsonl + images + metadata) and yields
supervision for all three PhotoTrace stages:

  - Stage 1: document corners (from ``homography_inv``), normalized [0,1]
  - Stage 2: 13 lead bounding boxes, normalized [0,1]
  - Stage 3: per-lead ground-truth waveform (when present in metadata)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from physiorender.config import STANDARD_LEADS
from physiorender.metadata import ECGMetadata

from .geometry import corners_from_h_inv

# Fixed ordering for the 13 regressed boxes (12 leads + rhythm strip).
BBOX_KEYS: list[str] = list(STANDARD_LEADS) + ["II_rhythm"]


class ECGPhotoDataset(Dataset):
    """Augmented ECG photos with geometry + signal supervision."""

    def __init__(self, root: str | Path, *, image_size: int = 128,
                 want_signals: bool = False) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.want_signals = want_signals
        manifest = self.root / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"no manifest.jsonl in {root}")
        with open(manifest, encoding="utf-8") as fh:
            self.items = [json.loads(line) for line in fh if line.strip()]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        img = Image.open(self.root / item["image"]).convert("RGB")
        w, h = img.size
        meta = ECGMetadata.load(str(self.root / item["metadata"]))

        # Image -> CHW float tensor in [0,1].
        small = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(small, np.float32) / 255.0).permute(2, 0, 1)

        # Stage 1: corners normalized by image size.
        corners = corners_from_h_inv(np.array(meta.homography_inv), w, h)
        corners_n = corners / np.array([w, h], np.float32)
        corners_t = torch.from_numpy(corners_n.reshape(-1).astype(np.float32))

        # Stage 2: 13 boxes normalized by image size.
        boxes = np.zeros((len(BBOX_KEYS), 4), np.float32)
        for i, key in enumerate(BBOX_KEYS):
            if key in meta.layout.lead_bboxes:
                x1, y1, x2, y2 = meta.layout.lead_bboxes[key]
                boxes[i] = [x1 / w, y1 / h, x2 / w, y2 / h]
        boxes_t = torch.from_numpy(boxes.reshape(-1))

        out = {"image": x, "corners": corners_t, "boxes": boxes_t,
               "size": torch.tensor([w, h], dtype=torch.float32), "idx": idx}

        if self.want_signals:
            sigs = {name: torch.tensor(meta.leads[name].signal_mv, dtype=torch.float32)
                    for name in meta.leads if meta.leads[name].signal_mv}
            out["signals"] = sigs
        return out


def collate_geometry(batch: list[dict]) -> dict:
    """Collate Stage 1/2 tensors (signals dropped — variable per-lead length)."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "corners": torch.stack([b["corners"] for b in batch]),
        "boxes": torch.stack([b["boxes"] for b in batch]),
        "size": torch.stack([b["size"] for b in batch]),
    }
