"""ECGMetadata — the metadata contract emitted alongside every augmented image.

This is the spine of the whole system (plan §8). PhysioRender writes it (Phase 4);
PhotoTrace reads it as supervision (Phases 6-7). Locking this schema in Phase 0
keeps writers and readers from drifting.

Pure stdlib by design.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .params import AugmentationParams

# A 4-tuple [x1, y1, x2, y2] in image pixel coordinates.
BBox = list  # runtime alias; documented shape is [int, int, int, int]
# A 3x3 row-major homography.
Matrix3x3 = list  # documented shape is list[list[float]] (3x3)


@dataclass
class LeadData:
    """Ground-truth signal for one lead (plan §8)."""

    signal_mv: list[float]
    sample_rate_hz: int


@dataclass
class Layout:
    """How the leads were laid out on the rendered page (plan §8)."""

    template: str                       # e.g. "standard_12lead_rhythm"
    paper_speed_mm_s: int               # 25 or 50
    gain_mm_mv: int                     # 10 or 5
    # lead name -> [x1, y1, x2, y2] in *augmented* (post-warp) image space (plan §8)
    lead_bboxes: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class ECGMetadata:
    """Complete per-image metadata record (plan §8)."""

    image_id: str
    source_xml: str
    format_detected: str
    leads: dict[str, LeadData]
    layout: Layout
    augmentation: AugmentationParams
    # 3x3 inverse homography — supervision for perspective correction (plan §6 L3, §9.1).
    homography_inv: list[list[float]]
    # Filename of the dense inverse displacement field (.npy), or None (plan §8).
    warp_field: Optional[str] = None
    # Randomized render style used for this image (traceability; diversity overhaul).
    render_style: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "source_xml": self.source_xml,
            "format_detected": self.format_detected,
            "leads": {
                name: asdict(lead) for name, lead in self.leads.items()
            },
            "layout": asdict(self.layout),
            "augmentation": self.augmentation.to_dict(),
            "homography_inv": self.homography_inv,
            "warp_field": self.warp_field,
            "render_style": self.render_style,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ECGMetadata":
        leads = {
            name: LeadData(**lead) for name, lead in data["leads"].items()
        }
        layout = Layout(**data["layout"])
        augmentation = AugmentationParams.from_dict(data["augmentation"])
        return cls(
            image_id=data["image_id"],
            source_xml=data["source_xml"],
            format_detected=data["format_detected"],
            leads=leads,
            layout=layout,
            augmentation=augmentation,
            homography_inv=data["homography_inv"],
            warp_field=data.get("warp_field"),
            render_style=data.get("render_style"),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "ECGMetadata":
        return cls.from_dict(json.loads(text))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "ECGMetadata":
        with open(path, encoding="utf-8") as fh:
            return cls.from_json(fh.read())

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self, *, expected_leads: int = 12) -> None:
        """Raise ValueError if the record is structurally inconsistent.

        Checks (plan §4.2 validation + §8 schema):
          - expected number of leads present
          - sample rates consistent across leads
          - every lead has a bounding box
          - homography_inv is 3x3
          - augmentation params in range
        """
        errors: list[str] = []

        # Lead count
        if len(self.leads) != expected_leads:
            errors.append(
                f"expected {expected_leads} leads, got {len(self.leads)}: "
                f"{sorted(self.leads)}"
            )

        # Sample-rate consistency
        rates = {lead.sample_rate_hz for lead in self.leads.values()}
        if len(rates) > 1:
            errors.append(f"inconsistent sample rates across leads: {rates}")

        # Every lead needs a bbox for downstream segmentation supervision
        missing_bboxes = [
            name for name in self.leads if name not in self.layout.lead_bboxes
        ]
        if missing_bboxes:
            errors.append(f"leads missing bboxes: {missing_bboxes}")

        for name, bbox in self.layout.lead_bboxes.items():
            if len(bbox) != 4:
                errors.append(f"bbox for {name!r} must be [x1,y1,x2,y2], got {bbox}")

        # Homography shape
        H = self.homography_inv
        if len(H) != 3 or any(len(row) != 3 for row in H):
            errors.append(f"homography_inv must be 3x3, got shape "
                          f"{len(H)}x{len(H[0]) if H else 0}")

        # Nested augmentation validity
        try:
            self.augmentation.validate()
        except ValueError as exc:
            errors.append(str(exc))

        if errors:
            raise ValueError(
                "Invalid ECGMetadata:\n  " + "\n  ".join(errors)
            )

    def is_valid(self, *, expected_leads: int = 12) -> bool:
        try:
            self.validate(expected_leads=expected_leads)
            return True
        except ValueError:
            return False
