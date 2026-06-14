"""AugmentationParams — the parameter contract for one augmented image.

This is the single source of truth for *what knobs exist* in the Physical
Degradation Engine. Every augmented image is produced by sampling one of these
dicts (Phase 5) and it is saved verbatim in the metadata JSON (plan §7, §8), so
that every augmentation is reversible/traceable in the training label.

Pure stdlib by design — the contract layer must import without numpy/opencv.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

# Allowed sampling ranges per numeric field, taken directly from plan §7.
# Used by validate() now and by the ParameterSampler in Phase 5.
FIELD_RANGES: dict[str, tuple[float, float]] = {
    # Paper
    "yellowing_intensity": (0.0, 0.3),
    "ink_density_variation": (0.0, 0.08),
    # Handling
    "n_wrinkles": (0, 6),
    "wrinkle_intensity": (0.0, 1.0),
    "n_folds": (0, 2),
    "stain_opacity": (0.1, 0.4),
    # Camera
    "tilt_x_deg": (-15.0, 15.0),
    "tilt_y_deg": (-18.0, 18.0),
    "rotation_deg": (-6.0, 6.0),
    "blur_strength": (0.0, 1.0),
    "lens_k1": (0.0, 0.08),
    "crop_margin": (0.02, 0.15),
    # Lighting
    "light_angle_deg": (0.0, 360.0),
    "light_elevation_deg": (20.0, 70.0),
    "specular_intensity": (0.3, 0.9),
    "shadow_width_fraction": (0.0, 0.25),
    # Compression
    "jpeg_quality": (65, 88),
    "noise_iso_equiv": (100, 1600),
    "colour_temp_delta_k": (-300, 300),
}

VALID_BLUR_TYPES: tuple[str, ...] = ("motion", "defocus", "handshake", "none")


@dataclass
class AugmentationParams:
    """One sampled augmentation configuration (plan §7).

    Defaults sit at a neutral/mid point so an instance is trivially
    constructible; real values come from the ParameterSampler in Phase 5.
    """

    # --- Paper ---
    yellowing_intensity: float = 0.1
    ink_density_variation: float = 0.04

    # --- Handling ---
    n_wrinkles: int = 0
    wrinkle_intensity: float = 0.0
    n_folds: int = 0
    has_stain: bool = False
    stain_opacity: float = 0.2
    has_pen_marks: bool = False

    # --- Camera ---
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0
    rotation_deg: float = 0.0
    blur_type: str = "none"
    blur_strength: float = 0.0
    lens_k1: float = 0.0
    has_lens_dirt: bool = False
    crop_margin: float = 0.05

    # --- Lighting ---
    light_angle_deg: float = 45.0
    light_elevation_deg: float = 45.0
    has_specular: bool = False
    specular_intensity: float = 0.5
    has_fl_banding: bool = False
    shadow_width_fraction: float = 0.0

    # --- Compression ---
    jpeg_quality: int = 85
    noise_iso_equiv: int = 200
    colour_temp_delta_k: int = 0

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict (JSON-serializable)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AugmentationParams":
        """Build from a dict, ignoring unknown keys (forward-compatible)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Raise ValueError if any field is outside its allowed range/domain.

        Collects all violations into a single error for fast debugging.
        """
        errors: list[str] = []

        for name, (lo, hi) in FIELD_RANGES.items():
            value = getattr(self, name)
            if not (lo <= value <= hi):
                errors.append(f"{name}={value!r} out of range [{lo}, {hi}]")

        if self.blur_type not in VALID_BLUR_TYPES:
            errors.append(
                f"blur_type={self.blur_type!r} not in {VALID_BLUR_TYPES}"
            )

        if errors:
            raise ValueError(
                "Invalid AugmentationParams:\n  " + "\n  ".join(errors)
            )

    def is_valid(self) -> bool:
        """Non-raising convenience wrapper around validate()."""
        try:
            self.validate()
            return True
        except ValueError:
            return False
