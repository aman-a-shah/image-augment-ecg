"""Correlation-aware augmentation parameter sampling (plan §7).

Sampling each parameter independently produces physically implausible
combinations (a razor-sharp photo with heavy motion blur, a pristine-white page
folded in quarters). The :class:`ParameterSampler` samples a base draw and then
applies configurable correlations so the joint distribution matches real capture
behaviour (plan §7):

  - high blur      -> less specular   (blur comes from motion, not flat glossy paper)
  - many folds     -> more yellowing  (old / stored documents)
  - high tilt      -> less crop margin (tilt shrinks apparent document size)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import AugmentationParams


@dataclass
class SamplerConfig:
    """Tunable probabilities and correlation coefficients (plan §7)."""

    # Occurrence probabilities
    p_stain: float = 0.30
    p_pen_marks: float = 0.20
    p_specular: float = 0.35
    p_lens_dirt: float = 0.20
    p_fl_banding: float = 0.30
    p_blur: float = 0.75            # chance the photo is blurred at all
    p_hand_shadow: float = 0.30

    # Correlation strengths in [0, 1]
    blur_suppresses_specular: float = 0.6
    folds_increase_yellowing: float = 0.5
    tilt_reduces_crop: float = 0.5


class ParameterSampler:
    """Draws :class:`AugmentationParams` with realistic joint structure."""

    def __init__(self, config: SamplerConfig | None = None) -> None:
        self.cfg = config or SamplerConfig()

    def sample(self, rng: np.random.Generator) -> AugmentationParams:
        cfg = self.cfg

        # --- base independent draws ---
        yellowing = float(rng.uniform(0.05, 0.30))
        ink = float(rng.uniform(0.0, 0.08))

        n_wrinkles = int(rng.integers(0, 7))
        wrinkle_int = float(rng.uniform(0.3, 1.0))
        n_folds = int(rng.integers(0, 3))

        tilt_x = float(np.clip(rng.normal(0, 8), -15, 15))
        tilt_y = float(np.clip(rng.normal(0, 10), -18, 18))
        rotation = float(np.clip(rng.normal(0, 4), -6, 6))

        blurred = rng.random() < cfg.p_blur
        blur_type = str(rng.choice(["motion", "defocus", "handshake"])) if blurred else "none"
        blur_strength = float(rng.uniform(0.2, 0.85)) if blurred else 0.0

        lens_k1 = float(rng.uniform(0.0, 0.08))
        crop_margin = float(rng.uniform(0.02, 0.15))

        # --- correlations (plan §7) ---
        # folds -> yellowing (older paper)
        fold_frac = n_folds / 2.0
        yellowing = float(np.clip(
            yellowing + cfg.folds_increase_yellowing * 0.12 * fold_frac, 0.0, 0.30))

        # tilt -> smaller crop margin
        tilt_mag = min(1.0, (abs(tilt_x) / 15.0 + abs(tilt_y) / 18.0) / 2.0)
        crop_margin = float(np.clip(
            crop_margin - cfg.tilt_reduces_crop * 0.08 * tilt_mag, 0.02, 0.15))

        # blur -> suppress specular probability
        p_spec = cfg.p_specular * (1.0 - cfg.blur_suppresses_specular * blur_strength)
        has_specular = rng.random() < p_spec

        return AugmentationParams(
            yellowing_intensity=yellowing,
            ink_density_variation=ink,
            n_wrinkles=n_wrinkles,
            wrinkle_intensity=wrinkle_int,
            n_folds=n_folds,
            has_stain=bool(rng.random() < cfg.p_stain),
            stain_opacity=float(rng.uniform(0.1, 0.4)),
            has_pen_marks=bool(rng.random() < cfg.p_pen_marks),
            tilt_x_deg=tilt_x,
            tilt_y_deg=tilt_y,
            rotation_deg=rotation,
            blur_type=blur_type,
            blur_strength=blur_strength,
            lens_k1=lens_k1,
            has_lens_dirt=bool(rng.random() < cfg.p_lens_dirt),
            crop_margin=crop_margin,
            light_angle_deg=float(rng.uniform(0, 360)),
            light_elevation_deg=float(rng.uniform(20, 70)),
            has_specular=has_specular,
            specular_intensity=float(rng.uniform(0.3, 0.9)),
            has_fl_banding=bool(rng.random() < cfg.p_fl_banding),
            shadow_width_fraction=(float(rng.uniform(0.05, 0.25))
                                   if rng.random() < cfg.p_hand_shadow else 0.0),
            jpeg_quality=int(rng.integers(65, 89)),
            noise_iso_equiv=int(rng.integers(100, 1601)),
            colour_temp_delta_k=int(rng.integers(-300, 301)),
        )
