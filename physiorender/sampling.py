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

    def sample_style(self, rng: np.random.Generator):
        """Sample a randomized clean-render style (see render.sample_render_style)."""
        from .render import sample_render_style
        return sample_render_style(rng)

    def sample(self, rng: np.random.Generator) -> AugmentationParams:
        cfg = self.cfg

        # --- per-image "severity" latent: pristine (0) -> trashed (1) ---
        # Makes the dataset span the full quality range instead of clustering
        # every image around the average amount of every effect.
        sev = float(rng.beta(1.6, 2.2))          # skewed toward lighter, full tail
        lo, hi = 0.3 + 0.7 * sev, sev            # convenience scalars

        def chance(p: float) -> bool:
            return bool(rng.random() < p)

        def cap(n: int, hi_: int) -> int:
            return int(min(hi_, max(0, n)))

        # --- paper / ink ---
        yellowing = float(np.clip(rng.uniform(0.0, 0.30) * lo, 0.0, 0.30))
        ink = float(np.clip(rng.uniform(0.0, 0.08) * lo, 0.0, 0.08))

        # --- handling ---
        n_wrinkles = cap(rng.binomial(6, 0.15 + 0.5 * sev), 6)
        wrinkle_int = float(rng.uniform(0.3, 1.0))
        n_folds = int(rng.choice([0, 1, 2],
                                 p=[max(0.05, 0.7 - 0.5 * sev),
                                    min(0.6, 0.25 + 0.25 * sev),
                                    min(0.35, 0.05 + 0.25 * sev)]))
        n_stains = cap(rng.binomial(4, 0.10 + 0.35 * sev), 4)
        n_pen = 0   # pen-mark annotations disabled: too noisy on the trace
        n_fp = cap(rng.binomial(2, 0.12 + 0.25 * sev), 2)

        # --- camera geometry ---
        tilt_x = float(np.clip(rng.normal(0, 6 + 6 * sev), -15, 15))
        tilt_y = float(np.clip(rng.normal(0, 7 + 7 * sev), -18, 18))
        rotation = float(np.clip(rng.normal(0, 3 + 3 * sev), -6, 6))
        lens_k1 = float(rng.uniform(0.0, 0.08) * hi)
        crop_margin = float(rng.uniform(0.02, 0.15))

        blurred = chance(cfg.p_blur * (0.5 + 0.5 * sev))
        blur_type = str(rng.choice(["motion", "defocus", "handshake"])) if blurred else "none"
        blur_strength = float(rng.uniform(0.1, 0.85)) if blurred else 0.0

        # --- correlations (plan §7) ---
        yellowing = float(np.clip(
            yellowing + cfg.folds_increase_yellowing * 0.12 * (n_folds / 2.0), 0.0, 0.30))
        tilt_mag = min(1.0, (abs(tilt_x) / 15.0 + abs(tilt_y) / 18.0) / 2.0)
        crop_margin = float(np.clip(
            crop_margin - cfg.tilt_reduces_crop * 0.08 * tilt_mag, 0.02, 0.15))
        p_spec = cfg.p_specular * (1.0 - cfg.blur_suppresses_specular * blur_strength)
        n_specular = cap(1 + rng.binomial(2, 0.25 * sev), 3) if chance(p_spec) else 0

        # --- global tone / ISP ---
        contrast = float(np.clip(rng.normal(1.0, 0.10 + 0.15 * sev), 0.65, 1.45))
        brightness = float(np.clip(rng.normal(1.0, 0.05 + 0.10 * sev), 0.78, 1.22))
        gamma = float(np.clip(rng.normal(1.0, 0.08 + 0.18 * sev), 0.7, 1.55))
        saturation = float(np.clip(rng.normal(1.0, 0.12 + 0.20 * sev), 0.45, 1.55))
        vignette = float(np.clip(rng.uniform(0.0, 0.6) * hi, 0, 0.6)) if chance(0.45 + 0.3 * sev) else 0.0
        chroma = float(np.clip(rng.uniform(0.0, 2.5) * hi, 0, 2.5)) if chance(0.4) else 0.0
        moire = float(np.clip(rng.uniform(0.0, 0.35) * hi, 0, 0.35)) if chance(0.12) else 0.0
        sharpen = float(np.clip(rng.uniform(0.0, 0.8), 0, 0.8)) if chance(0.4) else 0.0
        second_jpeg = int(rng.integers(40, 90)) if chance(0.25 + 0.25 * sev) else 95

        return AugmentationParams(
            yellowing_intensity=yellowing,
            ink_density_variation=ink,
            n_wrinkles=n_wrinkles,
            wrinkle_intensity=wrinkle_int,
            n_folds=n_folds,
            has_stain=n_stains > 0,
            stain_opacity=float(rng.uniform(0.1, 0.4)),
            has_pen_marks=False,
            tilt_x_deg=tilt_x,
            tilt_y_deg=tilt_y,
            rotation_deg=rotation,
            blur_type=blur_type,
            blur_strength=blur_strength,
            lens_k1=lens_k1,
            has_lens_dirt=chance(cfg.p_lens_dirt * (0.5 + sev)),
            crop_margin=crop_margin,
            light_angle_deg=float(rng.uniform(0, 360)),
            light_elevation_deg=float(rng.uniform(20, 70)),
            has_specular=n_specular > 0,
            specular_intensity=float(rng.uniform(0.3, 0.9)),
            has_fl_banding=chance(cfg.p_fl_banding),
            shadow_width_fraction=(float(rng.uniform(0.05, 0.25))
                                   if chance(cfg.p_hand_shadow * (0.5 + sev)) else 0.0),
            jpeg_quality=int(rng.integers(65, 89)),
            noise_iso_equiv=int(rng.integers(100, 1601)),
            colour_temp_delta_k=int(rng.integers(-300, 301)),
            second_jpeg_quality=second_jpeg,
            contrast=contrast, brightness=brightness, gamma=gamma, saturation=saturation,
            vignette_strength=vignette, chromatic_aberration=chroma,
            moire_strength=moire, sharpen_strength=sharpen,
            n_stains=n_stains, n_pen_marks=n_pen, n_specular=n_specular, n_fingerprints=n_fp,
        )
