"""Physical Degradation Engine — Layers 1-2 (plan §6).

Phase 3 scope: paper artifacts (Layer 1) and handling degradation (Layer 2).
All photometric effects are painted onto the flat page, geometric crease/curl
displacements accumulate into one :class:`DisplacementField`, and the page is
remapped **once** at the end. Camera / lighting / compression (Layers 3-5) plug
into this same pipeline in Phase 4.

Reproducibility: everything is driven by a single integer ``seed`` so an image is
fully regenerable, and the composite warp field is invertible for label recovery
(plan §8 warp_field, §14 "every augmentation is parameterised and traceable").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .. import config
from ..logging_setup import get_logger
from ..params import AugmentationParams
from . import imageutil as iu
from . import layer1_paper as l1
from . import layer2_handling as l2
from .light import LightSource
from .warp import DisplacementField

log = get_logger(__name__)


@dataclass
class DegradationResult:
    """Output of the PDE: degraded image + the supervision it produced."""

    image: Image.Image
    displacement: DisplacementField
    light: LightSource
    seed: int
    params: AugmentationParams
    applied: list[str] = field(default_factory=list)

    def inverse_warp(self, *, iters: int = 12) -> np.ndarray:
        """Return the inverse displacement as an [H, W, 2] (dx, dy) array (plan §8)."""
        idx, idy = self.displacement.invert(iters=iters)
        return np.stack([idx, idy], axis=-1).astype(np.float32)

    def warp_array(self) -> np.ndarray:
        """Composite forward (backward-convention) displacement as [H, W, 2]."""
        return np.stack([self.displacement.dx, self.displacement.dy], axis=-1)


class DegradationEngine:
    """Apply PDE Layers 1-2 to a clean rendered ECG image."""

    def __init__(self, *, dpi: int = config.DEFAULT_DPI) -> None:
        self.dpi = dpi
        self.ppm = config.mm_to_px(1.0, dpi)  # pixels per millimeter

    def apply(
        self,
        image: Image.Image,
        params: AugmentationParams,
        *,
        seed: int = 0,
    ) -> DegradationResult:
        params.validate()
        rng = np.random.default_rng(seed)
        arr = iu.pil_to_float(image)
        h, w = arr.shape[:2]
        light = LightSource(params.light_angle_deg, params.light_elevation_deg)
        disp = DisplacementField.zeros(h, w)
        applied: list[str] = []

        # --- Layer 1: paper & print artifacts (photometric) ---
        if params.yellowing_intensity > 0:
            arr = l1.apply_yellowing(arr, rng, params.yellowing_intensity)
            applied.append("yellowing")
        if params.ink_density_variation > 0:
            arr = l1.apply_ink_density(arr, rng, params.ink_density_variation)
            applied.append("ink_density")
        arr = l1.apply_grid_bleed(arr, sigma=0.4 * (self.ppm / 11.81))
        applied.append("grid_bleed")
        n_skips = int(rng.integers(0, 6))
        if n_skips:
            arr = l1.apply_ink_skip(arr, rng, n_skips)
            applied.append(f"ink_skip×{n_skips}")

        # --- Layer 2: handling degradation ---
        # Overlays first (on flat paper, so they deform with the warp).
        if params.has_stain:
            arr = l2.add_stain(arr, rng, opacity=params.stain_opacity)
            applied.append("stain")
        if params.has_pen_marks:
            arr = l2.add_pen_marks(arr, rng)
            applied.append("pen_marks")
        if rng.random() < 0.40:  # fingerprint probability (plan §6 L2)
            arr = l2.add_fingerprint(arr, rng)
            applied.append("fingerprint")

        # Creases & curl: brightness now, displacement accumulated for one remap.
        if params.n_wrinkles > 0 and params.wrinkle_intensity > 0:
            arr = l2.add_wrinkles(arr, disp, rng, n=params.n_wrinkles,
                                  intensity=params.wrinkle_intensity, light=light,
                                  ppm=self.ppm)
            applied.append(f"wrinkles×{params.n_wrinkles}")
        if params.n_folds > 0:
            arr = l2.add_folds(arr, disp, rng, n=params.n_folds,
                               intensity=max(0.4, params.wrinkle_intensity),
                               light=light, ppm=self.ppm)
            applied.append(f"folds×{params.n_folds}")
        curl = rng.uniform(0.0, 1.0)
        if curl > 0.3:
            arr = l2.apply_edge_curl(arr, disp, rng, strength=curl, light=light)
            applied.append("edge_curl")

        # Single composite remap for all geometric handling effects.
        if not disp.is_identity():
            arr = disp.apply(arr)

        log.info("degrade(seed=%d): %s", seed, ", ".join(applied) or "none")
        return DegradationResult(
            image=iu.float_to_pil(arr),
            displacement=disp,
            light=light,
            seed=seed,
            params=params,
            applied=applied,
        )
