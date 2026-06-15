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
from . import layer3_camera as l3
from . import layer4_lighting as l4
from . import layer5_capture as l5
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


@dataclass
class AugmentationResult:
    """Full pipeline output (Layers 1-5) + all supervision for metadata (plan §8)."""

    image: Image.Image                       # final augmented photo (capture frame)
    homography_inv: np.ndarray               # 3x3 photo-px -> doc-px (Stage-1 supervision)
    homography: np.ndarray                    # 3x3 doc-px -> photo-px
    lead_bboxes: dict[str, list[int]]        # post-warp bboxes in photo space (plan §8)
    handling_displacement: DisplacementField  # Layer-2 doc-space warp
    light: LightSource
    seed: int
    params: AugmentationParams
    applied: list[str] = field(default_factory=list)

    def warp_field_inverse(self, *, iters: int = 12) -> np.ndarray:
        """Inverse of the Layer-2 handling warp as [H, W, 2] (plan §8 warp_field)."""
        idx, idy = self.handling_displacement.invert(iters=iters)
        return np.stack([idx, idy], axis=-1).astype(np.float32)


class DegradationEngine:
    """Apply the Physical Degradation Engine to a clean rendered ECG image."""

    def __init__(self, *, dpi: int = config.DEFAULT_DPI) -> None:
        self.dpi = dpi
        self.ppm = config.mm_to_px(1.0, dpi)  # pixels per millimeter

    def apply(
        self,
        image: Image.Image,
        params: AugmentationParams,
        *,
        seed: int = 0,
        rng: np.random.Generator | None = None,
    ) -> DegradationResult:
        """Apply PDE Layers 1-2 (document degradation) only."""
        params.validate()
        if rng is None:
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
        n_stains = params.stain_count
        for _ in range(n_stains):
            arr = l2.add_stain(arr, rng, opacity=params.stain_opacity * rng.uniform(0.7, 1.2))
        if n_stains:
            applied.append(f"stain×{n_stains}")
        n_pens = params.pen_count
        for _ in range(n_pens):
            arr = l2.add_pen_marks(arr, rng)
        if n_pens:
            applied.append(f"pen×{n_pens}")
        n_fp = params.n_fingerprints if params.n_fingerprints > 0 else (
            1 if rng.random() < 0.40 else 0)
        for _ in range(n_fp):
            arr = l2.add_fingerprint(arr, rng, opacity=rng.uniform(0.08, 0.2))
        if n_fp:
            applied.append(f"fingerprint×{n_fp}")

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

    # ------------------------------------------------------------------ #
    # Full capture pipeline (Layers 1-5)
    # ------------------------------------------------------------------ #
    def augment(
        self,
        clean_image: Image.Image,
        params: AugmentationParams,
        *,
        seed: int = 0,
        lead_bboxes: dict[str, list[int]] | None = None,
    ) -> AugmentationResult:
        """Run the full pipeline: clean printout -> realistic smartphone photo.

        Capture order follows real physics (plan §6 / §14):
        document degradation (L1-2) -> lens distortion -> perspective -> blur
        -> lens dirt -> lighting (L4) -> sensor noise + JPEG (L5).
        """
        params.validate()
        rng = np.random.default_rng(seed)

        # --- Layers 1-2: degrade the flat document (shares the rng stream) ---
        doc_res = self.apply(clean_image, params, rng=rng)
        doc = iu.pil_to_float(doc_res.image)
        h, w = doc.shape[:2]
        out_size = (w, h)
        applied = list(doc_res.applied)
        light = doc_res.light

        # --- Layer 3: lens distortion (pre-warp) ---
        lens = l3.make_lens(w, h, params.lens_k1)
        doc_lens = l3.apply_barrel(doc, lens)
        if params.lens_k1 > 0:
            applied.append("lens_distortion")

        # --- Layer 3: perspective + framing (exports H_inv) ---
        persp = l3.frame_and_warp(
            doc_lens, rng,
            tilt_x_deg=params.tilt_x_deg, tilt_y_deg=params.tilt_y_deg,
            rotation_deg=params.rotation_deg, crop_margin=params.crop_margin,
            out_size=out_size,
        )
        bg = l3.make_background(h, w, rng)
        photo = l3.composite_on_background(persp.image, persp.mask, bg)
        applied.append("perspective")

        # Transform lead bboxes through lens + homography into photo space.
        post_bboxes: dict[str, list[int]] = {}
        for key, bbox in (lead_bboxes or {}).items():
            post_bboxes[key] = l3.transform_bbox(bbox, lens, persp.H, out_size)

        # --- Layer 3: blur (post-warp) + lens dirt + chromatic aberration ---
        photo = l3.apply_blur(photo, rng, blur_type=params.blur_type,
                              strength=params.blur_strength, ppm=self.ppm)
        if params.blur_type != "none" and params.blur_strength > 0:
            applied.append(f"blur:{params.blur_type}")
        if params.has_lens_dirt:
            photo = l3.apply_lens_dirt(photo, rng)
            applied.append("lens_dirt")
        if params.chromatic_aberration > 0:
            photo = l5.apply_chromatic_aberration(photo, px=params.chromatic_aberration)
            applied.append("chromatic_aberration")

        # --- Layer 4: lighting & environment ---
        photo = l4.apply_ambient_gradient(photo, light, strength=rng.uniform(0.10, 0.26))
        applied.append("ambient")
        if rng.random() < 0.35:  # occasional second light source (breaks single-light prior)
            light2 = LightSource(rng.uniform(0, 360), rng.uniform(20, 70))
            photo = l4.apply_ambient_gradient(photo, light2, strength=rng.uniform(0.06, 0.16))
            applied.append("ambient2")
        n_spec = params.specular_count
        for _ in range(n_spec):
            photo = l4.apply_specular(photo, rng, light,
                                      intensity=params.specular_intensity * rng.uniform(0.7, 1.1))
        if n_spec:
            applied.append(f"specular×{n_spec}")
        if params.has_fl_banding:
            photo = l4.apply_fluorescent_banding(photo, rng)
            applied.append("fl_banding")
        if params.shadow_width_fraction > 0:
            photo = l4.apply_hand_shadow(photo, rng,
                                         width_fraction=params.shadow_width_fraction)
            applied.append("hand_shadow")
        if params.vignette_strength > 0:
            photo = l4.apply_vignette(photo, rng, strength=params.vignette_strength)
            applied.append("vignette")
        if params.moire_strength > 0:
            photo = l4.apply_moire(photo, rng, strength=params.moire_strength)
            applied.append("moire")

        # --- Layer 5: sensor noise -> ISP tone -> sharpen -> JPEG (capture order) ---
        photo = l5.apply_sensor_noise(photo, rng, iso_equiv=params.noise_iso_equiv)
        photo = l5.apply_tone(photo, contrast=params.contrast, brightness=params.brightness,
                              gamma=params.gamma, saturation=params.saturation)
        photo = l5.apply_color_temperature(photo, delta_k=params.colour_temp_delta_k)
        if params.sharpen_strength > 0:
            photo = l5.apply_sharpen(photo, strength=params.sharpen_strength)
            applied.append("sharpen")
        photo = l5.apply_jpeg(photo, quality=params.jpeg_quality)
        applied += ["sensor_noise", "tone", "color_temp", "jpeg"]
        if params.second_jpeg_quality < 90:  # re-saved/messaged image
            photo = l5.apply_jpeg(photo, quality=params.second_jpeg_quality)
            applied.append("jpeg2")

        log.info("augment(seed=%d): %s", seed, ", ".join(applied))
        return AugmentationResult(
            image=iu.float_to_pil(photo),
            homography_inv=persp.H_inv,
            homography=persp.H,
            lead_bboxes=post_bboxes,
            handling_displacement=doc_res.displacement,
            light=light,
            seed=seed,
            params=params,
            applied=applied,
        )
