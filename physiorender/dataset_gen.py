"""Batch dataset generation (plan §10, §12 Phase 4).

Turns the simulator into a data factory: for each source ECG, render once and
emit N augmented variants with full metadata, under deterministic per-image
seeds. Writes a JSONL manifest and runs a metadata-integrity pass over the batch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .assemble import build_metadata
from .degrade import DegradationEngine
from .ingest import load_ecg
from .logging_setup import get_logger
from .metadata import ECGMetadata
from .render import ECGRenderer
from .sampling import ParameterSampler

log = get_logger(__name__)


@dataclass
class GenerationReport:
    n_sources: int = 0
    n_images: int = 0
    n_valid: int = 0
    n_failed: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.n_failed == 0 and self.n_images > 0


def generate_dataset(
    sources: list[str | Path],
    *,
    out_dir: str | Path,
    n_per_source: int = 10,
    dpi: int = 300,
    paper_speed_mm_s: int = 25,
    gain_mm_mv: int = 10,
    base_seed: int = 0,
    sampler: ParameterSampler | None = None,
    save_warp: bool = True,
    write_signals: bool = True,
) -> GenerationReport:
    """Generate ``n_per_source`` augmented images for each source ECG.

    ``write_signals=False`` drops the (large) ground-truth waveform arrays from
    the per-image JSON — useful for quick visual batches.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sampler = sampler or ParameterSampler()
    engine = DegradationEngine(dpi=dpi)
    report = GenerationReport(n_sources=len(sources))

    manifest_path = out / "manifest.jsonl"
    counter = 0
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for src in sources:
            try:
                record = load_ecg(src, validate=True)
            except Exception as exc:  # noqa: BLE001 - log-and-skip bad sources
                log.error("skipping %s: %s", src, exc)
                report.failures.append(f"{src}: load failed ({exc})")
                report.n_failed += 1
                continue

            stem = Path(src).stem

            for _ in range(n_per_source):
                seed = base_seed + counter
                counter += 1
                rng = np.random.default_rng(seed)
                # Each variant gets its own randomized render style (diversity).
                style = sampler.sample_style(rng)
                params = sampler.sample(rng)
                render = ECGRenderer(dpi=dpi, style=style).render(record)
                result = engine.augment(render.image, params, seed=seed,
                                        lead_bboxes=render.lead_bboxes)

                image_id = f"{stem}_aug_{seed:06d}"
                jpg = f"{image_id}.jpg"
                result.image.save(out / jpg, quality=92)

                warp_name = None
                if save_warp:
                    warp_name = f"warp_{image_id}.npy"
                    np.save(out / warp_name, result.warp_field_inverse())

                meta = build_metadata(record, render, result, image_id=image_id,
                                      warp_field_filename=warp_name)
                if not write_signals:
                    meta.leads = {k: _empty_lead(v) for k, v in meta.leads.items()}
                json_name = f"{image_id}.json"
                meta.save(str(out / json_name))

                valid = meta.is_valid()
                report.n_images += 1
                report.n_valid += int(valid)
                if not valid:
                    report.n_failed += 1
                    report.failures.append(image_id)

                manifest.write(json.dumps({
                    "image_id": image_id, "source": str(src),
                    "image": jpg, "metadata": json_name, "warp": warp_name,
                    "valid": valid, "seed": seed,
                }) + "\n")

    log.info("generated %d images from %d sources (%d valid, %d failed)",
             report.n_images, report.n_sources, report.n_valid, report.n_failed)
    return report


def _empty_lead(lead):
    lead.signal_mv = []
    return lead


def validate_dataset(out_dir: str | Path, *, expected_leads: int = 12
                     ) -> GenerationReport:
    """Re-load every metadata file in a dataset and validate it (integrity pass)."""
    out = Path(out_dir)
    report = GenerationReport()
    for jf in sorted(out.glob("*_aug_*.json")):
        report.n_images += 1
        try:
            meta = ECGMetadata.load(str(jf))
            # When signals are stripped we can't check lead count via signals,
            # but bboxes and homography are still validated.
            n = len(meta.leads) if meta.leads and meta.leads[
                next(iter(meta.leads))].signal_mv else expected_leads
            meta.validate(expected_leads=min(expected_leads, n) if n else expected_leads)
            report.n_valid += 1
        except Exception as exc:  # noqa: BLE001
            report.n_failed += 1
            report.failures.append(f"{jf.name}: {exc}")
    return report
