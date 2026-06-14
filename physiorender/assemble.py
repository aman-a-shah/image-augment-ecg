"""Assemble the metadata record for an augmented image (plan §8).

Bridges the numpy/PIL world (ingestion, render, degrade) to the pure-stdlib
:class:`ECGMetadata` contract — converting waveforms to lists and matrices to
nested lists so the result is JSON-serializable and validatable.
"""

from __future__ import annotations

import numpy as np

from .degrade.engine import AugmentationResult
from .ingest.record import ECGRecord
from .metadata import ECGMetadata, Layout, LeadData
from .render.renderer import RenderResult


def build_metadata(
    record: ECGRecord,
    render_result: RenderResult,
    aug_result: AugmentationResult,
    *,
    image_id: str,
    warp_field_filename: str | None = None,
) -> ECGMetadata:
    """Build a complete :class:`ECGMetadata` from pipeline outputs (plan §8)."""
    leads = {
        name: LeadData(
            signal_mv=[round(float(v), 5) for v in lead.signal_mv.tolist()],
            sample_rate_hz=int(lead.sample_rate_hz),
        )
        for name, lead in record.leads.items()
    }

    layout = Layout(
        template=render_result.template,
        paper_speed_mm_s=render_result.paper_speed_mm_s,
        gain_mm_mv=render_result.gain_mm_mv,
        lead_bboxes={k: [int(v) for v in box]
                     for k, box in aug_result.lead_bboxes.items()},
    )

    return ECGMetadata(
        image_id=image_id,
        source_xml=record.source_path,
        format_detected=record.source_format,
        leads=leads,
        layout=layout,
        augmentation=aug_result.params,
        homography_inv=np.asarray(aug_result.homography_inv,
                                  dtype=float).tolist(),
        warp_field=warp_field_filename,
    )
