"""PhysioRender — physically-grounded ECG photo simulator.

Phase 0 exposes the two core data contracts that the rest of the system binds to:
``AugmentationParams`` (the degradation knobs) and ``ECGMetadata`` (the per-image
record). See ROADMAP.md.
"""

from __future__ import annotations

from .metadata import ECGMetadata, Layout, LeadData
from .params import AugmentationParams

__version__ = "0.1.0"

__all__ = [
    "AugmentationParams",
    "ECGMetadata",
    "Layout",
    "LeadData",
    "__version__",
]
