"""PhotoTrace — recovers clean waveform signal from an ECG photo (plan §9).

Stage 1 (perspective correction) + Stage 2 (lead segmentation) + Stage 3
(column-wise digitization), plus the end-to-end :class:`DigitizationPipeline`
and the domain-gap evaluation harness.

Submodules import torch lazily (import them directly) so the package itself stays
import-light.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
