"""Physical Degradation Engine (plan §6).

Phase 3 implements Layers 1-2 (paper + handling). The public surface is
:class:`DegradationEngine` / :class:`DegradationResult`; individual layer
functions are exposed for isolated visual unit testing.
"""

from __future__ import annotations

from .engine import DegradationEngine, DegradationResult
from .light import LightSource
from .warp import DisplacementField, apply_displacement, invert_displacement

__all__ = [
    "DegradationEngine",
    "DegradationResult",
    "LightSource",
    "DisplacementField",
    "apply_displacement",
    "invert_displacement",
]
