"""Global light source (plan §6 L2 / L4).

The single most important realism lever: wrinkle/fold shading must be consistent
with one global light direction (plan §6 L2 — "what makes wrinkles look real
rather than fake"). This object is set once per image and shared by every layer
that casts shading, so Layer 2 (handling) and Layer 4 (lighting) agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LightSource:
    angle_deg: float        # azimuth in the image plane, 0 = +x (right), CW
    elevation_deg: float    # 20 (grazing) .. 90 (overhead)

    @property
    def direction(self) -> tuple[float, float]:
        """Unit light direction in image coords (y points down)."""
        a = math.radians(self.angle_deg)
        return (math.cos(a), math.sin(a))

    @property
    def grazing_strength(self) -> float:
        """0..1 — low elevation = stronger, longer shadows (grazing light)."""
        e = max(0.0, min(90.0, self.elevation_deg))
        return float(math.cos(math.radians(e)))  # 1 at horizon, 0 overhead

    def shading_sign(self, perp_x: float, perp_y: float) -> float:
        """Sign of shading for a surface whose tilt points along (perp_x, perp_y).

        Positive => that side faces the light (highlight); negative => shadow.
        """
        lx, ly = self.direction
        dot = perp_x * lx + perp_y * ly
        return 1.0 if dot >= 0 else -1.0
