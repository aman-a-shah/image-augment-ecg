"""Randomizable rendering style (plan §5; diversity overhaul).

The clean render is the single biggest source of "templating": if every clean
ECG has identical paper colour, grid hue, trace width, fonts, margins and layout,
a model trained at scale learns *those* constants instead of generalizing to real
printouts from many devices. :class:`RenderStyle` makes all of that samplable, so
the clean image varies per photo before any degradation is applied.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .. import config

RGB = Tuple[int, int, int]


def hex_to_rgb(h: str) -> RGB:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _hsv_rgb(h: float, s: float, v: float) -> RGB:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0, min(1, s)), max(0, min(1, v)))
    return (int(r * 255), int(g * 255), int(b * 255))


@dataclass
class RenderStyle:
    """All visual + geometric knobs of the clean renderer."""

    # Colours (RGB 0-255)
    bg_color: RGB = field(default_factory=lambda: hex_to_rgb(config.COLOR_BACKGROUND))
    grid_major_color: RGB = field(default_factory=lambda: hex_to_rgb(config.COLOR_GRID_LARGE))
    grid_minor_color: RGB = field(default_factory=lambda: hex_to_rgb(config.COLOR_GRID_SMALL))
    trace_color: RGB = field(default_factory=lambda: hex_to_rgb(config.COLOR_TRACE))
    trace_width_mm: float = config.TRACE_WIDTH_MM
    show_minor_grid: bool = True
    grid_fade: float = 0.0           # 0 = full strength, 1 = grid blended into paper

    # Acquisition
    paper_speed_mm_s: int = config.DEFAULT_PAPER_SPEED_MM_S
    gain_mm_mv: int = config.DEFAULT_GAIN_MM_MV
    template: str = "standard_12lead_rhythm"

    # Geometry (mm)
    margin_left_mm: float = 12.0
    margin_right_mm: float = 8.0
    margin_top_mm: float = 18.0
    margin_bottom_mm: float = 8.0
    row_height_mm: float = 40.0
    strip_seconds: float = 2.5

    # Decoration
    show_calibration: bool = True
    show_header: bool = True
    header_scale: float = 1.0
    label_scale: float = 1.0

    def faded(self, color: RGB) -> RGB:
        """Blend a grid colour toward the paper by ``grid_fade``."""
        if self.grid_fade <= 0:
            return color
        f = self.grid_fade
        return tuple(int(c * (1 - f) + b * f)
                     for c, b in zip(color, self.bg_color))  # type: ignore


def default_style() -> RenderStyle:
    """The canonical style (matches the original fixed renderer)."""
    return RenderStyle()


def sample_render_style(rng: np.random.Generator) -> RenderStyle:
    """Sample a wildly varied but plausible rendering style.

    Spans device-to-device variation: warm/cool/grey papers, red/pink/orange/
    faint/blue grids, black/blue/grey/brown traces, varied trace widths, margins,
    paper speeds, gains, layout templates, and decoration presence.
    """
    # --- paper: mostly warm whites, occasionally grey/blueish ---
    paper_v = rng.uniform(0.90, 1.0)
    paper_hue = rng.choice([0.10, 0.12, 0.08, 0.58])     # warm cream / neutral / cool
    paper_s = rng.uniform(0.0, 0.06)
    bg = _hsv_rgb(paper_hue, paper_s, paper_v)

    # --- grid: red/pink/orange family, sometimes faint or grey or blue ---
    grid_kind = rng.choice(["red", "pink", "orange", "brown", "grey", "blue"],
                           p=[0.34, 0.26, 0.16, 0.1, 0.1, 0.04])
    if grid_kind == "grey":
        g_hue, g_s = rng.uniform(0.0, 1.0), rng.uniform(0.0, 0.05)
    elif grid_kind == "blue":
        g_hue, g_s = rng.uniform(0.55, 0.62), rng.uniform(0.2, 0.5)
    else:
        base = {"red": 0.99, "pink": 0.96, "orange": 0.05, "brown": 0.07}[grid_kind]
        g_hue, g_s = (base + rng.uniform(-0.02, 0.02)) % 1.0, rng.uniform(0.18, 0.55)
    major = _hsv_rgb(g_hue, g_s, rng.uniform(0.72, 0.93))
    minor = _hsv_rgb(g_hue, g_s * rng.uniform(0.5, 0.85), rng.uniform(0.86, 0.98))

    # --- trace: dark, usually near-black, sometimes dark blue/green/brown ---
    t_kind = rng.choice(["black", "blue", "grey", "brown", "green"],
                        p=[0.6, 0.16, 0.12, 0.08, 0.04])
    t_hue = {"black": rng.uniform(0, 1), "blue": 0.62, "grey": 0.0,
             "brown": 0.07, "green": 0.33}[t_kind]
    t_s = {"black": 0.0, "blue": 0.5, "grey": 0.0, "brown": 0.4, "green": 0.4}[t_kind]
    trace = _hsv_rgb(t_hue, t_s, rng.uniform(0.05, 0.22))

    return RenderStyle(
        bg_color=bg,
        grid_major_color=major,
        grid_minor_color=minor,
        trace_color=trace,
        trace_width_mm=float(rng.uniform(0.3, 0.7)),
        show_minor_grid=bool(rng.random() < 0.9),
        grid_fade=float(np.clip(rng.normal(0.15, 0.18), 0.0, 0.7)),
        paper_speed_mm_s=int(rng.choice([25, 25, 25, 50])),
        gain_mm_mv=int(rng.choice([10, 10, 10, 5, 20])),
        template=str(rng.choice(["standard_12lead_rhythm", "standard_12lead"],
                                p=[0.7, 0.3])),
        margin_left_mm=float(rng.uniform(8, 16)),
        margin_right_mm=float(rng.uniform(5, 12)),
        margin_top_mm=float(rng.uniform(12, 24)),
        margin_bottom_mm=float(rng.uniform(5, 12)),
        row_height_mm=float(rng.uniform(32, 46)),
        strip_seconds=float(rng.choice([2.5, 2.5, 2.5, 5.0])),
        show_calibration=bool(rng.random() < 0.85),
        show_header=bool(rng.random() < 0.8),
        header_scale=float(rng.uniform(0.8, 1.3)),
        label_scale=float(rng.uniform(0.8, 1.3)),
    )
