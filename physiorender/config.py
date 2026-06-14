"""Global rendering constants for PhysioRender.

These are the physical/visual defaults shared across the renderer (Phase 2) and
the degradation engine (Phases 3-4). Kept dependency-free so the contract layer
can import them without pulling in numpy/opencv.

References: plan.md §5.1 (paper & grid), §5.3 (signal rendering).
"""

from __future__ import annotations

# --- Resolution -------------------------------------------------------------
# Real thermal printers produce ~300 DPI; we need that headroom to add
# realistic degradation later (plan §5.1).
DEFAULT_DPI: int = 300
MM_PER_INCH: float = 25.4

# --- ECG paper geometry -----------------------------------------------------
SMALL_GRID_MM: float = 1.0   # 1mm small grid
LARGE_GRID_MM: float = 5.0   # 5mm large grid

# --- Colors (plan §5.1, §5.3) ----------------------------------------------
COLOR_BACKGROUND: str = "#FFFDF8"   # slightly warm white, not pure white
COLOR_GRID_LARGE: str = "#E8A0A0"   # warm red, 5mm lines
COLOR_GRID_SMALL: str = "#F0C8C8"   # lighter red, 1mm lines
COLOR_TRACE: str = "#1A1A1A"        # near-black; thermal printers have a grey cast

# --- Trace ------------------------------------------------------------------
TRACE_WIDTH_MM: float = 0.5         # ~6px at 300 DPI

# --- Acquisition defaults ---------------------------------------------------
DEFAULT_PAPER_SPEED_MM_S: int = 25  # 25 or 50 mm/s
DEFAULT_GAIN_MM_MV: int = 10        # 10mm/mV standard, 5mm/mV for high amplitude

# Standard 12-lead set (extended adds V3R/V4R/V7 etc. -> 15).
STANDARD_LEADS: tuple[str, ...] = (
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6",
)


def mm_to_px(mm: float, dpi: int = DEFAULT_DPI) -> float:
    """Convert millimeters to pixels at the given DPI."""
    return mm / MM_PER_INCH * dpi


def px_to_mm(px: float, dpi: int = DEFAULT_DPI) -> float:
    """Convert pixels to millimeters at the given DPI."""
    return px * MM_PER_INCH / dpi
