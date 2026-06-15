"""ECG page layout templates (plan §5.2).

A layout is a page size (mm) plus a list of panels. Each panel places one lead's
strip on the page: which lead, which time window it shows, and the rectangle it
occupies (in millimeters, the device-independent coordinate system). The renderer
converts mm -> px at the target DPI.

Standard 12-lead is the canonical 3x4 grid where each column is a 2.5s time window:

    row 0:  I    aVR   V1   V4
    row 1:  II   aVL   V2   V5
    row 2:  III  aVF   V3   V6
    [rhythm: lead II, full 10s, across the bottom]
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Page margins / sizing (mm) --------------------------------------------
MARGIN_LEFT_MM = 12.0
MARGIN_RIGHT_MM = 8.0
MARGIN_TOP_MM = 18.0      # header block
MARGIN_BOTTOM_MM = 8.0
ROW_HEIGHT_MM = 40.0      # +/-2 mV headroom at 10 mm/mV
STRIP_SECONDS = 2.5       # per-column time window in the grid

# Canonical 3x4 lead arrangement (rows x columns).
GRID_ROWS: list[list[str]] = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]


@dataclass
class PanelSpec:
    """One lead strip on the page."""

    lead: str
    t_start_s: float          # first second of signal shown
    t_dur_s: float            # seconds of signal shown
    x_mm: float               # panel rectangle, top-left + size, in mm
    y_mm: float
    w_mm: float
    h_mm: float
    bbox_key: str = ""        # key used in metadata lead_bboxes (defaults to lead)

    def __post_init__(self) -> None:
        if not self.bbox_key:
            self.bbox_key = self.lead

    @property
    def baseline_y_mm(self) -> float:
        """Vertical center of the panel — the 0 mV isoelectric line."""
        return self.y_mm + self.h_mm / 2.0


@dataclass
class LayoutSpec:
    """A full page layout."""

    template: str
    page_w_mm: float
    page_h_mm: float
    panels: list[PanelSpec]
    # Left-margin calibration pulse drawn at the start of each of these rows (y_mm).
    calibration_row_baselines_mm: list[float] = field(default_factory=list)


def build_standard_12lead(*, rhythm: bool = True,
                          strip_seconds: float = STRIP_SECONDS,
                          style=None) -> LayoutSpec:
    """Build the standard 3x4 layout, optionally with a 10s rhythm strip (II).

    plan §5.2: "Standard 12-lead with rhythm strip: same + full 10s lead II."
    Geometry (margins, row height, strip length, template) comes from ``style``
    when provided, so the page layout itself varies per image.
    """
    if style is not None:
        margin_left = style.margin_left_mm
        margin_right = style.margin_right_mm
        margin_top = style.margin_top_mm
        margin_bottom = style.margin_bottom_mm
        row_height = style.row_height_mm
        strip_seconds = style.strip_seconds
        rhythm = style.template == "standard_12lead_rhythm"
    else:
        margin_left, margin_right = MARGIN_LEFT_MM, MARGIN_RIGHT_MM
        margin_top, margin_bottom = MARGIN_TOP_MM, MARGIN_BOTTOM_MM
        row_height = ROW_HEIGHT_MM

    n_cols = len(GRID_ROWS[0])
    col_w = strip_seconds * 25.0  # mm at 25 mm/s reference; renderer rescales by speed
    # NB: column *width in mm* is defined at the standard 25 mm/s so the page has a
    # fixed physical size; the renderer maps signal time -> mm using actual speed.

    panels: list[PanelSpec] = []
    cal_baselines: list[float] = []

    for r, row_leads in enumerate(GRID_ROWS):
        y = margin_top + r * row_height
        cal_baselines.append(y + row_height / 2.0)
        for c, lead in enumerate(row_leads):
            x = margin_left + c * col_w
            panels.append(PanelSpec(
                lead=lead,
                t_start_s=c * strip_seconds,
                t_dur_s=strip_seconds,
                x_mm=x, y_mm=y, w_mm=col_w, h_mm=row_height,
            ))

    page_w = margin_left + n_cols * col_w + margin_right
    n_rows = len(GRID_ROWS) + (1 if rhythm else 0)
    page_h = margin_top + n_rows * row_height + margin_bottom

    template = "standard_12lead"
    if rhythm:
        template = "standard_12lead_rhythm"
        y = margin_top + len(GRID_ROWS) * row_height
        cal_baselines.append(y + row_height / 2.0)
        panels.append(PanelSpec(
            lead="II",
            t_start_s=0.0,
            t_dur_s=strip_seconds * n_cols,   # full row duration (10s)
            x_mm=margin_left, y_mm=y,
            w_mm=n_cols * col_w, h_mm=row_height,
            bbox_key="II_rhythm",
        ))

    return LayoutSpec(
        template=template,
        page_w_mm=page_w,
        page_h_mm=page_h,
        panels=panels,
        calibration_row_baselines_mm=cal_baselines,
    )
