"""Clean ECG layout renderer (plan §5).

Renders an :class:`ECGRecord` to a photorealistic *clean* ECG printout at 300 DPI:
warm-cream paper, pink 1mm/5mm grid, anti-aliased near-black trace, a 1mV
calibration pulse per row, and a header block. Emits per-lead bounding boxes for
downstream supervision (plan §8).

Anti-aliasing is achieved by supersampling: draw at SS x resolution, then
downsample with LANCZOS (plan §5.3 — printers use smooth curves, not stepped lines).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import config
from ..ingest.record import ECGRecord
from ..logging_setup import get_logger
from .layout import LayoutSpec, PanelSpec, build_standard_12lead
from .style import RenderStyle, default_style

log = get_logger(__name__)

_CAL_PULSE_MV = 1.0       # calibration pulse height
_CAL_PULSE_S = 0.2        # calibration pulse width (seconds)
_CAL_GAP_MM = 3.0         # gap between cal pulse and panel start


@dataclass
class RenderResult:
    """Output of a render: the clean image plus geometry metadata (plan §8)."""

    image: Image.Image
    lead_bboxes: dict[str, list[int]]   # bbox_key -> [x1, y1, x2, y2] in image px
    template: str
    paper_speed_mm_s: int
    gain_mm_mv: int
    dpi: int
    layout: LayoutSpec | None = None    # the exact layout used (varies with style)
    style: RenderStyle | None = None
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path)


class ECGRenderer:
    """Render clean ECG printouts from waveform records."""

    def __init__(
        self,
        *,
        dpi: int = config.DEFAULT_DPI,
        paper_speed_mm_s: int | None = None,
        gain_mm_mv: int | None = None,
        style: RenderStyle | None = None,
        supersample: int = 2,
    ) -> None:
        self.dpi = dpi
        self.style = style or default_style()
        # Explicit speed/gain args override the style (back-compat).
        if paper_speed_mm_s is not None:
            self.style.paper_speed_mm_s = paper_speed_mm_s
        if gain_mm_mv is not None:
            self.style.gain_mm_mv = gain_mm_mv
        self.paper_speed_mm_s = self.style.paper_speed_mm_s
        self.gain_mm_mv = self.style.gain_mm_mv
        self.ss = max(1, int(supersample))
        # pixels per mm in the supersampled drawing space
        self._ppm = config.mm_to_px(1.0, dpi) * self.ss

    # ------------------------------------------------------------------ #
    def render(self, record: ECGRecord, layout: LayoutSpec | None = None) -> RenderResult:
        if layout is None:
            layout = build_standard_12lead(style=self.style)

        W = int(round(layout.page_w_mm * self._ppm))
        H = int(round(layout.page_h_mm * self._ppm))
        img = Image.new("RGB", (W, H), self.style.bg_color)
        draw = ImageDraw.Draw(img)

        self._draw_grid(draw, layout)
        if self.style.show_header:
            self._draw_header(draw, record, layout)

        bboxes_ss: dict[str, list[int]] = {}
        for panel in layout.panels:
            if panel.lead not in record.leads:
                log.warning("layout references missing lead %r; skipping", panel.lead)
                continue
            self._draw_panel(draw, record, panel)
            bboxes_ss[panel.bbox_key] = self._panel_bbox_px(panel)

        if self.style.show_calibration:
            self._draw_calibration_pulses(draw, layout)

        # Downsample for anti-aliasing.
        if self.ss > 1:
            img = img.resize(
                (W // self.ss, H // self.ss), resample=Image.LANCZOS
            )
        bboxes = {k: [int(round(v / self.ss)) for v in box]
                  for k, box in bboxes_ss.items()}

        return RenderResult(
            image=img,
            lead_bboxes=bboxes,
            template=layout.template,
            paper_speed_mm_s=self.paper_speed_mm_s,
            gain_mm_mv=self.gain_mm_mv,
            dpi=self.dpi,
            layout=layout,
            style=self.style,
            meta={"page_w_mm": layout.page_w_mm, "page_h_mm": layout.page_h_mm},
        )

    # ------------------------------------------------------------------ #
    # Coordinate transforms (supersampled px)
    # ------------------------------------------------------------------ #
    def _mm_x(self, mm: float) -> float:
        return mm * self._ppm

    def _mm_y(self, mm: float) -> float:
        return mm * self._ppm

    def _x_for_time(self, panel: PanelSpec, t_rel_s: float) -> float:
        """Map a time offset within the panel to an x pixel coordinate."""
        return self._mm_x(panel.x_mm) + t_rel_s * self.paper_speed_mm_s * self._ppm

    def _y_for_mv(self, panel: PanelSpec, mv: float) -> float:
        baseline = self._mm_y(panel.baseline_y_mm)
        return baseline - mv * self.gain_mm_mv * self._ppm

    def _panel_bbox_px(self, panel: PanelSpec) -> list[int]:
        x1 = int(round(self._mm_x(panel.x_mm)))
        y1 = int(round(self._mm_y(panel.y_mm)))
        x2 = int(round(self._mm_x(panel.x_mm + panel.w_mm)))
        y2 = int(round(self._mm_y(panel.y_mm + panel.h_mm)))
        return [x1, y1, x2, y2]

    # ------------------------------------------------------------------ #
    # Drawing primitives
    # ------------------------------------------------------------------ #
    def _draw_grid(self, draw: ImageDraw.ImageDraw, layout: LayoutSpec) -> None:
        w_mm, h_mm = layout.page_w_mm, layout.page_h_mm
        small_w = max(1, int(round(0.12 * self._ppm)))
        large_w = max(1, int(round(0.25 * self._ppm)))
        minor_color = self.style.faded(self.style.grid_minor_color)
        major_color = self.style.faded(self.style.grid_major_color)

        # Small 1mm grid (optional)
        if self.style.show_minor_grid:
            n_x = int(w_mm / config.SMALL_GRID_MM)
            n_y = int(h_mm / config.SMALL_GRID_MM)
            for i in range(n_x + 1):
                x = self._mm_x(i * config.SMALL_GRID_MM)
                draw.line([(x, 0), (x, h_mm * self._ppm)], fill=minor_color, width=small_w)
            for j in range(n_y + 1):
                y = self._mm_y(j * config.SMALL_GRID_MM)
                draw.line([(0, y), (w_mm * self._ppm, y)], fill=minor_color, width=small_w)

        # Large 5mm grid (drawn over small)
        n_x = int(w_mm / config.LARGE_GRID_MM)
        n_y = int(h_mm / config.LARGE_GRID_MM)
        for i in range(n_x + 1):
            x = self._mm_x(i * config.LARGE_GRID_MM)
            draw.line([(x, 0), (x, h_mm * self._ppm)], fill=major_color, width=large_w)
        for j in range(n_y + 1):
            y = self._mm_y(j * config.LARGE_GRID_MM)
            draw.line([(0, y), (w_mm * self._ppm, y)], fill=major_color, width=large_w)

    def _draw_panel(self, draw: ImageDraw.ImageDraw,
                    record: ECGRecord, panel: PanelSpec) -> None:
        lead = record.leads[panel.lead]
        fs = lead.sample_rate_hz
        i0 = max(0, int(round(panel.t_start_s * fs)))
        i1 = min(lead.n_samples, int(round((panel.t_start_s + panel.t_dur_s) * fs)))
        if i1 <= i0:
            return
        seg = lead.signal_mv[i0:i1]

        pts: list[tuple[float, float]] = []
        for k, mv in enumerate(seg):
            t_rel = k / fs
            pts.append((self._x_for_time(panel, t_rel), self._y_for_mv(panel, float(mv))))

        trace_w = max(1, int(round(self.style.trace_width_mm * self._ppm)))
        draw.line(pts, fill=self.style.trace_color, width=trace_w, joint="curve")

        # Lead label, top-left inside the panel.
        self._text(draw, panel.x_mm + 1.0, panel.y_mm + 1.0, panel.lead,
                   size_mm=3.0 * self.style.label_scale, bold=True)

    def _draw_calibration_pulses(self, draw: ImageDraw.ImageDraw,
                                 layout: LayoutSpec) -> None:
        """Draw a 1mV / 0.2s rectangular pulse in the left margin of each row."""
        pulse_h_px = _CAL_PULSE_MV * self.gain_mm_mv * self._ppm
        pulse_w_px = _CAL_PULSE_S * self.paper_speed_mm_s * self._ppm
        x_start = self._mm_x(layout.panels[0].x_mm) - self._mm_x(_CAL_GAP_MM) - pulse_w_px
        x_start = max(self._mm_x(1.0), x_start)
        trace_w = max(1, int(round(self.style.trace_width_mm * self._ppm)))

        for baseline_mm in layout.calibration_row_baselines_mm:
            b = self._mm_y(baseline_mm)
            top = b - pulse_h_px
            pts = [
                (x_start, b),
                (x_start, top),
                (x_start + pulse_w_px, top),
                (x_start + pulse_w_px, b),
            ]
            draw.line(pts, fill=self.style.trace_color, width=trace_w, joint="curve")

    def _draw_header(self, draw: ImageDraw.ImageDraw,
                     record: ECGRecord, layout: LayoutSpec) -> None:
        name = Path(record.source_path).stem
        line1 = f"ECG  |  {name}"
        line2 = (f"{self.paper_speed_mm_s} mm/s   {self.gain_mm_mv} mm/mV   "
                 f"{record.sample_rate_hz} Hz   {record.n_leads} leads")
        hx = layout.panels[0].x_mm if layout.panels else 12.0
        self._text(draw, hx, 4.0, line1, size_mm=4.0 * self.style.header_scale, bold=True)
        self._text(draw, hx, 10.0, line2, size_mm=3.0 * self.style.header_scale)

    # ------------------------------------------------------------------ #
    def _text(self, draw: ImageDraw.ImageDraw, x_mm: float, y_mm: float,
              text: str, *, size_mm: float, bold: bool = False) -> None:
        font = _load_font(int(round(size_mm * self._ppm)), bold=bold)
        draw.text((self._mm_x(x_mm), self._mm_y(y_mm)), text,
                  fill=self.style.trace_color, font=font)


# Font loading: prefer a real TTF for crisp labels, fall back to PIL default.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "DejaVuSans.ttf",
]
_FONT_BOLD_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "DejaVuSans-Bold.ttf",
]


def _load_font(size_px: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = _FONT_BOLD_CANDIDATES if bold else _FONT_CANDIDATES
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=max(8, size_px))
        except OSError:
            continue
    return ImageFont.load_default()
