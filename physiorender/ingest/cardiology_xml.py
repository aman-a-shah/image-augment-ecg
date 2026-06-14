"""Extractor for GE CardioSoft ``<CardiologyXML>`` files.

Structure (confirmed against the project's sample):
  <CardiologyXML>
    <StripData>            <- full rhythm, preferred for rendering
      <NumberOfLeads>12</NumberOfLeads>
      <SampleRate units="Hz">500</SampleRate>
      <ChannelSampleCountTotal>5000</ChannelSampleCountTotal>
      <Resolution units="uVperLsb">5</Resolution>
      <WaveformData lead="I">12,14,14,...</WaveformData>   <- CSV int counts
      ...
    </StripData>
    <MedianSamples> ... </MedianSamples>   <- averaged beat, used as fallback

Waveforms are comma-separated decimal counts (not base64). Counts are scaled to
millivolts via Resolution (uV per LSB).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ..logging_setup import get_logger
from .base import SignalExtractor
from .record import ECGRecord, LeadSignal

log = get_logger(__name__)

# Sample indices of fiducial points the device reports (optional, useful later).
_MEASUREMENT_TAGS = ("POnset", "POffset", "QOnset", "QOffset", "TOffset")


class CardiologyXMLExtractor(SignalExtractor):
    format_name = "CARDIOLOGY_XML"

    def extract(self, filepath: str | Path) -> ECGRecord:
        path = Path(filepath)
        root = ET.parse(path).getroot()

        # Prefer the full rhythm strip; fall back to the median beat.
        block = root.find("StripData")
        block_name = "StripData"
        if block is None:
            block = root.find("MedianSamples")
            block_name = "MedianSamples"
        if block is None:
            raise ValueError(
                f"{path}: no <StripData> or <MedianSamples> waveform block found"
            )

        sample_rate = int(float(self._text(block, "SampleRate", default="500")))
        uv_per_lsb = float(self._text(block, "Resolution", default="5"))

        leads: dict[str, LeadSignal] = {}
        lead_order: list[str] = []
        for wf in block.findall("WaveformData"):
            name = wf.get("lead")
            if not name:
                continue
            counts = self.parse_csv_ints(wf.text or "")
            signal_mv = self.counts_to_mv(counts, uv_per_lsb)
            leads[name] = LeadSignal(name=name, signal_mv=signal_mv,
                                     sample_rate_hz=sample_rate)
            lead_order.append(name)

        if not leads:
            raise ValueError(f"{path}: <{block_name}> contained no lead waveforms")

        # Preserve the device's declared LeadOrder when available.
        declared = root.findtext(".//LeadOrder")
        if declared:
            ordered = [s.strip() for s in declared.split(",") if s.strip()]
            if set(ordered) == set(lead_order):
                lead_order = ordered

        measurements = self._extract_measurements(block, root)

        log.info(
            "CardiologyXML: %d leads, %d Hz, %.2fs (%s, %g uV/LSB)",
            len(leads), sample_rate,
            next(iter(leads.values())).duration_s, block_name, uv_per_lsb,
        )

        return ECGRecord(
            leads=leads,
            sample_rate_hz=sample_rate,
            source_path=str(path),
            source_format=self.format_name,
            lead_order=lead_order,
            measurements=measurements,
            meta={"waveform_block": block_name, "uv_per_lsb": uv_per_lsb},
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _text(elem: ET.Element, tag: str, *, default: str) -> str:
        child = elem.find(tag)
        if child is None or child.text is None or not child.text.strip():
            return default
        return child.text.strip()

    @staticmethod
    def _extract_measurements(block: ET.Element, root: ET.Element) -> dict[str, float]:
        out: dict[str, float] = {}
        for tag in _MEASUREMENT_TAGS:
            # Fiducials may live in the block or the measurements section.
            text = block.findtext(tag) or root.findtext(f".//{tag}")
            if text and text.strip():
                try:
                    out[tag] = float(text.strip())
                except ValueError:
                    pass
        return out
