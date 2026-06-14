"""GenericXMLExtractor — heuristic parser for unknown XML flavours (plan §4.2).

Strategy: walk the tree, find elements that look like per-lead waveform data
(tag contains 'waveform'/'leaddata', or carries a lead/channel attribute), decode
their payload as either CSV ints or base64 int16, and scale by any resolution-like
value we can find. This is the fallback that keeps the system working on flavours
we haven't written a dedicated extractor for yet.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..logging_setup import get_logger
from .base import SignalExtractor
from .formats import _local_tag
from .record import ECGRecord, LeadSignal

log = get_logger(__name__)

_LEAD_ATTR_KEYS = ("lead", "leadid", "channel", "name", "id")
_RESOLUTION_HINTS = ("resolution", "uvperlsb", "lsbpermv", "scale", "amplituderesolution")
_RATE_HINTS = ("samplerate", "samplingrate", "sampleratehz", "frequency", "samplefrequency")
_CSV_RE = re.compile(r"^[\s\d,.\-+eE]+$")


class GenericXMLExtractor(SignalExtractor):
    format_name = "GENERIC_XML"

    def extract(self, filepath: str | Path) -> ECGRecord:
        path = Path(filepath)
        root = ET.parse(path).getroot()

        sample_rate = self._find_numeric(root, _RATE_HINTS, default=500.0)
        uv_per_lsb = self._find_numeric(root, _RESOLUTION_HINTS, default=1.0)

        leads: dict[str, LeadSignal] = {}
        lead_order: list[str] = []
        for elem in root.iter():
            name = self._lead_name(elem)
            if name is None or not (elem.text and elem.text.strip()):
                continue
            counts = self._decode_payload(elem.text)
            if counts is None or counts.size == 0:
                continue
            if name in leads:
                continue  # first occurrence wins (avoids median+strip duplicates)
            signal_mv = self.counts_to_mv(counts, uv_per_lsb)
            leads[name] = LeadSignal(name=name, signal_mv=signal_mv,
                                     sample_rate_hz=int(sample_rate))
            lead_order.append(name)

        if not leads:
            raise ValueError(
                f"{path}: GenericXMLExtractor found no decodable waveform elements"
            )

        log.warning(
            "GenericXMLExtractor used for %s: %d leads, %d Hz, %g uV/LSB "
            "(verify against a reference — heuristic parse)",
            path.name, len(leads), int(sample_rate), uv_per_lsb,
        )

        return ECGRecord(
            leads=leads,
            sample_rate_hz=int(sample_rate),
            source_path=str(path),
            source_format=self.format_name,
            lead_order=lead_order,
            meta={"uv_per_lsb": uv_per_lsb, "heuristic": True},
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _lead_name(elem: ET.Element) -> str | None:
        tag = _local_tag(elem.tag)
        looks_like_waveform = "waveform" in tag or "leaddata" in tag or tag == "lead"
        for key, val in elem.attrib.items():
            if key.lower() in _LEAD_ATTR_KEYS and val.strip():
                if looks_like_waveform or "wave" in tag or "data" in tag or tag == "lead":
                    return val.strip()
        return None

    def _decode_payload(self, text: str) -> np.ndarray | None:
        sample = text.strip()[:200]
        try:
            if "," in sample and _CSV_RE.match(sample):
                return self.parse_csv_ints(text)
            # No commas and looks like base64 -> try base64 int16.
            if re.fullmatch(r"[A-Za-z0-9+/=\s]+", sample):
                return self.decode_base64_int16(text)
        except (ValueError, Exception):  # noqa: BLE001 - heuristic, never fatal
            return None
        return None

    @staticmethod
    def _find_numeric(root: ET.Element, hints: tuple[str, ...], *, default: float) -> float:
        for elem in root.iter():
            tag = _local_tag(elem.tag)
            if any(h in tag for h in hints) and elem.text and elem.text.strip():
                try:
                    return float(elem.text.strip().split()[0])
                except ValueError:
                    continue
        return default
