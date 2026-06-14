"""ECG file format detection (plan §4.1).

Format-adaptive from the start: we sniff the file rather than trusting the
extension, and fall back to a heuristic generic parser for unknown flavours.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path


class ECGFormat(str, Enum):
    GE_MUSE = "GE_MUSE"            # <RestingECG>, base64 int16 waveforms
    PHILIPS = "PHILIPS"           # Philips <restingecgdata>
    HL7_AECG = "HL7_AECG"         # HL7 annotated ECG
    CARDIOLOGY_XML = "CARDIOLOGY_XML"  # GE CardioSoft <CardiologyXML>, CSV ints
    WFDB = "WFDB"                 # PhysioNet WFDB (.hea/.dat) — not XML
    UNKNOWN = "UNKNOWN"           # trigger generic heuristic parser


def _local_tag(tag: str) -> str:
    """Strip an XML namespace from a tag: '{ns}Foo' -> 'foo'."""
    return tag.split("}")[-1].lower()


def detect_ecg_xml_format(filepath: str | Path) -> ECGFormat:
    """Detect the ECG format of a file (plan §4.1).

    WFDB is detected by extension (it is not XML). XML flavours are detected
    from the root tag / namespace.
    """
    path = Path(filepath)

    # WFDB is a binary format described by a .hea header — not XML.
    if path.suffix.lower() in {".hea", ".dat"}:
        return ECGFormat.WFDB

    try:
        # Parse only enough to read the root element.
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return ECGFormat.UNKNOWN

    tag = _local_tag(root.tag)
    ns = root.tag.lower()

    if tag == "cardiologyxml":
        return ECGFormat.CARDIOLOGY_XML
    if tag == "restingecg" or "muse" in ns:
        return ECGFormat.GE_MUSE
    if tag == "restingecgdata" or "philips" in ns:
        return ECGFormat.PHILIPS
    if "hl7" in ns or "aecg" in ns or tag == "annotatedecg":
        return ECGFormat.HL7_AECG

    return ECGFormat.UNKNOWN
