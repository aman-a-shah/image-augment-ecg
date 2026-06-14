"""ECG ingestion (Phase 1).

Public entry point: :func:`load_ecg` detects the format and dispatches to the
right extractor, returning a normalized :class:`ECGRecord`. Unknown / not-yet-
implemented XML flavours fall back to the heuristic :class:`GenericXMLExtractor`.
"""

from __future__ import annotations

from pathlib import Path

from ..logging_setup import get_logger
from .base import SignalExtractor
from .cardiology_xml import CardiologyXMLExtractor
from .formats import ECGFormat, detect_ecg_xml_format
from .generic import GenericXMLExtractor
from .record import ECGRecord, LeadSignal
from .validate import ValidationReport, validate_record
from .wfdb_reader import WFDBExtractor

log = get_logger(__name__)

# Format -> extractor. Formats without a dedicated parser yet resolve to the
# heuristic generic extractor (plan §4.2: "useful for unknown flavours").
_REGISTRY: dict[ECGFormat, type[SignalExtractor]] = {
    ECGFormat.CARDIOLOGY_XML: CardiologyXMLExtractor,
    ECGFormat.WFDB: WFDBExtractor,
    ECGFormat.GE_MUSE: GenericXMLExtractor,
    ECGFormat.PHILIPS: GenericXMLExtractor,
    ECGFormat.HL7_AECG: GenericXMLExtractor,
    ECGFormat.UNKNOWN: GenericXMLExtractor,
}


def get_extractor(fmt: ECGFormat) -> SignalExtractor:
    """Return an extractor instance for the given format."""
    return _REGISTRY[fmt]()


def load_ecg(
    filepath: str | Path,
    *,
    validate: bool = True,
    expected_leads: int = 12,
) -> ECGRecord:
    """Load any supported ECG file into a normalized :class:`ECGRecord`.

    Detects the format, extracts signals, and (optionally) runs validation,
    logging warnings/errors. Validation failures are logged but do not raise —
    callers inspect ``record`` / re-run :func:`validate_record` to decide.
    """
    path = Path(filepath)
    fmt = detect_ecg_xml_format(path)
    log.info("loading %s (detected format: %s)", path.name, fmt.value)

    extractor = get_extractor(fmt)
    record = extractor.extract(path)

    if validate:
        validate_record(record, expected_leads=expected_leads)

    return record


__all__ = [
    "ECGFormat",
    "ECGRecord",
    "LeadSignal",
    "SignalExtractor",
    "CardiologyXMLExtractor",
    "GenericXMLExtractor",
    "WFDBExtractor",
    "ValidationReport",
    "detect_ecg_xml_format",
    "get_extractor",
    "load_ecg",
    "validate_record",
]
