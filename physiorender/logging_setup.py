"""Shared logging configuration.

Use ``get_logger(__name__)`` everywhere instead of ``print``. Data-quality
issues (plan §4.2) should be logged at WARNING so they surface during batch
generation without halting it.
"""

from __future__ import annotations

import logging
import os

_DEFAULT_LEVEL = os.environ.get("PHYSIORENDER_LOG_LEVEL", "INFO").upper()
_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=_DEFAULT_LEVEL,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    _configure_root()
    return logging.getLogger(name)
