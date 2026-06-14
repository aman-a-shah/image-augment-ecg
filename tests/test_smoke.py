"""Phase 0 gate: packages import clean and core constants exist."""

from __future__ import annotations


def test_packages_import():
    import physiorender
    import phototrace

    assert physiorender.__version__
    assert phototrace.__version__


def test_public_contracts_exported():
    from physiorender import AugmentationParams, ECGMetadata, Layout, LeadData

    assert AugmentationParams is not None
    assert ECGMetadata is not None
    assert Layout is not None
    assert LeadData is not None


def test_config_constants():
    from physiorender import config

    assert config.DEFAULT_DPI == 300
    assert config.SMALL_GRID_MM == 1.0
    assert config.LARGE_GRID_MM == 5.0
    assert len(config.STANDARD_LEADS) == 12
    # 25.4mm == 1 inch == DEFAULT_DPI pixels
    assert round(config.mm_to_px(25.4)) == config.DEFAULT_DPI
    assert round(config.px_to_mm(config.DEFAULT_DPI)) == 25
