"""Phase 8 gate: CLI, end-to-end digitization pipeline, domain-gap harness."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from physiorender.cli import main as cli_main  # noqa: E402
from physiorender.ingest import load_ecg  # noqa: E402
from physiorender.render import ECGRenderer, build_standard_12lead  # noqa: E402

from phototrace.digitize_data import StripDataset, build_strip_examples  # noqa: E402
from phototrace.domain_gap import compare, naive_digitize  # noqa: E402
from phototrace.metrics import signal_correlation  # noqa: E402
from phototrace.pipeline import DigitizationPipeline, panel_geometry  # noqa: E402
from phototrace.train_digitize import train_digitizer  # noqa: E402

from .conftest import DUMMY_XML


# --- CLI --------------------------------------------------------------------
@pytest.mark.skipif(not DUMMY_XML.exists(), reason="sample absent")
def test_cli_inspect():
    assert cli_main(["inspect", str(DUMMY_XML)]) == 0


@pytest.mark.skipif(not DUMMY_XML.exists(), reason="sample absent")
def test_cli_render(tmp_path):
    rc = cli_main(["render", str(DUMMY_XML), "--dpi", "100", "--out", str(tmp_path)])
    assert rc == 0
    assert list(tmp_path.glob("*_clean.png"))


# --- geometry helpers -------------------------------------------------------
def test_panel_geometry_within_bounds():
    geoms = panel_geometry(1024, 700, 10)
    assert len(geoms) == 13
    for g in geoms:
        x1, y1, x2, y2 = g.bbox
        assert 0 <= x1 < x2 <= 1024 and 0 <= y1 < y2 <= 700


def test_naive_digitize_returns_all_panels():
    gray = np.ones((700, 1024), np.float32)
    sig = naive_digitize(gray, 10)
    assert len(sig) == 13


# --- end-to-end (trains a small digitizer once) -----------------------------
@pytest.fixture(scope="module")
def trained():
    if not DUMMY_XML.exists():
        pytest.skip("sample absent")
    rec = load_ecg(DUMMY_XML, validate=False)
    render = ECGRenderer(dpi=150).render(rec)
    examples = build_strip_examples(rec, render)
    train = StripDataset(examples, n_variants=8, seed=0)
    val = StripDataset(examples, n_variants=2, seed=7, augment=False)
    report = train_digitizer(train, val, epochs=25, batch_size=16, seed=0)
    pipe = DigitizationPipeline(report.model, gain_mm_mv=10, target_fs=500)
    return rec, render, pipe


def test_end_to_end_digitization_recovers_signal(trained):
    rec, render, pipe = trained
    signals = pipe.digitize(render.image)
    layout = build_standard_12lead()
    panels = {p.bbox_key: p for p in layout.panels}

    corrs = []
    for key in ("II", "V2", "V5", "I"):
        panel = panels[key]
        lead = rec.leads[panel.lead]
        fs = lead.sample_rate_hz
        n = len(signals[key])
        t = panel.t_start_s + np.linspace(0, panel.t_dur_s, n, endpoint=False)
        gt = lead.signal_mv[np.clip((t * fs).astype(int), 0, lead.n_samples - 1)]
        corrs.append(signal_correlation(signals[key], gt))
    assert np.mean(corrs) > 0.85, corrs


def test_domain_gap_harness_learned_beats_naive(trained):
    rec, render, pipe = trained
    gray_clean = np.asarray(render.image.convert("L"), np.float32) / 255.0

    # Ground-truth signals keyed for the digital model (HR from lead II).
    gt_signals = {name: rec.leads[name].signal_mv for name in rec.leads}
    gt_signals["II_rhythm"] = rec.leads["II"].signal_mv

    # Simulate capture noise that breaks the naive 'darkest-row' filter.
    rng = np.random.default_rng(0)
    noisy = np.clip(gray_clean + rng.normal(0, 0.18, gray_clean.shape), 0, 1).astype(np.float32)
    speckle = rng.random(noisy.shape) < 0.02
    noisy[speckle] = 0.0

    result = compare([(noisy, gt_signals)], pipe)
    assert np.isfinite(result.naive_error) and np.isfinite(result.pipeline_error)
    # The learned digitizer should track the waveform better than the filter.
    assert result.pipeline_corr > result.naive_corr, (result.pipeline_corr,
                                                      result.naive_corr)
