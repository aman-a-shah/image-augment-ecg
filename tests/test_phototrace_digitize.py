"""Phase 7 gate: digitization data, losses, metrics, post-processing, training."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Only the training test is genuinely slow; the rest (losses/metrics) are fast,
# so we mark the slow one individually below rather than the whole module.

from phototrace.digitize_data import StripDataset, build_strip_examples, y_to_mv  # noqa: E402
from phototrace.losses import morphology_weighted_loss, qrs_peak_mask  # noqa: E402
from phototrace.metrics import (  # noqa: E402
    detect_r_peaks, dtw_distance, heart_rate_bpm, peak_f1, signal_correlation,
)
from phototrace.postprocess import remove_baseline_wander, resample_signal  # noqa: E402
from phototrace.train_digitize import train_digitizer  # noqa: E402

from .conftest import DUMMY_XML, make_synthetic_record


# --- metrics ----------------------------------------------------------------
def test_dtw_identical_is_zero():
    a = np.sin(np.linspace(0, 10, 200))
    assert dtw_distance(a, a) < 1e-9


def test_dtw_shift_tolerant():
    a = np.sin(np.linspace(0, 10, 200))
    b = np.sin(np.linspace(0, 10, 200) + 0.05)
    assert dtw_distance(a, b) < dtw_distance(a, np.zeros_like(a))


def test_detect_r_peaks_on_synthetic():
    fs = 500
    t = np.arange(0, 4, 1 / fs)
    sig = np.zeros_like(t)
    true = (np.arange(0.5, 4, 0.8) * fs).astype(int)  # ~75 bpm
    sig[true] = 2.0
    peaks = detect_r_peaks(sig, fs)
    assert abs(len(peaks) - len(true)) <= 1


def test_peak_f1_identical_is_one():
    fs = 500
    sig = np.zeros(2000)
    sig[(np.arange(0.5, 4, 0.8) * fs).astype(int)] = 2.0
    res = peak_f1(sig, sig, fs)
    assert res["f1"] == 1.0


def test_heart_rate_plausible():
    fs = 500
    sig = np.zeros(2000)
    sig[(np.arange(0.4, 4, 0.8) * fs).astype(int)] = 2.0  # 0.8s RR -> 75 bpm
    assert 65 < heart_rate_bpm(sig, fs) < 85


def test_signal_correlation_bounds():
    a = np.random.default_rng(0).standard_normal(100)
    assert abs(signal_correlation(a, a) - 1.0) < 1e-6
    assert signal_correlation(a, np.zeros_like(a)) == 0.0


# --- losses -----------------------------------------------------------------
def test_morphology_loss_zero_when_equal():
    y = torch.rand(4, 128)
    loss = morphology_weighted_loss(y, y)
    assert loss.item() < 1e-4


def test_morphology_loss_positive_and_peak_weighted():
    gt = torch.rand(2, 128)
    pred = gt + 0.1
    assert morphology_weighted_loss(pred, gt).item() > 0
    mask = qrs_peak_mask(gt)
    assert mask.shape == gt.shape
    assert ((mask >= 0) & (mask <= 1)).all()


# --- post-processing --------------------------------------------------------
def test_resample_changes_length():
    sig = np.sin(np.linspace(0, 10, 500))
    out = resample_signal(sig, 500, 250)
    assert len(out) == 250


def test_baseline_removal_reduces_drift():
    fs = 500
    t = np.arange(0, 4, 1 / fs)
    drift = 2.0 * np.sin(2 * np.pi * 0.1 * t)        # slow wander
    ecg = np.sin(2 * np.pi * 1.2 * t)
    out = remove_baseline_wander(ecg + drift, fs, cutoff_hz=0.5)
    # Low-frequency energy should drop a lot.
    assert np.abs(out).mean() < np.abs(ecg + drift).mean()


# --- strip data correctness -------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(not DUMMY_XML.exists(), reason="sample absent")
def test_strip_examples_reconstruct_signal():
    from physiorender.ingest import load_ecg
    from physiorender.render import ECGRenderer
    rec = load_ecg(DUMMY_XML, validate=False)
    render = ECGRenderer(dpi=150).render(rec)
    examples = build_strip_examples(rec, render)
    assert len(examples) == 13  # 12 leads + rhythm
    ex = next(e for e in examples if e.lead == "II")
    recon_mv = y_to_mv(ex.target_y, ex.a, ex.b)
    # Reconstructed mV should track the true lead-II signal closely.
    lead = rec.leads["II"]
    t = np.linspace(0, ex.target_y.shape[0], ex.target_y.shape[0], endpoint=False)
    # Sample gt at the panel's time window.
    from physiorender.render import build_standard_12lead
    panel = next(p for p in build_standard_12lead().panels if p.lead == "II")
    idx = np.clip(((panel.t_start_s + (t / len(t)) * panel.t_dur_s)
                  * lead.sample_rate_hz).astype(int), 0, lead.n_samples - 1)
    gt = lead.signal_mv[idx]
    assert signal_correlation(recon_mv, gt) > 0.95


# --- training ---------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(not DUMMY_XML.exists(), reason="sample absent")
def test_digitizer_learns_to_read_trace():
    from physiorender.ingest import load_ecg
    from physiorender.render import ECGRenderer
    rec = load_ecg(DUMMY_XML, validate=False)
    render = ECGRenderer(dpi=150).render(rec)
    examples = build_strip_examples(rec, render)
    train = StripDataset(examples, n_variants=8, seed=0)
    val = StripDataset(examples, n_variants=2, seed=999, augment=False)
    report = train_digitizer(train, val, epochs=25, batch_size=16, seed=0)
    assert report.train_losses[-1] < report.train_losses[0]
    assert report.val_corr > 0.9, report.val_corr
    assert report.beats_baseline
