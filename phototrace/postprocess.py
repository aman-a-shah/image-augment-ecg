"""Signal post-processing for digitized waveforms (plan §9.1 Stage 4).

Resampling to a target rate, baseline-wander removal (high-pass), and amplitude
calibration against the rendered 1 mV reference.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps


def resample_signal(sig: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    """Resample a 1-D signal from src_fs to dst_fs."""
    sig = np.asarray(sig, np.float64)
    if src_fs == dst_fs or sig.size == 0:
        return sig.astype(np.float32)
    n_out = int(round(len(sig) * dst_fs / src_fs))
    return sps.resample(sig, n_out).astype(np.float32)


def remove_baseline_wander(sig: np.ndarray, fs: int, *, cutoff_hz: float = 0.5
                           ) -> np.ndarray:
    """High-pass filter to remove low-frequency baseline drift (plan §9.1)."""
    sig = np.asarray(sig, np.float64)
    if sig.size < 12:
        return sig.astype(np.float32)
    nyq = fs / 2.0
    wn = min(0.99, cutoff_hz / nyq)
    b, a = sps.butter(2, wn, btype="highpass")
    return sps.filtfilt(b, a, sig).astype(np.float32)


def calibrate_amplitude(sig_mv: np.ndarray, *, measured_1mv_px: float,
                        expected_1mv_px: float) -> np.ndarray:
    """Rescale amplitude using the calibration pulse as reference (plan §9.1).

    If the detected calibration pulse height differs from the expected 1 mV
    height, correct the gain.
    """
    if measured_1mv_px <= 0:
        return np.asarray(sig_mv, np.float32)
    scale = expected_1mv_px / measured_1mv_px
    return (np.asarray(sig_mv, np.float64) * scale).astype(np.float32)
