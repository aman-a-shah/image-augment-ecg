"""Digitization accuracy metrics (plan §11).

DTW distance (timing-robust), QRS peak detection F1, and heart-rate error —
computed on reconstructed millivolt signals.
"""

from __future__ import annotations

import numpy as np


def dtw_distance(a: np.ndarray, b: np.ndarray, *, band: int | None = None) -> float:
    """Dynamic time warping distance between two 1-D signals (plan §11).

    Optional Sakoe-Chiba ``band`` keeps it O(n*band) for long signals.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    if band is None:
        band = max(n, m)
    INF = float("inf")
    prev = np.full(m + 1, INF)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, INF)
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[m]) / (n + m)


def detect_r_peaks(sig: np.ndarray, fs: int, *, refractory_s: float = 0.2,
                   thresh_k: float = 0.4) -> np.ndarray:
    """Simple R-peak detector: prominent local maxima with a refractory period."""
    sig = np.asarray(sig, np.float64)
    if sig.size < 3:
        return np.array([], dtype=int)
    x = sig - np.median(sig)
    thr = thresh_k * np.percentile(np.abs(x), 99)
    refr = int(refractory_s * fs)
    peaks: list[int] = []
    i = 1
    while i < len(x) - 1:
        if x[i] > thr and x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            if peaks and i - peaks[-1] < refr:
                if x[i] > x[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
        i += 1
    return np.array(peaks, dtype=int)


def peak_f1(pred: np.ndarray, gt: np.ndarray, fs: int, *, tol_ms: float = 50.0
            ) -> dict[str, float]:
    """Match predicted vs ground-truth R-peaks within a tolerance (plan §11)."""
    pp = detect_r_peaks(pred, fs)
    gp = detect_r_peaks(gt, fs)
    tol = tol_ms / 1000.0 * fs
    matched_gt = set()
    tp = 0
    for p in pp:
        for j, g in enumerate(gp):
            if j in matched_gt:
                continue
            if abs(p - g) <= tol:
                tp += 1
                matched_gt.add(j)
                break
    fp = len(pp) - tp
    fn = len(gp) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "n_pred": len(pp), "n_gt": len(gp)}


def heart_rate_bpm(sig: np.ndarray, fs: int) -> float:
    """Mean heart rate (bpm) from detected R-peaks."""
    peaks = detect_r_peaks(sig, fs)
    if len(peaks) < 2:
        return 0.0
    rr = np.diff(peaks) / fs
    return float(60.0 / np.mean(rr))


def signal_correlation(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, np.float64)
    gt = np.asarray(gt, np.float64)
    if pred.std() < 1e-9 or gt.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(pred, gt)[0, 1])
