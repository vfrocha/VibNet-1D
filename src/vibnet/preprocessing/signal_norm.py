"""
Reference RMS normalization and linear detrending (`src.vibnet.preprocessing.signal_norm`).

Implements ADR-001 (`docs/decisions/001-rms-reference-normalization-and-detrending.md`):
- Linear detrending removes slow thermal/sensor drift per 2,048-sample window.
- Reference RMS normalization divides all windows of a dataset by the mean RMS
  of its healthy reference class (`RMS_ref`), preserving relative fault-to-healthy
  energy severity while unifying scale across datasets.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

EPSILON_RMS: float = 1e-12


def compute_window_rms(window: Sequence[float]) -> float:
    """Compute the Root Mean Square (RMS) of a 1D signal window."""
    n = len(window)
    if n == 0:
        raise ValueError("Cannot compute RMS of an empty signal window.")
    mean_sq = sum(float(x) * float(x) for x in window) / float(n)
    return math.sqrt(mean_sq)


def compute_reference_rms(
    normal_windows: Iterable[Sequence[float]],
    eps: float = EPSILON_RMS,
) -> float:
    """
    Compute the dataset reference RMS (`RMS_ref`) as the arithmetic mean of
    window-level RMS values over healthy (`normal_class`) windows.
    """
    rms_values = [compute_window_rms(w) for w in normal_windows]
    if not rms_values:
        raise ValueError("No normal-class windows provided to compute reference RMS.")
    mean_rms = sum(rms_values) / float(len(rms_values))
    return max(mean_rms, eps)


def detrend_linear_window(window: Sequence[float]) -> List[float]:
    """
    Remove best-fit straight line (`a * t + b`) from a 1D window in pure Python
    (matching `scipy.signal.detrend(..., type='linear')`).
    """
    n = len(window)
    if n <= 1:
        return [0.0 for _ in window]
    t_mean = (n - 1) / 2.0
    y_vals = [float(v) for v in window]
    y_mean = sum(y_vals) / float(n)

    num = 0.0
    den = 0.0
    for idx, y in enumerate(y_vals):
        dt = idx - t_mean
        num += dt * (y - y_mean)
        den += dt * dt

    slope = (num / den) if den > 0.0 else 0.0
    intercept = y_mean - slope * t_mean
    return [y - (slope * idx + intercept) for idx, y in enumerate(y_vals)]


def normalize_windows_by_reference(
    windows: Iterable[Sequence[float]],
    rms_ref: float,
    detrend: bool = False,
    eps: float = EPSILON_RMS,
) -> List[List[float]]:
    """
    Normalize a sequence of 1D windows by `max(rms_ref, eps)`, optionally applying
    linear detrending first (`detrended_rms` variant).
    """
    denom = max(float(rms_ref), eps)
    normalized: List[List[float]] = []
    for w in windows:
        processed = detrend_linear_window(w) if detrend else [float(x) for x in w]
        normalized.append([val / denom for val in processed])
    return normalized
