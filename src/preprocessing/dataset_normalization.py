"""
Dataset-level preprocessing for vibration signals.

This module is intentionally independent from the existing window-level
normalization functions used by previous experiments.

Expected signal layout
----------------------
- A single signal: shape (n_samples,)
- A collection of signals/windows: shape (n_signals, n_samples)

The module does NOT decide which samples are normal. The caller must provide
only the reference normal signals when estimating dataset statistics. This is
important for preventing train/test leakage in LOCO experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import detrend


NormalizationMode = Literal[
    "none",
    "rms",
    "detrend",
    "rms_detrend",
]


@dataclass(frozen=True)
class DatasetNormalizationReference:
    """Reference statistics estimated from TRAINING normal signals only."""

    rms_mean: float | None = None


def _as_float_array(signal: np.ndarray) -> np.ndarray:
    """Validate and convert an input signal/collection to float64."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError(
            "signal must have shape (n_samples,) or (n_signals, n_samples)"
        )
    if arr.size == 0:
        raise ValueError("signal cannot be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("signal contains NaN or infinite values")
    return arr


def linear_detrend(signal: np.ndarray) -> np.ndarray:
    """Remove the best-fit linear trend from a signal or each signal row."""
    arr = _as_float_array(signal)
    return detrend(arr, axis=-1, type="linear")


def signal_rms(signal: np.ndarray) -> np.ndarray | float:
    """Return RMS per signal/window.

    For shape (n_samples,), returns one float.
    For shape (n_signals, n_samples), returns one RMS value per row.
    """
    arr = _as_float_array(signal)
    rms = np.sqrt(np.mean(np.square(arr), axis=-1))
    return float(rms) if arr.ndim == 1 else rms


def mean_normal_rms(
    normal_signals: np.ndarray,
    *,
    detrend_before_rms: bool = False,
) -> float:
    """Compute the mean RMS of the supplied normal signals.

    The statistic is the arithmetic mean of the RMS of each normal signal:

        RMS_ref = (1/N) * sum_i RMS(x_i)

    This is intentionally different from computing one RMS over all samples
    concatenated together.
    """
    arr = _as_float_array(normal_signals)
    if arr.ndim != 2:
        raise ValueError("normal_signals must have shape (n_signals, n_samples)")

    if detrend_before_rms:
        arr = linear_detrend(arr)

    rms_values = np.asarray(signal_rms(arr), dtype=np.float64)
    ref = float(np.mean(rms_values))

    if not np.isfinite(ref) or ref <= 0.0:
        raise ValueError(f"invalid RMS reference: {ref}")

    return ref


def build_reference_from_normal_signals(
    normal_signals: np.ndarray,
    *,
    detrend_before_rms: bool = False,
) -> DatasetNormalizationReference:
    """Build a normalization reference from training normal signals."""
    return DatasetNormalizationReference(
        rms_mean=mean_normal_rms(
            normal_signals,
            detrend_before_rms=detrend_before_rms,
        )
    )


def normalize_by_reference_rms(
    signal: np.ndarray,
    reference_rms: float,
) -> np.ndarray:
    """Scale every signal/window by one externally estimated RMS reference."""
    arr = _as_float_array(signal)
    ref = float(reference_rms)
    if not np.isfinite(ref) or ref <= 0.0:
        raise ValueError("reference_rms must be a finite positive number")
    return arr / ref


def apply_dataset_normalization(
    signal: np.ndarray,
    mode: NormalizationMode,
    *,
    reference_rms: float | None = None,
) -> np.ndarray:
    """Apply one dataset-level preprocessing strategy.

    Modes
    -----
    none:
        No transformation.
    rms:
        Divide by one RMS reference estimated from training normal signals.
    detrend:
        Remove a linear trend from the signal itself.
    rms_detrend:
        Remove the linear trend first, then divide by one RMS reference.

    The reference is supplied by the caller on purpose. The module never
    estimates it from the samples being transformed, which avoids accidental
    test-set leakage.
    """
    if mode not in {"none", "rms", "detrend", "rms_detrend"}:
        raise ValueError(f"unknown normalization mode: {mode}")

    arr = _as_float_array(signal)

    if mode == "none":
        return arr.copy()

    if mode == "rms":
        if reference_rms is None:
            raise ValueError("reference_rms is required for mode='rms'")
        return normalize_by_reference_rms(arr, reference_rms)

    if mode == "detrend":
        return linear_detrend(arr)

    # mode == "rms_detrend"
    detrended = linear_detrend(arr)
    if reference_rms is None:
        raise ValueError("reference_rms is required for mode='rms_detrend'")
    return normalize_by_reference_rms(detrended, reference_rms)


def apply_reference_to_collection(
    signals: np.ndarray,
    mode: NormalizationMode,
    reference: DatasetNormalizationReference | None = None,
) -> np.ndarray:
    """Convenience wrapper for a collection of signals/windows."""
    reference_rms = None if reference is None else reference.rms_mean
    return apply_dataset_normalization(
        signals,
        mode,
        reference_rms=reference_rms,
    )
