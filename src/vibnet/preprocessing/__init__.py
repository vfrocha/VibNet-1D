"""
Signal and feature preprocessing components (`src.vibnet.preprocessing`).
"""

from src.vibnet.preprocessing.scaling import (
    StandardScalerStats,
    fit_standard_scaler_stats,
    fit_transform_target_fold,
    scale_source_domains,
    transform_with_scaler_stats,
)
from src.vibnet.preprocessing.signal_norm import (
    EPSILON_RMS,
    compute_reference_rms,
    compute_window_rms,
    detrend_linear_window,
    normalize_windows_by_reference,
)

__all__ = [
    "EPSILON_RMS",
    "StandardScalerStats",
    "compute_reference_rms",
    "compute_window_rms",
    "detrend_linear_window",
    "fit_standard_scaler_stats",
    "fit_transform_target_fold",
    "normalize_windows_by_reference",
    "scale_source_domains",
    "transform_with_scaler_stats",
]
