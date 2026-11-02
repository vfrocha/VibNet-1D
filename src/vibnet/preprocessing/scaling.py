"""
Leak-free feature standardization for LOCO and Multi-Source Transfer Learning
(`src.vibnet.preprocessing.scaling`).

Enforces ADR-002 (`docs/decisions/002-loco-isolation-and-target-balancing.md`):
- Target `StandardScaler` is fitted **exclusively** on `X_target_train` (the K-1
  training operating conditions) and applied via `.transform()` to `X_target_test`
  (the held-out operating condition).
- Each source dataset `s` is standardized independently using its own domain
  statistics (`fit_transform(X_source_s)`), preventing domain-statistics leakage
  between sources and target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class StandardScalerStats:
    """Per-feature mean and standard deviation fitted strictly on training data."""

    mean: Tuple[float, ...]
    scale: Tuple[float, ...]
    n_samples_seen: int


def fit_standard_scaler_stats(
    X_train: Sequence[Sequence[float]],
) -> StandardScalerStats:
    """
    Compute per-feature mean and population standard deviation (`ddof=0`, matching
    `sklearn.preprocessing.StandardScaler`) strictly on `X_train`.
    """
    n_samples = len(X_train)
    if n_samples == 0:
        raise ValueError("Cannot fit StandardScalerStats on empty X_train.")
    n_features = len(X_train[0])
    if n_features == 0:
        raise ValueError("Cannot fit StandardScalerStats on 0-dimensional features.")

    means: List[float] = [0.0] * n_features
    for row in X_train:
        if len(row) != n_features:
            raise ValueError("Inconsistent feature dimension across X_train rows.")
        for j, val in enumerate(row):
            means[j] += float(val)
    means = [m / float(n_samples) for m in means]

    variances: List[float] = [0.0] * n_features
    for row in X_train:
        for j, val in enumerate(row):
            diff = float(val) - means[j]
            variances[j] += diff * diff
    scales: List[float] = []
    for var in variances:
        std = math.sqrt(var / float(n_samples))
        scales.append(std if std > 0.0 else 1.0)

    return StandardScalerStats(
        mean=tuple(means),
        scale=tuple(scales),
        n_samples_seen=n_samples,
    )


def transform_with_scaler_stats(
    X: Sequence[Sequence[float]],
    stats: StandardScalerStats,
) -> List[List[float]]:
    """Transform `X` using pre-fitted `StandardScalerStats`."""
    n_features = len(stats.mean)
    transformed: List[List[float]] = []
    for row in X:
        if len(row) != n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {n_features}, got {len(row)}."
            )
        transformed.append(
            [
                (float(val) - stats.mean[j]) / stats.scale[j]
                for j, val in enumerate(row)
            ]
        )
    return transformed


def fit_transform_target_fold(
    X_target_train: Sequence[Sequence[float]],
    X_target_test: Sequence[Sequence[float]],
) -> Tuple[List[List[float]], List[List[float]], StandardScalerStats]:
    """
    Fit standardization strictly on `X_target_train` and transform both
    `X_target_train` and `X_target_test` without test-condition leakage.
    """
    stats = fit_standard_scaler_stats(X_target_train)
    X_train_scaled = transform_with_scaler_stats(X_target_train, stats)
    X_test_scaled = transform_with_scaler_stats(X_target_test, stats)
    return X_train_scaled, X_test_scaled, stats


def scale_source_domains(
    source_features_by_name: Dict[str, Sequence[Sequence[float]]],
) -> Tuple[Dict[str, List[List[float]]], Dict[str, StandardScalerStats]]:
    """
    Standardize each source dataset independently (`fit_transform` per source domain).
    """
    scaled_sources: Dict[str, List[List[float]]] = {}
    source_stats: Dict[str, StandardScalerStats] = {}
    for src_name, X_src in source_features_by_name.items():
        stats = fit_standard_scaler_stats(X_src)
        scaled_sources[src_name] = transform_with_scaler_stats(X_src, stats)
        source_stats[src_name] = stats
    return scaled_sources, source_stats
