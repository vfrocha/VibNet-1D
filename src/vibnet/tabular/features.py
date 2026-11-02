"""
TabVibNet 141-dimensional feature extraction contract (`src.vibnet.tabular.features`).

Documents and wraps the 4-domain feature extraction pipeline (`src.features.extractors_v2`
and `src.features.signalai_wrapper`):
- Time domain: 18 statistical and impulse indicators
- Frequency domain (FFT + PSD): 24 spectral indicators
- Wavelet Packet Decomposition (WPD `db4`, level 3, 8 sub-bands): 73 features
- Empirical Mode Decomposition (EMD, up to 3 IMFs + residue): 26 features
Total base dimensionality: `141` features (`detrended_rms__base`).
Optional auxiliary absolute energy indicators (`rms_raw`, `band_energy_b1`,
`band_energy_b2`): `+3` features (`144` total, `detrended_rms__rms_b1_b2`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from src.vibnet.data.registry import get_sampling_rate_hz

TABVIBNET_FEATURE_DIM: int = 141
AUX_ENERGY_FEATURE_DIM: int = 3


@dataclass(frozen=True)
class TabVibNetFeatureSpec:
    """Canonical specification of the 141-feature TabVibNet vector."""

    time_domain_dim: int = 18
    frequency_domain_dim: int = 24
    wpd_domain_dim: int = 73
    emd_domain_dim: int = 26
    aux_energy_dim: int = 0
    wpd_wavelet: str = "db4"
    wpd_level: int = 3
    emd_max_imfs: int = 3

    @property
    def total_dim(self) -> int:
        return (
            self.time_domain_dim
            + self.frequency_domain_dim
            + self.wpd_domain_dim
            + self.emd_domain_dim
            + self.aux_energy_dim
        )


def get_tabvibnet_feature_spec(include_aux_energy: bool = False) -> TabVibNetFeatureSpec:
    """Return the canonical `TabVibNetFeatureSpec` (`141` or `144` features)."""
    return TabVibNetFeatureSpec(
        aux_energy_dim=AUX_ENERGY_FEATURE_DIM if include_aux_energy else 0
    )


def extract_tabvibnet_features(
    window: Sequence[float],
    dataset_name: str | None = None,
    fs_hz: int | None = None,
    include_aux_energy: bool = False,
    raw_window_for_aux: Sequence[float] | None = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Extract the 141-feature (or 144-feature) TabVibNet vector using the canonical
    sampling frequency `fs_hz` from `src.vibnet.data.registry` when `dataset_name`
    is provided.

    Raises `KeyError` if `dataset_name` is unknown and `fs_hz` is not explicitly given,
    preventing silent fallback to 12,000 Hz.
    """
    if fs_hz is None:
        if dataset_name is None:
            raise ValueError(
                "Either `dataset_name` or explicit `fs_hz` must be provided."
            )
        fs_hz = get_sampling_rate_hz(dataset_name)

    from src.features.signalai_wrapper import extract_features_from_signal

    feats = extract_features_from_signal(window, fs=fs_hz)
    meta: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "fs_hz": int(fs_hz),
        "expected_dim": TABVIBNET_FEATURE_DIM
        + (AUX_ENERGY_FEATURE_DIM if include_aux_energy else 0),
        "include_aux_energy": bool(include_aux_energy),
    }
    if include_aux_energy:
        if raw_window_for_aux is None:
            raise ValueError(
                "`raw_window_for_aux` is required when `include_aux_energy=True`."
            )
        from src.features.aux_energy_features import extract_aux_energy_features
        import numpy as np  # type: ignore

        aux = extract_aux_energy_features(np.asarray(raw_window_for_aux), fs=fs_hz)
        feats = np.concatenate([ np.asarray(feats), np.asarray(aux) ])

    return feats, meta
