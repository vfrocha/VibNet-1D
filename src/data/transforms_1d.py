"""Transformações para sinais 1D no domínio do tempo (janelas de 1 segundo).

Este módulo fornece as interfaces base para a fase de modelos 1D no domínio do
tempo (sem extração tabular), incluindo remoção de tendência linear (detrend),
reamostragem polifásica (resampling) e normalização (z-score por janela ou RMS
por referência livre de vazamento entre treino e teste).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.signal import detrend, resample_poly

NormStrategy1D = Literal["none", "window_zscore", "window_rms", "reference_rms"]


def detrend_1d(signal: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Remove a tendência linear do sinal 1D (ou lote de janelas 2D)."""
    arr = np.asarray(signal, dtype=np.float32)
    return detrend(arr, axis=axis, type="linear").astype(np.float32, copy=False)


def resample_1d(
    signal: np.ndarray,
    orig_fs: int,
    target_fs: int,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Reamostra sinal(is) de `orig_fs` para `target_fs` via filtro polifásico."""
    arr = np.asarray(signal, dtype=np.float32)
    if orig_fs <= 0 or target_fs <= 0:
        raise ValueError("As taxas de amostragem devem ser inteiros positivos.")
    if orig_fs == target_fs:
        return arr.copy()

    divisor = math.gcd(int(orig_fs), int(target_fs))
    up = int(target_fs) // divisor
    down = int(orig_fs) // divisor
    resampled = resample_poly(arr, up=up, down=down, axis=axis)
    return np.asarray(resampled, dtype=np.float32)


def zscore_normalize_1d(
    signal: np.ndarray,
    *,
    eps: float = 1e-8,
    axis: int = -1,
) -> np.ndarray:
    """Aplica normalização Z-score independente por janela temporal."""
    arr = np.asarray(signal, dtype=np.float32)
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    return (arr - mean) / np.maximum(std, eps)


def rms_normalize_1d(
    signal: np.ndarray,
    *,
    reference_rms: Optional[float] = None,
    eps: float = 1e-8,
    axis: int = -1,
) -> np.ndarray:
    """Normaliza pelo RMS da própria janela ou por um `reference_rms` externo.

    Em avaliações LOCO ou cross-dataset, `reference_rms` deve ser estimado
    exclusivamente no conjunto de treino (ex.: sinais normais de referência).
    """
    arr = np.asarray(signal, dtype=np.float32)
    if reference_rms is not None:
        if reference_rms <= 0 or not np.isfinite(reference_rms):
            raise ValueError(f"reference_rms inválido: {reference_rms}")
        return arr / np.float32(reference_rms)

    rms = np.sqrt(np.mean(np.square(arr), axis=axis, keepdims=True))
    return arr / np.maximum(rms, eps)


@dataclass
class TimeDomainTransform1D:
    """Pipeline configurável de pré-processamento para sinais 1D de 1 segundo."""

    apply_detrend: bool = True
    orig_fs: Optional[int] = None
    target_fs: Optional[int] = None
    normalization: NormStrategy1D = "window_zscore"
    reference_rms: Optional[float] = None
    eps: float = 1e-8

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        out = np.asarray(signal, dtype=np.float32)
        if self.apply_detrend:
            out = detrend_1d(out)

        if (
            self.orig_fs is not None
            and self.target_fs is not None
            and self.orig_fs != self.target_fs
        ):
            out = resample_1d(out, self.orig_fs, self.target_fs)

        if self.normalization == "none":
            return out
        if self.normalization == "window_zscore":
            return zscore_normalize_1d(out, eps=self.eps)
        if self.normalization == "window_rms":
            return rms_normalize_1d(out, reference_rms=None, eps=self.eps)
        if self.normalization == "reference_rms":
            if self.reference_rms is None:
                raise ValueError(
                    "reference_rms deve ser fornecido para normalization='reference_rms'."
                )
            return rms_normalize_1d(
                out, reference_rms=self.reference_rms, eps=self.eps
            )
        raise ValueError(
            f"Estratégia de normalização desconhecida: {self.normalization}"
        )


__all__ = [
    "NormStrategy1D",
    "TimeDomainTransform1D",
    "detrend_1d",
    "resample_1d",
    "rms_normalize_1d",
    "zscore_normalize_1d",
]

