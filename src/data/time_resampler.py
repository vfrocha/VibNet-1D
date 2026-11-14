"""Módulo de reamostragem e normalização no domínio do tempo (VibNet1DTime).

Implementa estratégias de redução dimensional temporal (`downsample_window`) e
normalização por janela (`normalize_window`) operando sobre sinais 1D de 1 segundo,
de forma 100% isolada do pipeline tabular.
"""

from __future__ import annotations

import fractions
import math
import numpy as np
from scipy.signal import detrend, resample, resample_poly


def downsample_window(
    sig: np.ndarray,
    target_len: int,
    mode: str = "uniform",
) -> np.ndarray:
    """Reamostra/reduz uma janela 1D de 1 segundo para `target_len` pontos.

    Parâmetros
    ----------
    sig : np.ndarray
        Sinal 1D no domínio do tempo.
    target_len : int
        Comprimento alvo da janela reamostrada (ex.: `k * 1024`).
    mode : str, default='uniform'
        Modo de reamostragem:
        - `'uniform'`: seleção de índices uniformemente espaçados via `np.linspace`.
        - `'fft'`: reamostragem espectral no domínio da frequência via `scipy.signal.resample`.
        - `'poly'`: filtro polifásico via `scipy.signal.resample_poly` usando fração
          simplificada (`math.gcd` ou `fractions.Fraction.limit_denominator(1000)`),
          com ajuste fino de comprimento caso difira por poucos pontos.
        - `'linear'`: interpolação linear rápida via `np.interp`.

    Retorna
    -------
    np.ndarray
        Vetor 1D `float32` com comprimento exato `target_len`.
    """
    arr = np.asarray(sig, dtype=np.float64).ravel()
    orig_len = len(arr)
    target_len = int(target_len)

    if target_len <= 0:
        raise ValueError(f"target_len deve ser positivo, recebido: {target_len}")
    if orig_len == 0:
        raise ValueError("O sinal de entrada está vazio (len(sig) == 0).")
    if orig_len == target_len:
        return arr.astype(np.float32, copy=False)

    mode_clean = mode.strip().lower()

    if mode_clean == "uniform":
        indices = np.linspace(0, orig_len - 1, target_len, dtype=int)
        out = arr[indices]

    elif mode_clean == "fft":
        out = resample(arr, target_len)

    elif mode_clean == "poly":
        gcd_val = math.gcd(target_len, orig_len)
        up = target_len // gcd_val
        down = orig_len // gcd_val
        if down > 1000 or up > 1000:
            frac = fractions.Fraction(target_len, orig_len).limit_denominator(1000)
            up, down = frac.numerator, frac.denominator

        out = resample_poly(arr, up, down)
        if len(out) != target_len:
            out = np.interp(
                np.linspace(0.0, 1.0, target_len),
                np.linspace(0.0, 1.0, len(out)),
                out,
            )

    elif mode_clean == "linear":
        out = np.interp(
            np.linspace(0.0, 1.0, target_len),
            np.linspace(0.0, 1.0, orig_len),
            arr,
        )

    else:
        raise ValueError(
            f"Modo de reamostragem inválido: '{mode}'. "
            "Use 'uniform', 'fft', 'poly' ou 'linear'."
        )

    return np.asarray(out, dtype=np.float32)


def normalize_window(
    sig: np.ndarray,
    norm_mode: str = "window_zscore",
) -> np.ndarray:
    """Remove tendência linear (`detrend`) e normaliza uma janela 1D.

    Parâmetros
    ----------
    sig : np.ndarray
        Sinal 1D no domínio do tempo.
    norm_mode : str, default='window_zscore'
        Estratégia de normalização após `scipy.signal.detrend(sig)`:
        - `'raw'`: apenas `detrend(sig)`.
        - `'window_zscore'`: `(sig_detrended - mean) / (std + 1e-8)`.
        - `'window_rms'`: `sig_detrended / (rms + 1e-8)`.

    Retorna
    -------
    np.ndarray
        Vetor 1D `float32` normalizado.
    """
    arr = np.asarray(sig, dtype=np.float64).ravel()
    detrended = detrend(arr)

    mode_clean = norm_mode.strip().lower()
    eps = 1e-8

    if mode_clean == "raw":
        out = detrended
    elif mode_clean == "window_zscore":
        mean = np.mean(detrended)
        std = np.std(detrended)
        out = (detrended - mean) / (std + eps)
    elif mode_clean == "window_rms":
        rms = np.sqrt(np.mean(np.square(detrended)))
        out = detrended / (rms + eps)
    else:
        raise ValueError(
            f"Modo de normalização inválido: '{norm_mode}'. "
            "Use 'raw', 'window_zscore' ou 'window_rms'."
        )

    return np.asarray(out, dtype=np.float32)


__all__ = [
    "downsample_window",
    "normalize_window",
]

