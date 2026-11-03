"""
Time-domain signal contracts and LOCO alignment adapters (`src.vibnet.signal`).

Prepares the modular interface for the upcoming 1D time-domain stage (`VibNet-1D`)
without prematurely implementing specific 1D CNN/ResNet/Transformer architectures
(`ADR-004`).
"""

from src.vibnet.signal.contracts import (
    DEFAULT_WINDOW_CHANNELS,
    DEFAULT_WINDOW_LENGTH,
    TimeDomainDatasetSpec,
    TimeDomainLOCOAdapter,
    TimeDomainWindowBatch,
    verify_tabular_signal_loco_alignment,
)

__all__ = [
    "DEFAULT_WINDOW_CHANNELS",
    "DEFAULT_WINDOW_LENGTH",
    "TimeDomainDatasetSpec",
    "TimeDomainLOCOAdapter",
    "TimeDomainWindowBatch",
    "verify_tabular_signal_loco_alignment",
]
