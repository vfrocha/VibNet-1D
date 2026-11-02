"""
Data and evaluation contracts for the future 1D time-domain stage
(`src.vibnet.signal.contracts`).

Implements ADR-004 (`docs/decisions/004-planned-modular-architecture-vibnet.md`):
- Consumes the exact same 2,048-sample windows from `data/processed_tl/detrended_rms/`
  used to extract the 141 `TabVibNet` features.
- Enforces shape `(N, C, L) = (N, 1, 2048)` (`DEFAULT_WINDOW_CHANNELS = 1`,
  `DEFAULT_WINDOW_LENGTH = 2048`).
- Guarantees 1-to-1 sample, condition, and label alignment between `TabVibNet`
  and `VibNet-1D` across every LOCO fold so future comparisons are 100% paired
  and scientifically valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

from src.vibnet.data.registry import DatasetSpec, get_dataset_spec
from src.vibnet.evaluation.protocols import LOCOFoldSplit, iter_loco_folds

DEFAULT_WINDOW_CHANNELS: int = 1
DEFAULT_WINDOW_LENGTH: int = 2048


@dataclass(frozen=True)
class TimeDomainDatasetSpec:
    """Specification binding a canonical dataset to 1D time-domain window tensors."""

    dataset_spec: DatasetSpec
    variant: str = "detrended_rms"
    channels: int = DEFAULT_WINDOW_CHANNELS
    window_length: int = DEFAULT_WINDOW_LENGTH

    @classmethod
    def from_dataset_name(
        cls,
        name: str,
        variant: str = "detrended_rms",
        channels: int = DEFAULT_WINDOW_CHANNELS,
        window_length: int = DEFAULT_WINDOW_LENGTH,
    ) -> "TimeDomainDatasetSpec":
        spec = get_dataset_spec(name)
        if window_length != spec.window_length_samples:
            raise ValueError(
                f"Window length {window_length} does not match canonical "
                f"{spec.window_length_samples} for dataset '{name}'."
            )
        return cls(
            dataset_spec=spec,
            variant=variant,
            channels=channels,
            window_length=window_length,
        )


@dataclass(frozen=True)
class TimeDomainWindowBatch:
    """
    Contract for a batch of 1D vibration signal windows of logical shape `(N, C, L)`
    paired with integer fault labels and operating condition identifiers.
    """

    dataset_name: str
    variant: str
    num_samples: int
    channels: int
    window_length: int
    labels: Tuple[int, ...]
    conditions: Tuple[str, ...]

    def validate(self) -> None:
        """Validate batch metadata invariants (`N > 0`, `C == 1`, `L == 2048`)."""
        if self.num_samples <= 0:
            raise ValueError("TimeDomainWindowBatch must have num_samples > 0.")
        if self.channels != DEFAULT_WINDOW_CHANNELS:
            raise ValueError(
                f"Expected {DEFAULT_WINDOW_CHANNELS} channel(s), got {self.channels}."
            )
        if self.window_length != DEFAULT_WINDOW_LENGTH:
            raise ValueError(
                f"Expected window_length={DEFAULT_WINDOW_LENGTH}, got {self.window_length}."
            )
        if len(self.labels) != self.num_samples:
            raise ValueError(
                f"Labels length ({len(self.labels)}) != num_samples ({self.num_samples})."
            )
        if len(self.conditions) != self.num_samples:
            raise ValueError(
                f"Conditions length ({len(self.conditions)}) != num_samples ({self.num_samples})."
            )


class TimeDomainLOCOAdapter:
    """
    Produces Leave-One-Condition-Out (`LOCOFoldSplit`) partitions for a
    `TimeDomainWindowBatch` using the exact same `iter_loco_folds` engine as
    `TabVibNet`.
    """

    def __init__(self, batch: TimeDomainWindowBatch) -> None:
        batch.validate()
        self.batch = batch

    def iter_splits(self) -> Iterator[LOCOFoldSplit]:
        """Yield validated LOCO fold splits over `self.batch.conditions`."""
        return iter_loco_folds(self.batch.conditions)


def verify_tabular_signal_loco_alignment(
    tabular_conditions: Sequence[str],
    tabular_labels: Sequence[int],
    signal_batch: TimeDomainWindowBatch,
) -> Dict[str, int | bool]:
    """
    Verify that a tabular feature dataset and a `TimeDomainWindowBatch` produce
    identical LOCO fold partitions, condition names, train/test sample counts,
    and label sequences.

    Raises `ValueError` if any discrepancy is found between the tabular and
    time-domain representations.
    """
    signal_batch.validate()
    if len(tabular_conditions) != signal_batch.num_samples:
        raise ValueError(
            f"Sample count mismatch: tabular has {len(tabular_conditions)}, "
            f"signal_batch has {signal_batch.num_samples}."
        )
    if len(tabular_labels) != signal_batch.num_samples:
        raise ValueError(
            f"Label count mismatch: tabular has {len(tabular_labels)}, "
            f"signal_batch has {signal_batch.num_samples}."
        )

    tab_conds_tuple = tuple(str(c) for c in tabular_conditions)
    if tab_conds_tuple != signal_batch.conditions:
        raise ValueError(
            "Condition sequence mismatch between tabular dataset and TimeDomainWindowBatch."
        )

    tab_labels_tuple = tuple(int(y) for y in tabular_labels)
    if tab_labels_tuple != signal_batch.labels:
        raise ValueError(
            "Label sequence mismatch between tabular dataset and TimeDomainWindowBatch."
        )

    tab_splits = list(iter_loco_folds(tab_conds_tuple))
    sig_splits = list(TimeDomainLOCOAdapter(signal_batch).iter_splits())

    if len(tab_splits) != len(sig_splits):
        raise ValueError(
            f"LOCO fold count mismatch: {len(tab_splits)} vs {len(sig_splits)}."
        )

    for t_split, s_split in zip(tab_splits, sig_splits):
        if t_split != s_split:
            raise ValueError(
                f"LOCO split mismatch on fold {t_split.fold_index} "
                f"({t_split.test_condition} vs {s_split.test_condition})."
            )

    return {
        "aligned": True,
        "num_samples": signal_batch.num_samples,
        "num_folds": len(tab_splits),
        "window_length": signal_batch.window_length,
        "channels": signal_batch.channels,
    }
