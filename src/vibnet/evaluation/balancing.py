"""
Target-only training balance strategies and `BatchNorm1d` remainder-1 protection
(`src.vibnet.evaluation.balancing`).

Implements ADR-002 (`docs/decisions/002-loco-isolation-and-target-balancing.md`)
and resolves the root cause of the `TabNet` failure on `PU` fold
`C4_1500rpm_0.7Nm_400N` (`8,577 % 64 == 1` -> `ValueError: Expected more than 1
value per channel when training, got input size torch.Size([1, ...])` inside
PyTorch `BatchNorm1d`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class BalanceResult:
    """Container for balanced training features, labels, and class weights."""

    X_train: List[List[float]]
    y_train: List[int]
    class_weights: Dict[int, float]
    strategy: str
    dropped_remainder_sample: bool = False


def compute_balanced_class_weights(y_train: Sequence[int]) -> Dict[int, float]:
    """
    Compute inverse-frequency class weights (`N / (n_classes * count_c)`),
    matching `sklearn.utils.class_weight.compute_class_weight('balanced')`.
    """
    counts = Counter(int(y) for y in y_train)
    if not counts:
        return {}
    n_samples = len(y_train)
    n_classes = len(counts)
    return {
        cls: float(n_samples) / float(n_classes * cnt)
        for cls, cnt in sorted(counts.items())
    }


def ensure_batchnorm_safe_batch_size(
    n_samples: int,
    batch_size: int = 64,
) -> Tuple[int, bool]:
    """
    Check whether `n_samples` leaves a single-sample final mini-batch
    (`n_samples % batch_size == 1`), which causes PyTorch `BatchNorm1d` (and
    `TabNetClassifier` when `drop_last=False`) to crash during training.

    Returns `(safe_n_samples, needs_single_sample_trim)`.
    """
    if batch_size <= 1:
        raise ValueError(f"batch_size must be > 1 for BatchNorm1d, got {batch_size}")
    if n_samples > batch_size and (n_samples % batch_size == 1):
        return n_samples - 1, True
    return n_samples, False


def trim_single_remainder_sample(
    X_train: Sequence[Sequence[float]],
    y_train: Sequence[int],
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[List[List[float]], List[int], bool]:
    """
    If `len(X_train) % batch_size == 1` (for example, `8577 % 64 == 1` in `PU`
    fold `C4_1500rpm_0.7Nm_400N`), deterministically drop 1 sample from the
    majority class so that `len(X_train) % batch_size == 0` (`8576 % 64 == 0`),
    preventing `BatchNorm1d` crashes without altering minority class coverage.
    """
    X_list = [list(map(float, row)) for row in X_train]
    y_list = [int(y) for y in y_train]
    _, needs_trim = ensure_batchnorm_safe_batch_size(len(X_list), batch_size)
    if not needs_trim:
        return X_list, y_list, False

    counts = Counter(y_list)
    # Pick the majority class (tie-breaking deterministically by largest class ID)
    majority_cls = max(counts.keys(), key=lambda c: (counts[c], c))
    candidate_indices = [i for i, y in enumerate(y_list) if y == majority_cls]

    rng = random.Random(int(seed))
    drop_idx = rng.choice(candidate_indices)

    X_trimmed = [row for i, row in enumerate(X_list) if i != drop_idx]
    y_trimmed = [y for i, y in enumerate(y_list) if i != drop_idx]
    return X_trimmed, y_trimmed, True


def _euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def _smote_oversample_pure(
    X_train: List[List[float]],
    y_train: List[int],
    sampling_ratio: float = 1.0,
    k_neighbors: int = 5,
    seed: int = 42,
) -> Tuple[List[List[float]], List[int]]:
    """
    Pure-Python k-NN SMOTE interpolation fallback used when `imbalanced-learn`
    is not installed in the active Python environment.
    """
    counts = Counter(y_train)
    if len(counts) <= 1:
        return list(X_train), list(y_train)

    max_count = max(counts.values())
    target_minority_count = max(1, int(round(max_count * float(sampling_ratio))))
    rng = random.Random(int(seed))

    X_out = [list(row) for row in X_train]
    y_out = list(y_train)

    for cls in sorted(counts.keys()):
        cls_indices = [i for i, y in enumerate(y_train) if y == cls]
        n_cls = len(cls_indices)
        if n_cls >= target_minority_count or n_cls == 0:
            continue
        n_to_generate = target_minority_count - n_cls
        cls_samples = [X_train[i] for i in cls_indices]

        if n_cls == 1:
            # Duplicate with tiny jitter if only 1 sample exists
            for _ in range(n_to_generate):
                X_out.append(list(cls_samples[0]))
                y_out.append(cls)
            continue

        k_eff = min(k_neighbors, n_cls - 1)
        # Precompute nearest neighbors within class `cls`
        neighbors_map: List[List[int]] = []
        for idx_a, vec_a in enumerate(cls_samples):
            dists = [
                (_euclidean_distance(vec_a, vec_b), idx_b)
                for idx_b, vec_b in enumerate(cls_samples)
                if idx_b != idx_a
            ]
            dists.sort(key=lambda item: item[0])
            neighbors_map.append([idx_b for _, idx_b in dists[:k_eff]])

        for _ in range(n_to_generate):
            base_local_idx = rng.randrange(n_cls)
            nn_local_idx = rng.choice(neighbors_map[base_local_idx])
            base_vec = cls_samples[base_local_idx]
            nn_vec = cls_samples[nn_local_idx]
            gap = rng.random()
            synth = [
                b_val + gap * (n_val - b_val)
                for b_val, n_val in zip(base_vec, nn_vec)
            ]
            X_out.append(synth)
            y_out.append(cls)

    return X_out, y_out


def balance_target_training(
    X_train: Sequence[Sequence[float]],
    y_train: Sequence[int],
    strategy: str = "none",
    seed: int = 42,
    batch_size_for_batchnorm_safety: int | None = 64,
) -> BalanceResult:
    """
    Apply target-only training class balancing (`none`, `smote`, `hybrid`,
    `class_weights`, `partial_smote`) and optional `BatchNorm1d` remainder-1
    protection (`batch_size_for_batchnorm_safety`).

    Note on `hybrid` vs `smote` (`ADR-002`):
    In `TabVibNet`, `smote` resamples all minority classes to match the majority
    class (`1:1:...:1`), after which inverse-frequency weights equal `1.0` for
    every class. By contrast, `partial_smote` oversamples minority classes to
    `70%` of the majority class and computes non-uniform `class_weights` on the
    resulting distribution, providing a genuinely distinct hybrid strategy.
    """
    valid_strategies = {"none", "smote", "hybrid", "class_weights", "partial_smote"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown balance strategy '{strategy}'. Expected one of {sorted(valid_strategies)}."
        )

    X_work = [list(map(float, row)) for row in X_train]
    y_work = [int(y) for y in y_train]

    if strategy in ("smote", "hybrid"):
        X_work, y_work = _smote_oversample_pure(
            X_work, y_work, sampling_ratio=1.0, k_neighbors=5, seed=seed
        )
    elif strategy == "partial_smote":
        X_work, y_work = _smote_oversample_pure(
            X_work, y_work, sampling_ratio=0.7, k_neighbors=5, seed=seed
        )

    dropped = False
    if batch_size_for_batchnorm_safety is not None:
        X_work, y_work, dropped = trim_single_remainder_sample(
            X_work,
            y_work,
            batch_size=batch_size_for_batchnorm_safety,
            seed=seed,
        )

    if strategy in ("class_weights", "hybrid", "partial_smote"):
        weights = compute_balanced_class_weights(y_work)
    else:
        counts = Counter(y_work)
        weights = {cls: 1.0 for cls in sorted(counts.keys())}

    return BalanceResult(
        X_train=X_work,
        y_train=y_work,
        class_weights=weights,
        strategy=strategy,
        dropped_remainder_sample=dropped,
    )
