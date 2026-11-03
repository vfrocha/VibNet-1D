"""
Leave-One-Condition-Out (LOCO) evaluation protocol and source-domain combinatorics
(`src.vibnet.evaluation.protocols`).

Enforces ADR-002 (`docs/decisions/002-loco-isolation-and-target-balancing.md`):
- Each LOCO fold holds out all windows of one operating condition `c_k` for testing
  and trains strictly on the remaining `K - 1` conditions.
- Verifies that `train_conditions` and `test_conditions` are strictly disjoint and
  that `train_indices` and `test_indices` have zero overlap.
- Generates deterministic source-domain combinations (`single`, `pairs`, `triplets`,
  `all`, yielding the 15 non-empty subsets for 4 source datasets).
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Dict, Iterator, List, Sequence, Tuple


@dataclass(frozen=True)
class LOCOFoldSplit:
    """Represents a single Leave-One-Condition-Out (LOCO) fold partition."""

    fold_index: int
    test_condition: str
    train_conditions: Tuple[str, ...]
    train_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]


def assert_no_condition_leakage(
    conditions: Sequence[str],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> None:
    """
    Verify that a LOCO split has zero index overlap and zero operating-condition
    overlap between training and test partitions.

    Raises `ValueError` if any leakage is detected.
    """
    train_idx_set = set(train_indices)
    test_idx_set = set(test_indices)
    overlap_idx = train_idx_set & test_idx_set
    if overlap_idx:
        raise ValueError(
            f"Sample index leakage detected between train and test: {sorted(overlap_idx)[:5]}"
        )

    train_conds = {str(conditions[i]) for i in train_indices}
    test_conds = {str(conditions[i]) for i in test_indices}
    overlap_conds = train_conds & test_conds
    if overlap_conds:
        raise ValueError(
            f"Operating condition leakage detected between train and test: {sorted(overlap_conds)}"
        )
    if len(test_conds) != 1:
        raise ValueError(
            f"LOCO test fold must contain exactly 1 operating condition, got {sorted(test_conds)}"
        )


def iter_loco_folds(conditions: Sequence[str]) -> Iterator[LOCOFoldSplit]:
    """
    Yield deterministic Leave-One-Condition-Out (`LOCOFoldSplit`) partitions sorted
    by unique condition name (matching `sklearn.model_selection.LeaveOneGroupOut`).
    """
    str_conds = [str(c) for c in conditions]
    unique_conds = sorted(set(str_conds))
    if len(unique_conds) < 2:
        raise ValueError(
            f"LOCO requires at least 2 distinct operating conditions, got {len(unique_conds)}."
        )

    for fold_idx, held_out_cond in enumerate(unique_conds):
        train_idx = tuple(
            i for i, cond in enumerate(str_conds) if cond != held_out_cond
        )
        test_idx = tuple(
            i for i, cond in enumerate(str_conds) if cond == held_out_cond
        )
        assert_no_condition_leakage(str_conds, train_idx, test_idx)
        train_conds = tuple(c for c in unique_conds if c != held_out_cond)
        yield LOCOFoldSplit(
            fold_index=fold_idx,
            test_condition=held_out_cond,
            train_conditions=train_conds,
            train_indices=train_idx,
            test_indices=test_idx,
        )


def build_source_combinations(
    sources: Sequence[str],
    modes: Sequence[str] = ("single", "pairs", "triplets", "all"),
    target_dataset: str | None = None,
) -> List[Tuple[str, ...]]:
    """
    Generate deterministic source dataset combinations excluding `target_dataset`
    if present.

    Supported modes:
    - `single`: subsets of size 1 (`C(4,1) = 4`)
    - `pairs`: subsets of size 2 (`C(4,2) = 6`)
    - `triplets`: subsets of size 3 (`C(4,3) = 4`)
    - `all`: full eligible source list (`C(4,4) = 1`)
    For 4 eligible sources and all 4 modes, returns all `2^4 - 1 = 15` subsets.
    """
    eligible = [s for s in sources if s != target_dataset]
    if not eligible:
        return []

    mode_to_sizes: Dict[str, List[int]] = {
        "single": [1],
        "pairs": [2],
        "triplets": [3],
        "all": [len(eligible)],
    }

    combos: List[Tuple[str, ...]] = []
    seen: set[Tuple[str, ...]] = set()
    for mode in modes:
        if mode not in mode_to_sizes:
            raise ValueError(
                f"Unknown source combination mode '{mode}'. "
                f"Expected one of {sorted(mode_to_sizes.keys())}."
            )
        for k in mode_to_sizes[mode]:
            if 1 <= k <= len(eligible):
                for subset in itertools.combinations(eligible, k):
                    if subset not in seen:
                        seen.add(subset)
                        combos.append(subset)
    return combos
