"""
Dataset validation and condition-class coverage audit (`src.vibnet.data.validation`).

Provides pure-Python condition/class audit helpers for LOCO eligibility checks
as well as a bridge to `src.data.validate_dataset_tl.validate_datasets`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set


def audit_class_condition_coverage(
    condition_to_classes: Mapping[str, Iterable[str]],
) -> Dict[str, Any]:
    """
    Audit whether an operating-condition -> class-set mapping is eligible for
    Leave-One-Condition-Out (LOCO) evaluation without missing classes.

    Parameters
    ----------
    condition_to_classes : Mapping[str, Iterable[str]]
        Mapping from each operating condition identifier to the collection of
        fault classes present in that condition.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - `num_conditions`: total number of conditions
        - `global_classes`: sorted list of all classes across the dataset
        - `conditions_missing_classes`: dict mapping condition -> missing classes
        - `single_class_conditions`: list of conditions with <= 1 class
        - `loco_eligible`: True if there are >= 2 conditions and at least 2
          classes per condition (no complete condition-fault confounding)
        - `full_class_coverage_every_fold`: True if every condition contains
          100% of `global_classes`.
    """
    normalized: Dict[str, Set[str]] = {
        str(cond): {str(cls) for cls in classes}
        for cond, classes in condition_to_classes.items()
    }
    global_classes = sorted(
        {cls for class_set in normalized.values() for cls in class_set}
    )
    global_set = set(global_classes)

    conditions_missing: Dict[str, List[str]] = {}
    single_class_conditions: List[str] = []

    for cond in sorted(normalized.keys()):
        present = normalized[cond]
        if len(present) <= 1:
            single_class_conditions.append(cond)
        missing = sorted(global_set - present)
        if missing:
            conditions_missing[cond] = missing

    num_conditions = len(normalized)
    loco_eligible = (
        num_conditions >= 2
        and len(global_classes) >= 2
        and len(single_class_conditions) < num_conditions
    )
    full_coverage = len(conditions_missing) == 0 and num_conditions >= 2

    return {
        "num_conditions": num_conditions,
        "global_classes": global_classes,
        "conditions_missing_classes": conditions_missing,
        "single_class_conditions": single_class_conditions,
        "loco_eligible": loco_eligible,
        "full_class_coverage_every_fold": full_coverage,
    }


def run_dataset_validation(
    processed_tl_root: str | Path = "data/processed_tl",
    output_dir: str | Path = "results/dataset_validation",
    variants: Sequence[str] = ("raw", "detrended", "raw_rms", "detrended_rms"),
) -> Any:
    """
    Delegate full on-disk dataset validation to `src.data.validate_dataset_tl`.
    """
    from src.data.validate_dataset_tl import validate_datasets

    return validate_datasets(
        processed_tl_root=Path(processed_tl_root),
        output_dir=Path(output_dir),
        variants=list(variants),
    )
