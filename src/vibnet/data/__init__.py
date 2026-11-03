"""
Canonical dataset registry and validation interfaces for VibNet-1D (`src.vibnet.data`).
"""

from src.vibnet.data.registry import (
    CANONICAL_DATASETS,
    DatasetSpec,
    build_reference_map,
    get_dataset_spec,
    get_sampling_rate_hz,
    list_all_datasets,
    list_sources,
    list_targets,
    load_yaml_config,
)
from src.vibnet.data.validation import (
    audit_class_condition_coverage,
    run_dataset_validation,
)

__all__ = [
    "CANONICAL_DATASETS",
    "DatasetSpec",
    "audit_class_condition_coverage",
    "build_reference_map",
    "get_dataset_spec",
    "get_sampling_rate_hz",
    "list_all_datasets",
    "list_sources",
    "list_targets",
    "load_yaml_config",
    "run_dataset_validation",
]
