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
from src.vibnet.data.transactional_io import (
    DatasetCommitReceipt,
    ManifestCommitReceipt,
    assert_no_symlink_components,
    assert_no_symlinks_in_tree,
    assert_safe_dataset_target_path,
    assert_same_filesystem,
    cleanup_staging_dir,
    commit_staged_dataset_dir,
    create_staging_dataset_dir,
    get_device_id,
    transactional_update_manifest,
    transactional_write_csv_manifest,
    transactional_write_json_manifest,
    validate_staged_dataset_dir,
)
from src.vibnet.data.validation import (
    audit_class_condition_coverage,
    run_dataset_validation,
)

__all__ = [
    "CANONICAL_DATASETS",
    "DatasetCommitReceipt",
    "DatasetSpec",
    "ManifestCommitReceipt",
    "assert_no_symlink_components",
    "assert_no_symlinks_in_tree",
    "assert_safe_dataset_target_path",
    "assert_same_filesystem",
    "audit_class_condition_coverage",
    "build_reference_map",
    "cleanup_staging_dir",
    "commit_staged_dataset_dir",
    "create_staging_dataset_dir",
    "get_dataset_spec",
    "get_device_id",
    "get_sampling_rate_hz",
    "list_all_datasets",
    "list_sources",
    "list_targets",
    "load_yaml_config",
    "run_dataset_validation",
    "transactional_update_manifest",
    "transactional_write_csv_manifest",
    "transactional_write_json_manifest",
    "validate_staged_dataset_dir",
]
