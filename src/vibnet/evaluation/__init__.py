"""
Evaluation protocols, target-only balancing, metrics, and provenance recording
for VibNet-1D (`src.vibnet.evaluation`).
"""

from src.vibnet.evaluation.balancing import (
    BalanceResult,
    balance_target_training,
    ensure_batchnorm_safe_batch_size,
    trim_single_remainder_sample,
)
from src.vibnet.evaluation.metrics import (
    FoldMetricsResult,
    compute_fold_metrics,
    compute_selection_score,
)
from src.vibnet.evaluation.protocols import (
    LOCOFoldSplit,
    assert_no_condition_leakage,
    build_source_combinations,
    iter_loco_folds,
)
from src.vibnet.evaluation.provenance import (
    collect_run_provenance,
    compute_file_sha256,
    write_run_metadata,
)

__all__ = [
    "BalanceResult",
    "FoldMetricsResult",
    "LOCOFoldSplit",
    "assert_no_condition_leakage",
    "balance_target_training",
    "build_source_combinations",
    "collect_run_provenance",
    "compute_file_sha256",
    "compute_fold_metrics",
    "compute_selection_score",
    "ensure_batchnorm_safe_batch_size",
    "iter_loco_folds",
    "trim_single_remainder_sample",
    "write_run_metadata",
]
