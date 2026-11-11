#!/usr/bin/env python3
"""
Complementary Classical Machine Learning Baselines (`RandomForest`, `SVM`, `XGBoost`)
for the Scratch vs. Transfer Learning Benchmark
(`src/experiments/run_scratch_vs_tl_classical_baselines.py`).

Experiment ID : `scratch_vs_tl_classical_baselines`
Task          : `diagnosis`

Key Methodological & Reproducibility Guarantees:
  1. Strict Manifest Reuse (`results/scratch_vs_tl/run_20261105_141743/fold_manifest.csv`):
     - Never generates new random train/test splits.
     - Recovers the exact ordered `train_sample_ids` and `test_sample_ids` from the
       396 manifest entries (66 LOCO folds x 3 seeds x 2 balance strategies).
     - Verifies that `split_signature` and `preprocessing_signature` match the
       original benchmark identically before training any model.
     - If a manifest entry cannot be reproduced or its sample IDs cannot be resolved,
       halts that evaluation and records an explicit error (`ERROR_MANIFEST_MISMATCH`).
  2. Target-Only Classical Baselines (`method="ClassicalBaseline"`):
     - `RandomForest` (`sklearn.ensemble.RandomForestClassifier`)
     - `SVM` (`sklearn.svm.SVC`, `kernel="rbf"`, `probability=True`)
     - `XGBoost` (`xgboost.XGBClassifier`, `objective="multi:softprob"`)
     - Trained strictly on the target dataset's LOCO training fold (`StandardScaler`
       fitted exclusively on original target training data before `SMOTE`, then applied
       to `X_test`). Zero source-dataset samples or pretrained weights are used.
  3. Complete Isolation from Original Benchmark Artifacts:
     - Writes all outputs to `results/scratch_vs_tl/classical_baselines/run_<timestamp>/`
       and never modifies `results/scratch_vs_tl/run_20261105_141743/`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.experiments.audit_scratch_vs_tl import collect_git_and_script_hashes
from src.experiments.run_baseline_loco_v2 import (
    BALANCE_STRATEGIES,
    EXPECTED_LOCO_FOLDS,
    FEATURE_DIMENSION,
    PREPROCESSING_VARIANT,
    SEEDS,
    TARGET_DATASETS,
    TASK_NAME,
    _prepare_contiguous_labels,
    apply_smote_with_audit,
    compute_fold_metrics,
    compute_sample_split_signature,
    load_and_extract_target_dataset,
    project_predictions_to_full_vocabulary,
    scale_loco_fold_strict,
    set_global_seed,
    validate_data_root_is_detrended_rms,
    verify_canonical_fold_metadata,
)
from src.experiments.run_scratch_vs_tl_benchmark import (
    _df_to_table_str,
    _holm_bonferroni_adjust,
    compute_preprocessing_signature,
    validate_predictions_integrity,
)
from src.vibnet.data.registry import load_yaml_config
from src.vibnet.evaluation.provenance import get_dependency_versions

EXPERIMENT_ID: str = "scratch_vs_tl_classical_baselines"
CLASSICAL_ARCHITECTURES: List[str] = ["RandomForest", "SVM", "XGBoost"]
CLASSICAL_METHOD: str = "ClassicalBaseline"
EXPECTED_MANIFEST_ENTRIES: int = 396  # 66 folds * 3 seeds * 2 balance strategies
EXPECTED_FULL_EVALUATIONS: int = 1188  # 396 * 3 classical models

DEFAULT_CONFIG_PATH: Path = (
    PROJECT_ROOT / "configs" / "scratch_vs_tl_classical_baselines.yaml"
)
DEFAULT_REFERENCE_BENCHMARK_DIR: Path = (
    RESULTS_DIR / "scratch_vs_tl" / "run_20261105_141743"
)
DEFAULT_MANIFEST_PATH: Path = DEFAULT_REFERENCE_BENCHMARK_DIR / "fold_manifest.csv"
DEFAULT_DATA_ROOT: Path = PROJECT_ROOT / "data" / "processed_tl" / PREPROCESSING_VARIANT
DEFAULT_OUTPUT_ROOT: Path = RESULTS_DIR / "scratch_vs_tl" / "classical_baselines"


# =============================================================================
# 1. CONFIGURATION & PRE-EXECUTION VALIDATION
# =============================================================================


def load_classical_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> Dict[str, Any]:
    """Load `configs/scratch_vs_tl_classical_baselines.yaml` and validate keys."""
    cfg = load_yaml_config(config_path)
    archs = cfg.get("architectures", CLASSICAL_ARCHITECTURES)
    for a in archs:
        if a not in CLASSICAL_ARCHITECTURES:
            raise ValueError(
                f"Unsupported classical architecture '{a}' in {config_path}. "
                f"Expected subset of {CLASSICAL_ARCHITECTURES}."
            )
    return cfg


def verify_classical_dependencies(
    architectures: Sequence[str] = CLASSICAL_ARCHITECTURES,
) -> Dict[str, str]:
    """
    Verify that scikit-learn, imbalanced-learn, and xgboost (when requested) are
    installed and importable. Raises a clear RuntimeError with installation
    instructions if XGBoost is missing — never silently substitutes another model.
    """
    versions: Dict[str, str] = {}
    try:
        import sklearn
        from sklearn.ensemble import RandomForestClassifier as _RF  # noqa: F401
        from sklearn.svm import SVC as _SVC  # noqa: F401

        versions["scikit-learn"] = str(sklearn.__version__)
    except ImportError as exc:
        raise RuntimeError(
            "Missing required dependency 'scikit-learn'. "
            "Install it with: python -m pip install scikit-learn>=1.3.0"
        ) from exc

    try:
        import imblearn
        from imblearn.over_sampling import SMOTE as _SMOTE  # noqa: F401

        versions["imbalanced-learn"] = str(imblearn.__version__)
    except ImportError as exc:
        raise RuntimeError(
            "Missing required dependency 'imbalanced-learn'. "
            "Install it with: python -m pip install imbalanced-learn>=0.11.0"
        ) from exc

    if "XGBoost" in architectures:
        try:
            import xgboost
            from xgboost import XGBClassifier as _XGB  # noqa: F401

            versions["xgboost"] = str(xgboost.__version__)
        except ImportError as exc:
            raise RuntimeError(
                "Missing required dependency 'xgboost' for the XGBoost baseline. "
                "Do NOT substitute XGBoost with another algorithm. "
                "Install it in your active environment with:\n"
                "  python -m pip install xgboost>=2.0.0\n"
                "or:\n"
                "  python -m pip install -r requirements.txt"
            ) from exc

    return versions


def load_and_validate_reference_manifest(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    enforce_full_396: bool = True,
) -> pd.DataFrame:
    """
    Load the original benchmark's `fold_manifest.csv` and validate:
      1. The file exists on disk.
      2. All required columns are present.
      3. There are zero duplicate `(target_dataset, fold_id, seed, balance_strategy)` keys.
      4. When `enforce_full_396=True`, exactly 396 manifest entries are present covering
         all 5 target datasets (66 folds x 3 seeds x 2 balance strategies).
    """
    m_path = Path(manifest_path)
    if not m_path.exists() or not m_path.is_file():
        raise FileNotFoundError(
            f"Original fold manifest not found at '{m_path}'. "
            "Cannot run classical baselines without the original fold manifest."
        )

    df_manifest = pd.read_csv(m_path)
    required_cols = [
        "target_dataset",
        "fold_id",
        "condition",
        "train_sample_ids",
        "test_sample_ids",
        "seed",
        "balance_strategy",
        "split_signature",
        "preprocessing_signature",
        "n_train_before_balance",
        "n_train_after_balance",
        "n_test_samples",
        "class_counts_before",
        "class_counts_after",
    ]
    missing_cols = [c for c in required_cols if c not in df_manifest.columns]
    if missing_cols:
        raise ValueError(
            f"Manifest '{m_path}' is missing required columns: {missing_cols}"
        )

    key_cols = ["target_dataset", "fold_id", "seed", "balance_strategy"]
    dups = int(df_manifest.duplicated(subset=key_cols).sum())
    if dups > 0:
        raise ValueError(
            f"Manifest '{m_path}' contains {dups} duplicate keys on {key_cols}."
        )

    if enforce_full_396:
        if len(df_manifest) != EXPECTED_MANIFEST_ENTRIES:
            raise ValueError(
                f"Manifest '{m_path}' has {len(df_manifest)} entries, but exactly "
                f"{EXPECTED_MANIFEST_ENTRIES} entries (66 folds x 3 seeds x 2 balance "
                "strategies) are required."
            )
        for ds_name, expected_k in EXPECTED_LOCO_FOLDS.items():
            ds_sub = df_manifest[df_manifest["target_dataset"] == ds_name]
            expected_ds_rows = expected_k * len(SEEDS) * len(BALANCE_STRATEGIES)
            if len(ds_sub) != expected_ds_rows:
                raise ValueError(
                    f"Manifest '{m_path}' has {len(ds_sub)} rows for target '{ds_name}', "
                    f"expected {expected_ds_rows} ({expected_k} folds x 3 seeds x 2 balance strategies)."
                )

    return df_manifest


# =============================================================================
# 2. MANIFEST-DRIVEN SPLIT RECOVERY & ANTI-LEAKAGE PREPROCESSING
# =============================================================================


def recover_and_verify_fold_from_manifest(
    manifest_row: Mapping[str, Any],
    X_all: np.ndarray,
    y_all_enc: np.ndarray,
    conds_all: np.ndarray,
    sample_id_to_idx: Mapping[str, int],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """
    Recover the exact training and test feature matrices and label vectors for a
    single `(target_dataset, fold_id, condition, seed, balance_strategy)` manifest
    entry, and verify:
      1. Every training and test `sample_id` in the manifest exists in `sample_id_to_idx`.
      2. Training and test `sample_id` sets are strictly disjoint.
      3. All test samples belong to `condition` and zero training samples belong to `condition`.
      4. Recomputed `split_signature` matches `manifest_row["split_signature"]`.
      5. `StandardScaler` fitted strictly on `X_train_raw` + `apply_smote_with_audit`
         yields the exact `preprocessing_signature`, sample counts, and class counts
         recorded in `manifest_row`.
    """
    ds_name = str(manifest_row["target_dataset"])
    fold_id = int(manifest_row["fold_id"])
    expected_cond = str(manifest_row["condition"])
    seed = int(manifest_row["seed"])
    balance_strategy = str(manifest_row["balance_strategy"])

    try:
        train_sample_ids: List[str] = [
            str(s) for s in json.loads(str(manifest_row["train_sample_ids"]))
        ]
        test_sample_ids: List[str] = [
            str(s) for s in json.loads(str(manifest_row["test_sample_ids"]))
        ]
    except Exception as exc:
        raise ValueError(
            f"Failed to parse train/test sample IDs JSON for {ds_name} fold {fold_id}: {exc}"
        ) from exc

    if not train_sample_ids or not test_sample_ids:
        raise ValueError(
            f"Empty train ({len(train_sample_ids)}) or test ({len(test_sample_ids)}) "
            f"sample IDs in manifest for {ds_name} fold {fold_id}."
        )

    overlap = set(train_sample_ids).intersection(set(test_sample_ids))
    if overlap:
        raise ValueError(
            f"Data leakage in manifest for {ds_name} fold {fold_id}: "
            f"{len(overlap)} sample IDs appear in both train and test!"
        )

    missing_train = [s for s in train_sample_ids if s not in sample_id_to_idx]
    missing_test = [s for s in test_sample_ids if s not in sample_id_to_idx]
    if missing_train or missing_test:
        raise ValueError(
            f"Cannot resolve sample IDs from disk for {ds_name} fold {fold_id}: "
            f"{len(missing_train)} missing train IDs, {len(missing_test)} missing test IDs."
        )

    train_indices = np.asarray(
        [sample_id_to_idx[s] for s in train_sample_ids], dtype=np.int64
    )
    test_indices = np.asarray(
        [sample_id_to_idx[s] for s in test_sample_ids], dtype=np.int64
    )

    train_conds = set(conds_all[train_indices].tolist())
    test_conds = set(conds_all[test_indices].tolist())
    if test_conds != {expected_cond}:
        raise ValueError(
            f"Condition mismatch on test samples for {ds_name} fold {fold_id}: "
            f"found {sorted(test_conds)}, expected [{expected_cond}]."
        )
    if expected_cond in train_conds:
        raise ValueError(
            f"LOCO condition leakage on {ds_name} fold {fold_id}: held-out condition "
            f"'{expected_cond}' appears in training samples!"
        )

    recomputed_split_sig = compute_sample_split_signature(
        train_sample_ids=train_sample_ids,
        test_sample_ids=test_sample_ids,
    )
    expected_split_sig = str(manifest_row["split_signature"])
    if recomputed_split_sig != expected_split_sig:
        raise ValueError(
            f"split_signature mismatch for {ds_name} fold {fold_id}: "
            f"recomputed '{recomputed_split_sig}' != manifest '{expected_split_sig}'."
        )

    X_train_raw = X_all[train_indices]
    y_train_raw = y_all_enc[train_indices]
    X_test_raw = X_all[test_indices]
    y_test = y_all_enc[test_indices]

    if len(X_train_raw) != int(manifest_row["n_train_before_balance"]):
        raise ValueError(
            f"n_train_before_balance mismatch for {ds_name} fold {fold_id}: "
            f"recovered {len(X_train_raw)} != manifest {int(manifest_row['n_train_before_balance'])}."
        )
    if len(X_test_raw) != int(manifest_row["n_test_samples"]):
        raise ValueError(
            f"n_test_samples mismatch for {ds_name} fold {fold_id}: "
            f"recovered {len(X_test_raw)} != manifest {int(manifest_row['n_test_samples'])}."
        )

    # Fit StandardScaler strictly on original training data (before SMOTE)
    X_train_scaled, X_test_scaled, scaler = scale_loco_fold_strict(
        X_train_raw, X_test_raw
    )

    # Apply balance strategy ("none" or "smote") strictly to scaled training data
    X_train_bal, y_train_bal, smote_audit = apply_smote_with_audit(
        X_train_scaled=X_train_scaled,
        y_train_encoded=y_train_raw,
        class_names=class_names,
        strategy=balance_strategy,
        seed=seed,
    )

    recomputed_preproc_sig = compute_preprocessing_signature(
        preprocessing_variant=PREPROCESSING_VARIANT,
        feature_dim=int(X_all.shape[1]),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        balance_strategy=balance_strategy,
        smote_params=smote_audit["smote_params"],
    )
    expected_preproc_sig = str(manifest_row["preprocessing_signature"])
    if recomputed_preproc_sig != expected_preproc_sig:
        raise ValueError(
            f"preprocessing_signature mismatch for {ds_name} fold {fold_id} "
            f"(seed={seed}, bal={balance_strategy}): "
            f"recomputed '{recomputed_preproc_sig}' != manifest '{expected_preproc_sig}'."
        )

    if int(smote_audit["n_train_after"]) != int(manifest_row["n_train_after_balance"]):
        raise ValueError(
            f"n_train_after_balance mismatch for {ds_name} fold {fold_id} "
            f"(seed={seed}, bal={balance_strategy}): "
            f"recomputed {smote_audit['n_train_after']} != manifest {manifest_row['n_train_after_balance']}."
        )

    expected_counts_before = json.loads(str(manifest_row["class_counts_before"]))
    expected_counts_after = json.loads(str(manifest_row["class_counts_after"]))
    if smote_audit["counts_before"] != expected_counts_before:
        raise ValueError(
            f"class_counts_before mismatch for {ds_name} fold {fold_id}: "
            f"{smote_audit['counts_before']} != {expected_counts_before}."
        )
    if smote_audit["counts_after"] != expected_counts_after:
        raise ValueError(
            f"class_counts_after mismatch for {ds_name} fold {fold_id}: "
            f"{smote_audit['counts_after']} != {expected_counts_after}."
        )

    return {
        "train_sample_ids": train_sample_ids,
        "test_sample_ids": test_sample_ids,
        "X_train_raw": X_train_raw,
        "y_train_raw": y_train_raw,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_bal": X_train_bal,
        "y_train_bal": y_train_bal,
        "y_test": y_test,
        "smote_audit": smote_audit,
        "split_signature": recomputed_split_sig,
        "preprocessing_signature": recomputed_preproc_sig,
    }


# =============================================================================
# 3. CLASSICAL MACHINE LEARNING MODELS (RF, SVM, XGBOOST)
# =============================================================================


def train_eval_classical_model(
    architecture: str,
    X_train_bal: np.ndarray,
    y_train_bal: np.ndarray,
    X_test_scaled: np.ndarray,
    n_total_classes: int,
    seed: int,
    model_hyperparams: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Train and evaluate a classical ML baseline (`RandomForest`, `SVM`, or `XGBoost`)
    exclusively on the target dataset's LOCO training fold `(X_train_bal, y_train_bal)`
    and evaluate on `X_test_scaled`.

    Maps predicted class indices and probability columns from `present_train_classes`
    onto the full canonical dataset vocabulary (`0 .. n_total_classes - 1`) via
    `project_predictions_to_full_vocabulary`.
    """
    if architecture not in CLASSICAL_ARCHITECTURES:
        raise ValueError(
            f"Unsupported classical architecture '{architecture}'. "
            f"Expected one of {CLASSICAL_ARCHITECTURES}."
        )

    set_global_seed(seed)
    hp = dict(model_hyperparams or {})
    y_local, present_classes = _prepare_contiguous_labels(y_train_bal)
    n_present = int(len(present_classes))
    if n_present < 2:
        raise ValueError(
            f"Cannot train {architecture} with fewer than 2 classes in training fold "
            f"(n_present={n_present})."
        )

    if architecture == "RandomForest":
        rf_kwargs: Dict[str, Any] = {
            "n_estimators": int(hp.get("n_estimators", 500)),
            "max_depth": hp.get("max_depth", None),
            "min_samples_split": int(hp.get("min_samples_split", 2)),
            "min_samples_leaf": int(hp.get("min_samples_leaf", 1)),
            "max_features": hp.get("max_features", "sqrt"),
            "class_weight": hp.get("class_weight", None),
            "random_state": int(seed),
            "n_jobs": int(hp.get("n_jobs", -1)),
        }
        clf = RandomForestClassifier(**rf_kwargs)
        clf.fit(X_train_bal, y_local)
        preds_local = np.asarray(clf.predict(X_test_scaled), dtype=np.int64)
        probs_local = np.asarray(clf.predict_proba(X_test_scaled), dtype=np.float64)

    elif architecture == "SVM":
        svm_kwargs: Dict[str, Any] = {
            "kernel": str(hp.get("kernel", "rbf")),
            "C": float(hp.get("C", 1.0)),
            "gamma": hp.get("gamma", "scale"),
            "probability": bool(hp.get("probability", True)),
            "class_weight": hp.get("class_weight", None),
            "random_state": int(seed),
        }
        if not svm_kwargs["probability"]:
            raise ValueError("SVM baseline requires probability=True for ROC-AUC evaluation.")
        clf = SVC(**svm_kwargs)
        clf.fit(X_train_bal, y_local)
        preds_local = np.asarray(clf.predict(X_test_scaled), dtype=np.int64)
        probs_local = np.asarray(clf.predict_proba(X_test_scaled), dtype=np.float64)

    elif architecture == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost ('xgboost') is not installed. Install with: "
                "python -m pip install xgboost>=2.0.0"
            ) from exc

        xgb_kwargs: Dict[str, Any] = {
            "objective": str(hp.get("objective", "multi:softprob")),
            "n_estimators": int(hp.get("n_estimators", 500)),
            "max_depth": int(hp.get("max_depth", 6)),
            "learning_rate": float(hp.get("learning_rate", 0.05)),
            "subsample": float(hp.get("subsample", 0.8)),
            "colsample_bytree": float(hp.get("colsample_bytree", 0.8)),
            "eval_metric": str(hp.get("eval_metric", "mlogloss")),
            "random_state": int(seed),
            "n_jobs": int(hp.get("n_jobs", -1)),
            "verbosity": int(hp.get("verbosity", 0)),
        }
        if xgb_kwargs["objective"].startswith("multi:"):
            xgb_kwargs["num_class"] = int(n_present)

        clf = XGBClassifier(**xgb_kwargs)
        clf.fit(X_train_bal, y_local)
        probs_local = np.asarray(clf.predict_proba(X_test_scaled), dtype=np.float64)
        if probs_local.ndim == 1:
            probs_local = np.column_stack([1.0 - probs_local, probs_local])
        preds_local = np.argmax(probs_local, axis=1).astype(np.int64)

    preds, probs = project_predictions_to_full_vocabulary(
        preds_local=preds_local,
        probs_local=probs_local,
        present_train_classes=present_classes,
        n_total_classes=n_total_classes,
    )

    return (
        preds,
        probs,
        {
            "validation_split_mode": "none_fixed_hyperparams",
            "selected_epochs": 0,
            "backbone_frozen": False,
            "backbone_param_max_abs_diff": float("nan"),
            "backbone_buffer_max_abs_diff": float("nan"),
        },
    )


# =============================================================================
# 4. POST-EXECUTION INTEGRITY CHECKS & STATISTICAL SUMMARIES
# =============================================================================


def validate_post_execution_integrity(
    df_results: pd.DataFrame,
    df_preds: pd.DataFrame,
    df_manifest_selected: pd.DataFrame,
    expected_architectures: Sequence[str],
    expected_method: str = CLASSICAL_METHOD,
    is_full_run: bool = True,
) -> Dict[str, Any]:
    """
    Execute all mandatory post-execution reproducibility and integrity checks
    (Section 8):
      1. Exact number of evaluations (1,188 for full run; or `len(df_manifest_selected) * len(expected_architectures)`).
      2. Zero duplicate evaluation keys `(target_dataset, fold_id, seed, balance_strategy, architecture, method)`.
      3. Valid status on every evaluation; non-empty `error_message` on any failed evaluation.
      4. Exact match of `split_signature` and `preprocessing_signature` against `fold_manifest.csv`.
      5. Prediction records have zero duplicate keys when `sample_id` is included.
      6. Every `OK` evaluation has a matching set of prediction records in `predictions.csv`,
         and every prediction record has a corresponding `OK` evaluation record in `results.csv`.
      7. Test `sample_id` sets in `predictions.csv` match `test_sample_ids` in `fold_manifest.csv`
         and have zero overlap with `train_sample_ids`.
      8. Every probability vector matches `class_order`, is finite, and sums to ~1.0.
    """
    expected_eval_count = (
        EXPECTED_FULL_EVALUATIONS
        if is_full_run
        else int(len(df_manifest_selected) * len(expected_architectures))
    )
    if len(df_results) != expected_eval_count:
        raise ValueError(
            f"Evaluation count mismatch: found {len(df_results)} rows in results.csv, "
            f"expected {expected_eval_count}."
        )

    eval_key_cols = [
        "target_dataset",
        "fold_id",
        "seed",
        "balance_strategy",
        "architecture",
        "method",
    ]
    dup_evals = int(df_results.duplicated(subset=eval_key_cols).sum())
    if dup_evals > 0:
        raise ValueError(
            f"Duplicate evaluation keys detected in results.csv: {dup_evals} duplicates."
        )

    # Check status and error_message invariants
    for idx, r in df_results.iterrows():
        status_val = str(r["status"])
        err_val = "" if pd.isna(r["error_message"]) else str(r["error_message"]).strip()
        if not status_val or status_val == "nan":
            raise ValueError(f"Missing status at results.csv row {idx}.")
        if status_val != "OK" and not err_val:
            raise ValueError(
                f"Failed evaluation at results.csv row {idx} (status='{status_val}') "
                "is missing an informative error_message."
            )

    # Verify split_signature and preprocessing_signature match manifest
    manifest_lookup: Dict[Tuple[str, int, int, str], Mapping[str, Any]] = {}
    for _, m_row in df_manifest_selected.iterrows():
        m_key = (
            str(m_row["target_dataset"]),
            int(m_row["fold_id"]),
            int(m_row["seed"]),
            str(m_row["balance_strategy"]),
        )
        manifest_lookup[m_key] = m_row

    ok_eval_keys: set[Tuple[str, int, int, str, str, str]] = set()
    for idx, r in df_results.iterrows():
        m_key = (
            str(r["target_dataset"]),
            int(r["fold_id"]),
            int(r["seed"]),
            str(r["balance_strategy"]),
        )
        if m_key not in manifest_lookup:
            raise ValueError(
                f"Evaluation at row {idx} ({m_key}) has no matching entry in fold_manifest.csv."
            )
        m_row = manifest_lookup[m_key]
        if str(r["split_signature"]) != str(m_row["split_signature"]):
            raise ValueError(
                f"split_signature mismatch at results.csv row {idx} ({m_key}): "
                f"{r['split_signature']} != {m_row['split_signature']}"
            )
        if str(r["preprocessing_signature"]) != str(m_row["preprocessing_signature"]):
            raise ValueError(
                f"preprocessing_signature mismatch at results.csv row {idx} ({m_key}): "
                f"{r['preprocessing_signature']} != {m_row['preprocessing_signature']}"
            )
        if str(r["status"]) == "OK":
            ok_eval_keys.add(
                (
                    str(r["target_dataset"]),
                    int(r["fold_id"]),
                    int(r["seed"]),
                    str(r["balance_strategy"]),
                    str(r["architecture"]),
                    str(r["method"]),
                )
            )

    # Validate predictions.csv integrity using shared validator + explicit class_order length check
    pred_integrity = validate_predictions_integrity(df_preds, df_manifest_selected)

    if not df_preds.empty:
        pred_eval_keys: set[Tuple[str, int, int, str, str, str]] = set()
        for (ds, fold_id, seed, bal, arch, meth), grp in df_preds.groupby(
            eval_key_cols, sort=False
        ):
            e_key = (str(ds), int(fold_id), int(seed), str(bal), str(arch), str(meth))
            pred_eval_keys.add(e_key)
            if e_key not in ok_eval_keys:
                raise ValueError(
                    f"Prediction group {e_key} has no corresponding OK evaluation in results.csv."
                )
            m_row = manifest_lookup[(str(ds), int(fold_id), int(seed), str(bal))]
            train_ids_set = set(json.loads(str(m_row["train_sample_ids"])))
            pred_ids_set = set(grp["sample_id"].astype(str).tolist())
            if not pred_ids_set.isdisjoint(train_ids_set):
                raise ValueError(
                    f"Test predictions for {e_key} overlap with training sample IDs!"
                )

        missing_pred_groups = ok_eval_keys.difference(pred_eval_keys)
        if missing_pred_groups:
            raise ValueError(
                f"{len(missing_pred_groups)} OK evaluations are missing predictions in predictions.csv."
            )

        for idx, row in df_preds.iterrows():
            p_vec = json.loads(str(row["probabilities"]))
            c_ord = json.loads(str(row["class_order"]))
            if len(p_vec) != len(c_ord):
                raise ValueError(
                    f"Probability vector length ({len(p_vec)}) != class_order length "
                    f"({len(c_ord)}) at predictions.csv row {idx}."
                )

    return {
        "valid": True,
        "is_full_run": bool(is_full_run),
        "expected_evaluations": int(expected_eval_count),
        "actual_evaluations": int(len(df_results)),
        "ok_evaluations": int((df_results["status"] == "OK").sum()),
        "failed_evaluations": int((df_results["status"] != "OK").sum()),
        "prediction_rows_checked": int(pred_integrity.get("checked_rows", 0)),
        "duplicate_evaluation_keys": 0,
        "duplicate_prediction_keys": 0,
        "manifest_signatures_verified": True,
    }


def compute_classical_summaries_and_statistics(
    df_results: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Generate summary and statistical comparison tables for the classical baselines:
      - `summary_by_dataset.csv`
      - `summary_by_architecture.csv`
      - `summary_by_method.csv`
      - `statistical_tests.csv` (pairwise Wilcoxon signed-rank across classical models
        at the seed-averaged LOCO fold level + Holm-Bonferroni correction)
      - `rankings.csv`
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    metrics_cols = ["macro_f1", "accuracy", "balanced_accuracy", "roc_auc"]

    valid_df = df_results[df_results["status"] == "OK"].copy()

    def _agg_block(grp: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "total_evaluations": int(len(grp)),
            "valid_evaluations": int((grp["status"] == "OK").sum()),
            "failed_evaluations": int((grp["status"] != "OK").sum()),
        }
        v_grp = grp[grp["status"] == "OK"]
        for m_col in metrics_cols:
            s = v_grp[m_col].dropna()
            out[f"{m_col}_mean"] = float(s.mean()) if len(s) > 0 else float("nan")
            out[f"{m_col}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            out[f"{m_col}_median"] = float(s.median()) if len(s) > 0 else float("nan")
        return out

    # 1. summary_by_dataset.csv
    ds_rows: List[Dict[str, Any]] = []
    for (ds, arch, meth, bal), grp in df_results.groupby(
        ["target_dataset", "architecture", "method", "balance_strategy"], sort=False
    ):
        row = {
            "target_dataset": ds,
            "architecture": arch,
            "method": meth,
            "balance_strategy": bal,
            "n_folds": int(grp["fold_id"].nunique()),
            "n_seeds": int(grp["seed"].nunique()),
        }
        row.update(_agg_block(grp))
        ds_rows.append(row)
    df_sum_ds = pd.DataFrame(ds_rows)
    p_sum_ds = output_dir / "summary_by_dataset.csv"
    df_sum_ds.to_csv(p_sum_ds, index=False)
    paths["summary_by_dataset"] = p_sum_ds

    # 2. summary_by_architecture.csv
    arch_rows: List[Dict[str, Any]] = []
    for (arch, meth, bal), grp in df_results.groupby(
        ["architecture", "method", "balance_strategy"], sort=False
    ):
        row = {
            "architecture": arch,
            "method": meth,
            "balance_strategy": bal,
            "n_datasets": int(grp["target_dataset"].nunique()),
        }
        row.update(_agg_block(grp))
        arch_rows.append(row)
    df_sum_arch = pd.DataFrame(arch_rows)
    p_sum_arch = output_dir / "summary_by_architecture.csv"
    df_sum_arch.to_csv(p_sum_arch, index=False)
    paths["summary_by_architecture"] = p_sum_arch

    # 3. summary_by_method.csv
    meth_rows: List[Dict[str, Any]] = []
    for (meth, bal), grp in df_results.groupby(
        ["method", "balance_strategy"], sort=False
    ):
        row = {
            "method": meth,
            "balance_strategy": bal,
            "n_architectures": int(grp["architecture"].nunique()),
            "n_datasets": int(grp["target_dataset"].nunique()),
        }
        row.update(_agg_block(grp))
        meth_rows.append(row)
    df_sum_meth = pd.DataFrame(meth_rows)
    p_sum_meth = output_dir / "summary_by_method.csv"
    df_sum_meth.to_csv(p_sum_meth, index=False)
    paths["summary_by_method"] = p_sum_meth

    # 4. Pairwise Wilcoxon signed-rank tests among the classical baselines per dataset & balance_strategy
    stat_rows: List[Dict[str, Any]] = []
    if not valid_df.empty:
        fold_level = (
            valid_df.groupby(
                ["target_dataset", "balance_strategy", "fold_id", "condition", "architecture"],
                as_index=False,
            )[metrics_cols]
            .mean()
        )
        pairs = [
            ("XGBoost", "RandomForest"),
            ("RandomForest", "SVM"),
            ("XGBoost", "SVM"),
        ]
        for (ds, bal), grp in fold_level.groupby(
            ["target_dataset", "balance_strategy"], sort=False
        ):
            for model_a, model_b in pairs:
                sub_a = grp[grp["architecture"] == model_a].set_index("fold_id")
                sub_b = grp[grp["architecture"] == model_b].set_index("fold_id")
                common_folds = sorted(
                    set(sub_a.index.tolist()).intersection(set(sub_b.index.tolist()))
                )
                if not common_folds:
                    continue
                for m_col in metrics_cols:
                    a_vals = sub_a.loc[common_folds, m_col].to_numpy(dtype=np.float64)
                    b_vals = sub_b.loc[common_folds, m_col].to_numpy(dtype=np.float64)
                    valid_mask = ~(np.isnan(a_vals) | np.isnan(b_vals))
                    a_v = a_vals[valid_mask]
                    b_v = b_vals[valid_mask]
                    n_paired = int(len(a_v))
                    if n_paired == 0:
                        continue

                    diffs = a_v - b_v
                    mean_diff = float(np.mean(diffs))
                    median_diff = float(np.median(diffs))
                    std_diff = float(np.std(diffs, ddof=1)) if n_paired > 1 else 0.0
                    if n_paired > 1 and std_diff > 0.0:
                        se = std_diff / math.sqrt(n_paired)
                        t_crit = float(stats.t.ppf(0.975, df=n_paired - 1))
                        ci_low = mean_diff - t_crit * se
                        ci_high = mean_diff + t_crit * se
                    else:
                        ci_low = mean_diff
                        ci_high = mean_diff

                    w_stat = float("nan")
                    p_raw = float("nan")
                    test_note = "OK_wilcoxon_signed_rank"
                    if np.allclose(diffs, 0.0):
                        w_stat = 0.0
                        p_raw = 1.0
                        test_note = "All paired differences are identically zero"
                    else:
                        try:
                            res_w = stats.wilcoxon(
                                diffs, zero_method="zsplit", alternative="two-sided"
                            )
                            w_stat = float(res_w.statistic)
                            p_raw = float(res_w.pvalue)
                            if n_paired < 6:
                                test_note = (
                                    f"Low sample size (N={n_paired} LOCO folds < 6 minimum "
                                    "for alpha=0.05 two-sided Wilcoxon significance)"
                                )
                        except Exception as exc:
                            test_note = f"Wilcoxon error: {exc}"

                    stat_rows.append(
                        {
                            "target_dataset": ds,
                            "balance_strategy": bal,
                            "comparison": f"{model_a} vs. {model_b}",
                            "model_a": model_a,
                            "model_b": model_b,
                            "metric": m_col,
                            "statistical_unit": "loco_fold_seed_averaged",
                            "n_paired_folds": n_paired,
                            "model_a_mean": float(np.mean(a_v)),
                            "model_b_mean": float(np.mean(b_v)),
                            "mean_diff_a_minus_b": mean_diff,
                            "median_diff_a_minus_b": median_diff,
                            "std_diff": std_diff,
                            "ci95_low": float(ci_low),
                            "ci95_high": float(ci_high),
                            "n_folds_a_better": int(np.sum(diffs > 1e-12)),
                            "n_folds_b_better": int(np.sum(diffs < -1e-12)),
                            "n_folds_tied": int(np.sum(np.abs(diffs) <= 1e-12)),
                            "wilcoxon_statistic": w_stat,
                            "p_value_raw": p_raw,
                            "p_value_holm": float("nan"),
                            "significant_alpha_0_05": False,
                            "methodological_note": test_note,
                        }
                    )

    df_stats = pd.DataFrame(stat_rows)
    if not df_stats.empty:
        for (ds, bal, m_col), idxs in df_stats.groupby(
            ["target_dataset", "balance_strategy", "metric"]
        ).groups.items():
            idx_list = list(idxs)
            raw_ps = [float(df_stats.loc[i, "p_value_raw"]) for i in idx_list]
            adj_ps = _holm_bonferroni_adjust(raw_ps)
            for i, adj_p in zip(idx_list, adj_ps):
                df_stats.loc[i, "p_value_holm"] = adj_p
                df_stats.loc[i, "significant_alpha_0_05"] = bool(
                    not math.isnan(adj_p) and adj_p < 0.05
                )

    p_stats = output_dir / "statistical_tests.csv"
    df_stats.to_csv(p_stats, index=False)
    paths["statistical_tests"] = p_stats

    # 5. rankings.csv
    rank_rows: List[Dict[str, Any]] = []
    if not df_sum_ds.empty:
        for (ds, bal), grp in df_sum_ds.groupby(
            ["target_dataset", "balance_strategy"], sort=False
        ):
            grp_sorted = grp.copy()
            grp_sorted["rank_within_dataset"] = grp_sorted["macro_f1_mean"].rank(
                ascending=False, method="average"
            )
            for _, r in grp_sorted.iterrows():
                rank_rows.append(
                    {
                        "scope": "dataset_classical_models",
                        "target_dataset": ds,
                        "architecture": r["architecture"],
                        "method": r["method"],
                        "balance_strategy": bal,
                        "metric": "macro_f1_mean",
                        "metric_value": float(r["macro_f1_mean"]),
                        "rank": float(r["rank_within_dataset"]),
                    }
                )
        df_ds_ranks = pd.DataFrame(rank_rows)
        if not df_ds_ranks.empty:
            for (arch, meth, bal), grp in df_ds_ranks.groupby(
                ["architecture", "method", "balance_strategy"], sort=False
            ):
                rank_rows.append(
                    {
                        "scope": "mean_rank_across_datasets",
                        "target_dataset": "ALL_TARGETS",
                        "architecture": arch,
                        "method": meth,
                        "balance_strategy": bal,
                        "metric": "macro_f1_mean",
                        "metric_value": float(grp["metric_value"].mean()),
                        "rank": float(grp["rank"].mean()),
                    }
                )

    df_ranks = pd.DataFrame(rank_rows)
    p_ranks = output_dir / "rankings.csv"
    df_ranks.to_csv(p_ranks, index=False)
    paths["rankings"] = p_ranks

    return paths


def generate_classical_report_md(
    config: Mapping[str, Any],
    git_meta: Mapping[str, Any],
    integrity_report: Mapping[str, Any],
    df_results: pd.DataFrame,
    df_sum_ds: pd.DataFrame,
    df_sum_arch: pd.DataFrame,
    df_stats: pd.DataFrame,
    df_ranks: pd.DataFrame,
    is_synthetic_smoke: bool = False,
) -> str:
    """Generate the concise Markdown report (`report.md`) required by Section 8."""
    total_evals = int(len(df_results))
    ok_evals = int((df_results["status"] == "OK").sum()) if not df_results.empty else 0
    failed_evals = total_evals - ok_evals

    scope_banner = (
        "> **AVISO DE ESCOPO (EXECUÇÃO PILOTO / SINTÉTICA DE VALIDAÇÃO)**: "
        "Este diretório corresponde a uma execução de validação em subconjunto ou dados sintéticos "
        "(`is_synthetic_smoke=True`). Conforme `AGENTS.md`, resultados de teste rápido validam a "
        "integridade funcional e anti-vazamento do pipeline.\n"
        if is_synthetic_smoke
        else "> **ESCOPO DE DADOS REAIS**: Execução complementar realizada sobre as janelas reais "
        "de `data/processed_tl/detrended_rms/` reutilizando estritamente o `fold_manifest.csv` "
        "de `results/scratch_vs_tl/run_20261105_141743/`.\n"
    )

    ds_cols = [
        "target_dataset",
        "architecture",
        "method",
        "balance_strategy",
        "n_folds",
        "n_seeds",
        "macro_f1_mean",
        "macro_f1_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "accuracy_mean",
        "accuracy_std",
        "roc_auc_mean",
        "roc_auc_std",
    ]
    ds_table = (
        _df_to_table_str(df_sum_ds[[c for c in ds_cols if c in df_sum_ds.columns]])
        if not df_sum_ds.empty
        else "Nenhum resumo por dataset gerado."
    )
    arch_table = _df_to_table_str(df_sum_arch, "Nenhum resumo global por modelo gerado.")
    stats_table = _df_to_table_str(df_stats, "Nenhum teste estatístico pareado gerado.")
    ranks_table = _df_to_table_str(df_ranks, "Nenhum ranking gerado.")
    hp_json = json.dumps(config.get("hyperparameters", {}), indent=2, ensure_ascii=False)

    return f"""# Relatório Técnico — Baselines Clássicos de Machine Learning (`scratch_vs_tl_classical_baselines`)

{scope_banner}

- **Commit Git**: `{git_meta.get('git_commit')}` (`git_dirty={git_meta.get('git_dirty')}`)
- **Manifesto de Referência Reutilizado**: `{config.get('reference_manifest_path')}`
- **Total de Avaliações Executadas**: `{total_evals}` (`OK={ok_evals}`, `Falhas={failed_evals}`)
- **Datasets Alvo**: `{config.get('target_datasets')}`
- **Modelos (`architecture`)**: `{config.get('architectures')}`
- **Método (`method`)**: `{config.get('method')}`
- **Estratégias de Balanceamento**: `{config.get('balance_strategies')}`
- **Seeds**: `{config.get('seeds')}`

---

## 1. Configuração Experimental e Hiperparâmetros dos Modelos

Os três classificadores clássicos (`RandomForest`, `SVM`, `XGBoost`) foram treinados exclusivamente nos dados de treinamento de cada fold LOCO do dataset de destino, reutilizando exatamente os mesmos splits (`train_sample_ids` e `test_sample_ids`), `StandardScaler` (`split_signature`) e balanceamento `SMOTE` (`preprocessing_signature`) registrados em `results/scratch_vs_tl/run_20261105_141743/fold_manifest.csv`.

```json
{hp_json}
```

---

## 2. Verificação de Integridade e Reprodutibilidade

- **Assinaturas de Split e Pré-processamento Verificadas**: `{integrity_report.get('manifest_signatures_verified')}`
- **Duplicatas em `results.csv`**: `{integrity_report.get('duplicate_evaluation_keys')}`
- **Duplicatas em `predictions.csv`**: `{integrity_report.get('duplicate_prediction_keys')}`
- **Predições Individuais Validadas**: `{integrity_report.get('prediction_rows_checked')}`

---

## 3. Resultados por Dataset, Modelo e Estratégia de Balanceamento (Média e Desvio Padrão)

```text
{ds_table}
```

---

## 4. Resumo Global por Modelo (`Architecture-Level Summary`)

```text
{arch_table}
```

---

## 5. Comparações Estatísticas Pareadas entre Baselines Clássicos (Wilcoxon + Holm)

```text
{stats_table}
```

---

## 6. Rankings por Dataset e Média de Ranks (`Macro-F1`)

```text
{ranks_table}
```

---

## 7. Desvios em Relação ao Protocolo Original

- **Nenhum desvio metodológico**: todos os folds, amostras de treino/teste, transformações `StandardScaler`, reamostragens `SMOTE`, vocabulários de classes e definições de métricas (`Accuracy`, `Balanced_Accuracy`, `Macro_F1`, `ROC_AUC`) são idênticos aos do benchmark original `run_20261105_141743`.
"""


# =============================================================================
# 5. MAIN COMPLEMENTARY BENCHMARK ORCHESTRATOR
# =============================================================================


def run_classical_baselines_benchmark(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    manifest_path: Optional[str | Path] = None,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_root: Optional[str | Path] = None,
    resume_dir: Optional[str | Path] = None,
    targets_override: Optional[Sequence[str]] = None,
    architectures_override: Optional[Sequence[str]] = None,
    balance_override: Optional[Sequence[str]] = None,
    seeds_override: Optional[Sequence[int]] = None,
    max_folds: Optional[int] = None,
    feature_extractor: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    enforce_expected_folds: bool = True,
    enforce_full_manifest: bool = True,
    is_synthetic_smoke: bool = False,
) -> Path:
    """
    Execute the Classical Machine Learning Baselines (`RandomForest`, `SVM`, `XGBoost`)
    reusing the exact `fold_manifest.csv` splits from `run_20261105_141743`.
    Supports incremental checkpointing and automatic resumption via `resume_dir`.
    """
    cfg = load_classical_config(config_path)
    targets = (
        list(targets_override)
        if targets_override is not None
        else list(cfg.get("target_datasets", TARGET_DATASETS))
    )
    architectures = (
        list(architectures_override)
        if architectures_override is not None
        else list(cfg.get("architectures", CLASSICAL_ARCHITECTURES))
    )
    method_name = str(cfg.get("method", CLASSICAL_METHOD))
    balance_strategies = (
        list(balance_override)
        if balance_override is not None
        else list(cfg.get("balance_strategies", BALANCE_STRATEGIES))
    )
    seeds = (
        [int(s) for s in seeds_override]
        if seeds_override is not None
        else [int(s) for s in cfg.get("seeds", SEEDS)]
    )
    hp_all = dict(cfg.get("hyperparameters", {}))

    resolved_manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else PROJECT_ROOT / str(cfg.get("reference_manifest_path", DEFAULT_MANIFEST_PATH))
    )
    resolved_output_root = (
        Path(output_root)
        if output_root is not None
        else PROJECT_ROOT / str(cfg.get("output_root", DEFAULT_OUTPUT_ROOT))
    )

    # Guard against accidentally pointing output_root or resume_dir to the original benchmark dir
    if resume_dir is not None and Path(resume_dir).resolve() == DEFAULT_REFERENCE_BENCHMARK_DIR.resolve():
        raise ValueError(
            f"Refusing to write into original benchmark directory '{DEFAULT_REFERENCE_BENCHMARK_DIR}'! "
            "Classical baselines must be saved in an independent directory."
        )

    # Pre-execution checks: dependencies, canonical metadata, data_root, and manifest
    dep_versions = verify_classical_dependencies(architectures)
    verified_folds = verify_canonical_fold_metadata(targets)
    resolved_data_root = validate_data_root_is_detrended_rms(data_root)

    is_filtered_subset = (
        targets_override is not None
        or architectures_override is not None
        or balance_override is not None
        or seeds_override is not None
        or max_folds is not None
    )
    df_manifest_full = load_and_validate_reference_manifest(
        manifest_path=resolved_manifest_path,
        enforce_full_396=(enforce_full_manifest and not is_synthetic_smoke),
    )

    # Filter manifest entries according to requested targets, seeds, balance_strategies, and max_folds
    df_manifest_sel = df_manifest_full[
        df_manifest_full["target_dataset"].isin(targets)
        & df_manifest_full["seed"].astype(int).isin(seeds)
        & df_manifest_full["balance_strategy"].astype(str).isin(balance_strategies)
    ].copy()
    if max_folds is not None:
        df_manifest_sel = df_manifest_sel[
            df_manifest_sel["fold_id"].astype(int) < int(max_folds)
        ].copy()

    if df_manifest_sel.empty:
        raise ValueError(
            "No manifest rows matched the requested targets/seeds/balance_strategies/max_folds filter."
        )

    base_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = 1
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = resolved_output_root / f"run_{base_ts}"
        while run_dir.exists():
            run_dir = resolved_output_root / f"run_{base_ts}_{suffix:02d}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)

    log_path = run_dir / "run.log"
    logger = logging.getLogger(f"{EXPERIMENT_ID}_{base_ts}_{suffix}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    git_meta = collect_git_and_script_hashes()
    git_meta["config_sha256"] = (
        hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
        if Path(config_path).exists()
        else None
    )
    git_meta["reference_manifest_sha256"] = hashlib.sha256(
        resolved_manifest_path.read_bytes()
    ).hexdigest()
    (run_dir / "git_metadata.json").write_text(
        json.dumps(git_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    all_dep_versions = get_dependency_versions()
    all_dep_versions.update(dep_versions)
    env_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependency_versions": all_dep_versions,
    }
    (run_dir / "environment.json").write_text(
        json.dumps(env_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    comparison_config = {
        "experiment_id": EXPERIMENT_ID,
        "task": TASK_NAME,
        "timestamp": base_ts,
        "is_synthetic_smoke": bool(is_synthetic_smoke or is_filtered_subset),
        "reference_manifest_path": str(resolved_manifest_path),
        "reference_manifest_sha256": git_meta["reference_manifest_sha256"],
        "data_root": str(resolved_data_root),
        "output_dir": str(run_dir),
        "target_datasets": targets,
        "expected_loco_folds": verified_folds,
        "architectures": architectures,
        "method": method_name,
        "source_datasets": "None (Target-only Classical Baselines)",
        "preprocessing": {
            "variant": PREPROCESSING_VARIANT,
            "feature_dimension": FEATURE_DIMENSION,
            "scaler": "StandardScaler (fitted strictly on original target training fold before SMOTE)",
        },
        "balance_strategies": balance_strategies,
        "seeds": seeds,
        "max_folds": max_folds,
        "hyperparameters": hp_all,
        "metric_definitions": {
            "primary_metric": "macro_f1 (average='macro' over classes relevant to the fold)",
            "secondary_metrics": [
                "accuracy",
                "balanced_accuracy",
                "roc_auc (macro One-vs-Rest over present test classes when >= 2 classes present; NaN otherwise)",
            ],
        },
        "software_versions": all_dep_versions,
    }
    (run_dir / "comparison_config.json").write_text(
        json.dumps(comparison_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save a copy of the selected reference fold_manifest.csv in run_dir for self-contained traceability
    manifest_csv_path = run_dir / "fold_manifest.csv"
    df_manifest_sel.to_csv(manifest_csv_path, index=False)

    results_csv_path = run_dir / "results.csv"
    preds_csv_path = run_dir / "predictions.csv"

    results_rows: List[Dict[str, Any]] = []
    predictions_rows: List[Dict[str, Any]] = []
    completed_eval_keys: set[Tuple[str, int, int, str, str, str]] = set()

    if resume_dir is not None:
        if results_csv_path.exists():
            df_r_prev = pd.read_csv(results_csv_path)
            ok_prev = df_r_prev[df_r_prev["status"] == "OK"]
            results_rows.extend(ok_prev.to_dict("records"))
            for r in results_rows:
                completed_eval_keys.add(
                    (
                        str(r["target_dataset"]),
                        int(r["fold_id"]),
                        int(r["seed"]),
                        str(r["balance_strategy"]),
                        str(r["architecture"]),
                        str(r["method"]),
                    )
                )
        if preds_csv_path.exists():
            df_p_prev = pd.read_csv(preds_csv_path)
            for r in df_p_prev.to_dict("records"):
                key = (
                    str(r["target_dataset"]),
                    int(r["fold_id"]),
                    int(r["seed"]),
                    str(r["balance_strategy"]),
                    str(r["architecture"]),
                    str(r["method"]),
                )
                if key in completed_eval_keys:
                    predictions_rows.append(r)
        logger.info(
            f"Resuming classical baselines run in {run_dir}: "
            f"loaded {len(completed_eval_keys)} completed evaluations."
        )

    for ds_name in targets:
        ds_manifest_rows = df_manifest_sel[
            df_manifest_sel["target_dataset"] == ds_name
        ]
        if ds_manifest_rows.empty:
            continue

        X_all, y_all_str, conds_all, sample_ids_all, _ = load_and_extract_target_dataset(
            dataset_name=ds_name,
            data_root=resolved_data_root,
            feature_extractor=feature_extractor,
            logger=logger,
            enforce_expected_folds=enforce_expected_folds,
        )

        le_target = LabelEncoder()
        y_all_enc = le_target.fit_transform(y_all_str).astype(np.int64)
        class_names = [str(c) for c in le_target.classes_]
        n_total_classes = len(class_names)

        sample_id_to_idx: Dict[str, int] = {
            str(sid): int(idx) for idx, sid in enumerate(sample_ids_all.tolist())
        }

        for _, m_row in ds_manifest_rows.iterrows():
            fold_id = int(m_row["fold_id"])
            test_cond = str(m_row["condition"])
            seed = int(m_row["seed"])
            balance_strategy = str(m_row["balance_strategy"])

            # Check if all requested classical architectures are already completed for this manifest block
            block_keys = [
                (ds_name, fold_id, seed, balance_strategy, arch, method_name)
                for arch in architectures
            ]
            if all(k in completed_eval_keys for k in block_keys):
                for arch in architectures:
                    logger.info(
                        f"  [SKIP COMPLETED] {ds_name} | Fold {fold_id}:{test_cond} | "
                        f"Seed {seed} | Bal={balance_strategy} | {arch} - {method_name}"
                    )
                continue

            # Recover and verify exact split & preprocessing from the manifest entry
            try:
                recovered = recover_and_verify_fold_from_manifest(
                    manifest_row=m_row,
                    X_all=X_all,
                    y_all_enc=y_all_enc,
                    conds_all=conds_all,
                    sample_id_to_idx=sample_id_to_idx,
                    class_names=class_names,
                )
            except Exception as exc:
                logger.error(
                    f"  [MANIFEST ERROR] {ds_name} | Fold {fold_id}:{test_cond} | "
                    f"Seed {seed} | Bal={balance_strategy}: {exc}"
                )
                for arch in architectures:
                    eval_key = (ds_name, fold_id, seed, balance_strategy, arch, method_name)
                    if eval_key in completed_eval_keys:
                        continue
                    results_rows.append(
                        {
                            "target_dataset": ds_name,
                            "task": TASK_NAME,
                            "fold_id": fold_id,
                            "condition": test_cond,
                            "seed": seed,
                            "architecture": arch,
                            "method": method_name,
                            "balance_strategy": balance_strategy,
                            "source_datasets": "None",
                            "preprocessing": PREPROCESSING_VARIANT,
                            "feature_dimension": int(X_all.shape[1]),
                            "accuracy": float("nan"),
                            "balanced_accuracy": float("nan"),
                            "macro_f1": float("nan"),
                            "roc_auc": float("nan"),
                            "roc_auc_valid": False,
                            "roc_auc_note": f"Manifest verification error: {exc}",
                            "test_sample_count": int(m_row["n_test_samples"]),
                            "test_class_count": 0,
                            "train_sample_count_before": int(m_row["n_train_before_balance"]),
                            "train_sample_count_after": int(m_row["n_train_after_balance"]),
                            "train_class_count": 0,
                            "class_counts_before": str(m_row["class_counts_before"]),
                            "class_counts_after": str(m_row["class_counts_after"]),
                            "backbone_frozen": False,
                            "backbone_param_max_abs_diff": float("nan"),
                            "backbone_buffer_max_abs_diff": float("nan"),
                            "selected_epochs": 0,
                            "split_signature": str(m_row["split_signature"]),
                            "preprocessing_signature": str(m_row["preprocessing_signature"]),
                            "status": "ERROR_MANIFEST_MISMATCH",
                            "error_message": f"{type(exc).__name__}: {exc}",
                            "runtime_seconds": 0.0,
                        }
                    )
                pd.DataFrame(results_rows).to_csv(results_csv_path, index=False)
                continue

            X_train_bal = recovered["X_train_bal"]
            y_train_bal = recovered["y_train_bal"]
            X_test_scaled = recovered["X_test_scaled"]
            y_test = recovered["y_test"]
            test_sample_ids = recovered["test_sample_ids"]
            smote_audit = recovered["smote_audit"]
            split_sig = recovered["split_signature"]
            preproc_sig = recovered["preprocessing_signature"]

            if X_train_bal is None or y_train_bal is None:
                for arch in architectures:
                    eval_key = (ds_name, fold_id, seed, balance_strategy, arch, method_name)
                    if eval_key in completed_eval_keys:
                        continue
                    results_rows.append(
                        {
                            "target_dataset": ds_name,
                            "task": TASK_NAME,
                            "fold_id": fold_id,
                            "condition": test_cond,
                            "seed": seed,
                            "architecture": arch,
                            "method": method_name,
                            "balance_strategy": balance_strategy,
                            "source_datasets": "None",
                            "preprocessing": PREPROCESSING_VARIANT,
                            "feature_dimension": int(X_all.shape[1]),
                            "accuracy": float("nan"),
                            "balanced_accuracy": float("nan"),
                            "macro_f1": float("nan"),
                            "roc_auc": float("nan"),
                            "roc_auc_valid": False,
                            "roc_auc_note": "Fold skipped prior to training due to SMOTE constraint",
                            "test_sample_count": int(len(y_test)),
                            "test_class_count": int(len(np.unique(y_test))),
                            "train_sample_count_before": int(smote_audit["n_train_before"]),
                            "train_sample_count_after": int(smote_audit["n_train_after"]),
                            "train_class_count": int(len(np.unique(recovered["y_train_raw"]))),
                            "class_counts_before": json.dumps(smote_audit["counts_before"], ensure_ascii=False),
                            "class_counts_after": json.dumps(smote_audit["counts_after"], ensure_ascii=False),
                            "backbone_frozen": False,
                            "backbone_param_max_abs_diff": float("nan"),
                            "backbone_buffer_max_abs_diff": float("nan"),
                            "selected_epochs": 0,
                            "split_signature": split_sig,
                            "preprocessing_signature": preproc_sig,
                            "status": str(smote_audit["balance_status"]),
                            "error_message": str(smote_audit["balance_reason"]),
                            "runtime_seconds": 0.0,
                        }
                    )
                pd.DataFrame(results_rows).to_csv(results_csv_path, index=False)
                continue

            for arch in architectures:
                eval_key = (ds_name, fold_id, seed, balance_strategy, arch, method_name)
                if eval_key in completed_eval_keys:
                    logger.info(
                        f"  [SKIP COMPLETED] {ds_name} | Fold {fold_id}:{test_cond} | "
                        f"Seed {seed} | Bal={balance_strategy} | {arch} - {method_name}"
                    )
                    continue

                t0 = time.perf_counter()
                try:
                    preds, probs, model_meta = train_eval_classical_model(
                        architecture=arch,
                        X_train_bal=X_train_bal,
                        y_train_bal=y_train_bal,
                        X_test_scaled=X_test_scaled,
                        n_total_classes=n_total_classes,
                        seed=seed,
                        model_hyperparams=hp_all.get(arch, {}),
                    )
                    runtime_sec = time.perf_counter() - t0
                    metrics = compute_fold_metrics(
                        y_true=y_test,
                        y_pred=preds,
                        probs=probs,
                    )

                    logger.info(
                        f"  [{ds_name} | Fold {fold_id}:{test_cond} | Seed {seed} | "
                        f"Bal={balance_strategy} | {arch} - {method_name}] "
                        f"F1={metrics['Macro_F1']:.4f} | BalAcc={metrics['Balanced_Accuracy']:.4f} | "
                        f"Acc={metrics['Accuracy']:.4f} ({runtime_sec:.2f}s)"
                    )

                    results_rows.append(
                        {
                            "target_dataset": ds_name,
                            "task": TASK_NAME,
                            "fold_id": fold_id,
                            "condition": test_cond,
                            "seed": seed,
                            "architecture": arch,
                            "method": method_name,
                            "balance_strategy": balance_strategy,
                            "source_datasets": "None",
                            "preprocessing": PREPROCESSING_VARIANT,
                            "feature_dimension": int(X_all.shape[1]),
                            "accuracy": float(metrics["Accuracy"]),
                            "balanced_accuracy": float(metrics["Balanced_Accuracy"]),
                            "macro_f1": float(metrics["Macro_F1"]),
                            "roc_auc": float(metrics["ROC_AUC"]),
                            "roc_auc_valid": bool(metrics["ROC_AUC_Valid"]),
                            "roc_auc_note": str(metrics["ROC_AUC_Note"]),
                            "test_sample_count": int(metrics["Test_Sample_Count"]),
                            "test_class_count": int(metrics["Test_Class_Count"]),
                            "train_sample_count_before": int(smote_audit["n_train_before"]),
                            "train_sample_count_after": int(len(y_train_bal)),
                            "train_class_count": int(len(np.unique(y_train_bal))),
                            "class_counts_before": json.dumps(smote_audit["counts_before"], ensure_ascii=False),
                            "class_counts_after": json.dumps(smote_audit["counts_after"], ensure_ascii=False),
                            "backbone_frozen": bool(model_meta["backbone_frozen"]),
                            "backbone_param_max_abs_diff": float(model_meta["backbone_param_max_abs_diff"]),
                            "backbone_buffer_max_abs_diff": float(model_meta["backbone_buffer_max_abs_diff"]),
                            "selected_epochs": int(model_meta["selected_epochs"]),
                            "split_signature": split_sig,
                            "preprocessing_signature": preproc_sig,
                            "status": "OK",
                            "error_message": "",
                            "runtime_seconds": float(round(runtime_sec, 6)),
                        }
                    )
                    completed_eval_keys.add(eval_key)

                    class_order_json = json.dumps(class_names, ensure_ascii=False)
                    for idx_s in range(len(y_test)):
                        predictions_rows.append(
                            {
                                "target_dataset": ds_name,
                                "fold_id": fold_id,
                                "condition": test_cond,
                                "sample_id": str(test_sample_ids[idx_s]),
                                "y_true": int(y_test[idx_s]),
                                "y_pred": int(preds[idx_s]),
                                "probabilities": json.dumps(
                                    [round(float(p), 6) for p in probs[idx_s]]
                                ),
                                "class_order": class_order_json,
                                "architecture": arch,
                                "method": method_name,
                                "balance_strategy": balance_strategy,
                                "seed": seed,
                            }
                        )
                except Exception as exc:
                    runtime_sec = time.perf_counter() - t0
                    logger.error(
                        f"  [ERROR] {ds_name} | Fold {fold_id}:{test_cond} | Seed {seed} | "
                        f"Bal={balance_strategy} | {arch} - {method_name}: {exc}"
                    )
                    results_rows.append(
                        {
                            "target_dataset": ds_name,
                            "task": TASK_NAME,
                            "fold_id": fold_id,
                            "condition": test_cond,
                            "seed": seed,
                            "architecture": arch,
                            "method": method_name,
                            "balance_strategy": balance_strategy,
                            "source_datasets": "None",
                            "preprocessing": PREPROCESSING_VARIANT,
                            "feature_dimension": int(X_all.shape[1]),
                            "accuracy": float("nan"),
                            "balanced_accuracy": float("nan"),
                            "macro_f1": float("nan"),
                            "roc_auc": float("nan"),
                            "roc_auc_valid": False,
                            "roc_auc_note": f"Error: {exc}",
                            "test_sample_count": int(len(y_test)),
                            "test_class_count": int(len(np.unique(y_test))),
                            "train_sample_count_before": int(smote_audit["n_train_before"]),
                            "train_sample_count_after": int(len(y_train_bal)),
                            "train_class_count": int(len(np.unique(y_train_bal))),
                            "class_counts_before": json.dumps(smote_audit["counts_before"], ensure_ascii=False),
                            "class_counts_after": json.dumps(smote_audit["counts_after"], ensure_ascii=False),
                            "backbone_frozen": False,
                            "backbone_param_max_abs_diff": float("nan"),
                            "backbone_buffer_max_abs_diff": float("nan"),
                            "selected_epochs": 0,
                            "split_signature": split_sig,
                            "preprocessing_signature": preproc_sig,
                            "status": "ERROR_MODEL_EXECUTION",
                            "error_message": f"{type(exc).__name__}: {exc}",
                            "runtime_seconds": float(round(runtime_sec, 6)),
                        }
                    )

            pd.DataFrame(results_rows).to_csv(results_csv_path, index=False)
            pd.DataFrame(predictions_rows).to_csv(preds_csv_path, index=False)

    df_results = pd.DataFrame(results_rows)
    df_preds = pd.DataFrame(predictions_rows)

    df_results.to_csv(results_csv_path, index=False)
    df_preds.to_csv(preds_csv_path, index=False)

    is_full_run = (
        not is_filtered_subset
        and not is_synthetic_smoke
        and len(df_manifest_sel) == EXPECTED_MANIFEST_ENTRIES
        and len(architectures) == len(CLASSICAL_ARCHITECTURES)
    )
    integrity_report = validate_post_execution_integrity(
        df_results=df_results,
        df_preds=df_preds,
        df_manifest_selected=df_manifest_sel,
        expected_architectures=architectures,
        expected_method=method_name,
        is_full_run=is_full_run,
    )
    (run_dir / "integrity_validation.json").write_text(
        json.dumps(integrity_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary_paths = compute_classical_summaries_and_statistics(df_results, run_dir)
    df_sum_ds = pd.read_csv(summary_paths["summary_by_dataset"])
    df_sum_arch = pd.read_csv(summary_paths["summary_by_architecture"])
    df_stats = (
        pd.read_csv(summary_paths["statistical_tests"])
        if summary_paths["statistical_tests"].exists()
        and summary_paths["statistical_tests"].stat().st_size > 1
        else pd.DataFrame()
    )
    df_ranks = (
        pd.read_csv(summary_paths["rankings"])
        if summary_paths["rankings"].exists()
        and summary_paths["rankings"].stat().st_size > 1
        else pd.DataFrame()
    )

    report_md = generate_classical_report_md(
        config=comparison_config,
        git_meta=git_meta,
        integrity_report=integrity_report,
        df_results=df_results,
        df_sum_ds=df_sum_ds,
        df_sum_arch=df_sum_arch,
        df_stats=df_stats,
        df_ranks=df_ranks,
        is_synthetic_smoke=bool(is_synthetic_smoke or is_filtered_subset),
    )
    (run_dir / "report.md").write_text(report_md, encoding="utf-8")

    logger.info("=" * 80)
    logger.info(f"CLASSICAL BASELINES BENCHMARK COMPLETED: {run_dir}")
    logger.info("=" * 80)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    return run_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Classical Machine Learning Baselines (RandomForest, SVM, XGBoost) "
            "reusing the fold_manifest.csv from the Scratch vs. Transfer Learning benchmark."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configs/scratch_vs_tl_classical_baselines.yaml.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional override path to fold_manifest.csv.",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Optional existing run_<timestamp> directory to resume from checkpoint.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        choices=TARGET_DATASETS,
        help="Optional override of target datasets.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=None,
        choices=CLASSICAL_ARCHITECTURES,
        help="Optional override of classical models ('RandomForest', 'SVM', 'XGBoost').",
    )
    parser.add_argument(
        "--balance-strategies",
        nargs="+",
        default=None,
        choices=BALANCE_STRATEGIES,
        help="Optional override of balance strategies ('none', 'smote').",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional override of random seeds (default: 42 123 2026).",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional cap on LOCO folds per target dataset (for small-scale pilot validation).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Path to data/processed_tl/{PREPROCESSING_VARIANT}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional override for output base directory.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_classical_baselines_benchmark(
        config_path=args.config,
        manifest_path=args.manifest,
        data_root=args.data_root,
        output_root=args.output_root,
        resume_dir=args.resume_dir,
        targets_override=args.targets,
        architectures_override=args.architectures,
        balance_override=args.balance_strategies,
        seeds_override=args.seeds,
        max_folds=args.max_folds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

