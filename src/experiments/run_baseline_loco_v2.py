#!/usr/bin/env python3
"""
Leakage-Free, Paired Leave-One-Condition-Out (LOCO) Supervised Baseline Benchmark
(`src/experiments/run_baseline_loco_v2.py`).

Experiment ID : `baseline_loco_v2`
Task          : `diagnosis`

Establishes a rigorous, reproducible, and directly comparable target-only reference
for Transfer Learning (TL) experiments in `VibNet-1D` (`scripts/exp_tl_cross_dataset_mlp_tabnet.py`).

Scientific Protocol & Leakage Prevention (`ADR-002`):
  1. Target-only training: Models are trained strictly from scratch on the target
     dataset's `K - 1` training operating conditions, with zero source-domain samples
     and zero pretrained weights.
  2. Representation parity: Uses the `detrended_rms` preprocessing variant from
     `data/processed_tl/detrended_rms/` (never substituting `data/processed/`) and
     the canonical 141-dimensional feature vector (`extract_fusion_features` +
     `extract_advanced_features`), without auxiliary features.
  3. Strict LOCO isolation:
     - All windows of 1 operating condition (`Test_Condition`) are held out as the
       test set; all remaining `K - 1` operating conditions form the target training set.
     - `StandardScaler` is fitted exclusively on the target training data (`X_train`)
       and applied via `.transform()` to `X_train` and `X_test`.
     - `SMOTE` (when `balance_strategy == "smote"`, with
       `k_neighbors = min(5, minority_class_count - 1)`) is applied exclusively to
       the scaled training data (`X_train_scaled`).
     - Test data is never used for hyperparameter tuning, early stopping, epoch
       selection, model selection, threshold selection, feature selection, or scaling.
  4. Internal validation strictly within training conditions:
     - For `TabNet Scratch` early stopping (and optional internal grouped CV), an
       internal validation partition (`inner_train`, `inner_val`) is created
       exclusively within the target training conditions using operating-condition
       groups (`GroupShuffleSplit` / `GroupKFold`) when at least 2 training
       conditions cover the training classes, falling back to a stratified split
       strictly within the training partition if grouped splitting cannot preserve
       class coverage.
     - `TabNet Scratch` passes only `[(X_inner_val_scaled, y_inner_val)]` to
       `eval_set` (never `X_test`), selects `best_epoch` on `inner_val`, and refits
       from scratch on the full allowed balanced training partition for
       `selected_epochs = max(1, best_epoch + 1)` without test leakage.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features
from src.vibnet.data.registry import CANONICAL_DATASETS, get_sampling_rate_hz
from src.vibnet.evaluation.balancing import trim_single_remainder_sample
from src.vibnet.evaluation.protocols import assert_no_condition_leakage
from src.vibnet.evaluation.provenance import collect_run_provenance, write_run_metadata
from src.vibnet.tabular.features import TABVIBNET_FEATURE_DIM

# =============================================================================
# 1. EXPERIMENT CONSTANTS & CANONICAL METADATA
# =============================================================================

EXPERIMENT_ID: str = "baseline_loco_v2"
TASK_NAME: str = "diagnosis"
PREPROCESSING_VARIANT: str = "detrended_rms"
FEATURE_DIMENSION: int = TABVIBNET_FEATURE_DIM  # 141

TARGET_DATASETS: List[str] = [
    "CWRU_12k",
    "CWRU_48k",
    "HUST_Gearbox",
    "UOEMD",
    "PU",
]

EXPECTED_LOCO_FOLDS: Dict[str, int] = {
    "CWRU_12k": 4,
    "CWRU_48k": 12,
    "HUST_Gearbox": 30,
    "UOEMD": 16,
    "PU": 4,
}
EXPECTED_TOTAL_FOLDS: int = 66

CANONICAL_MODELS: List[str] = [
    "Random Forest",
    "SVM (RBF)",
    "XGBoost",
    "MLP Scratch",
    "TabNet Scratch",
]

MODEL_ALIASES: Dict[str, str] = {
    "random forest": "Random Forest",
    "random_forest": "Random Forest",
    "rf": "Random Forest",
    "svm": "SVM (RBF)",
    "svm (rbf)": "SVM (RBF)",
    "svm_rbf": "SVM (RBF)",
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
    "mlp": "MLP Scratch",
    "mlp scratch": "MLP Scratch",
    "mlp_scratch": "MLP Scratch",
    "tabnet": "TabNet Scratch",
    "tabnet scratch": "TabNet Scratch",
    "tabnet_scratch": "TabNet Scratch",
}

BALANCE_STRATEGIES: List[str] = ["none", "smote"]
SEEDS: List[int] = [42, 123, 2026]

REQUIRED_RESULTS_COLUMNS: List[str] = [
    "Dataset",
    "Task",
    "Test_Condition",
    "Fold",
    "Seed",
    "Model",
    "Balance_Strategy",
    "Preprocessing",
    "Feature_Dimension",
    "Accuracy",
    "Balanced_Accuracy",
    "Macro_F1",
    "ROC_AUC",
    "ROC_AUC_Valid",
    "Test_Sample_Count",
    "Test_Class_Count",
    "Train_Sample_Count",
    "Train_Class_Count",
    "Status",
    "Error_Message",
    "Runtime_Seconds",
]

AUDIT_EXTRA_COLUMNS: List[str] = [
    "Train_Sample_Count_Before_Balance",
    "Class_Counts_Before",
    "Class_Counts_After",
    "SMOTE_Params",
    "Unbalanced_Classes",
    "ROC_AUC_Note",
    "Validation_Split_Mode",
    "Selected_Epochs",
    "Sample_Split_Signature",
]

DEFAULT_PROCESSED_TL_DIR: Path = (
    PROJECT_ROOT / "data" / "processed_tl" / PREPROCESSING_VARIANT
)
DEFAULT_OUTPUT_ROOT: Path = RESULTS_DIR / EXPERIMENT_ID

MODEL_HYPERPARAMETERS_DOC: Dict[str, Dict[str, Any]] = {
    "Random Forest": {
        "estimator": "sklearn.ensemble.RandomForestClassifier",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
        "n_jobs": -1,
    },
    "SVM (RBF)": {
        "estimator": "sklearn.svm.SVC",
        "kernel": "rbf",
        "C": 10.0,
        "gamma": "scale",
        "probability": True,
        "class_weight": None,
    },
    "XGBoost": {
        "estimator": "xgboost.XGBClassifier",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "n_jobs": -1,
    },
    "MLP Scratch": {
        "architecture": (
            "MLPEncoder(141->256->128->64, BatchNorm1d, LeakyReLU, Dropout=0.3) + "
            "Bottleneck(64->128->64, BatchNorm1d, LeakyReLU, Dropout=0.2) + "
            "LinearHead(64->C_target)"
        ),
        "initialization": "random_scratch (no source pretraining)",
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
        "scheduler": "StepLR(step_size=10, gamma=0.5)",
        "loss": "CrossEntropyLoss()",
        "epochs": 30,
        "batch_size": 64,
    },
    "TabNet Scratch": {
        "architecture": (
            "TabNetClassifier(n_d=16, n_a=16, n_steps=4, gamma=1.3, "
            "n_independent=2, n_shared=2, mask_type='sparsemax')"
        ),
        "initialization": "random_scratch (from_unsupervised=None, no TabNetPretrainer)",
        "optimizer": "Adam(lr=2e-2, weight_decay=1e-5)",
        "scheduler": "StepLR(step_size=10, gamma=0.9)",
        "max_epochs": 30,
        "patience": 15,
        "early_stopping_eval_set": "internal_validation_from_training_conditions_only",
        "refit_strategy": "refit_full_train_with_selected_epoch",
        "batch_size": 64,
        "virtual_batch_size": 32,
    },
}


# =============================================================================
# 2. REPRODUCIBILITY, SEEDING & METADATA VALIDATION
# =============================================================================


def normalize_model_name(name: str) -> str:
    """Normalize a user-provided model name to its canonical identifier."""
    key = name.strip().lower()
    if key not in MODEL_ALIASES:
        valid = ", ".join(CANONICAL_MODELS)
        raise ValueError(f"Unknown model '{name}'. Expected one of: {valid}.")
    return MODEL_ALIASES[key]


def set_global_seed(seed: int) -> None:
    """Fix global random seeds across Python, NumPy, and PyTorch (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def verify_canonical_fold_metadata(
    datasets: Sequence[str] = TARGET_DATASETS,
) -> Dict[str, int]:
    """
    Verify that the canonical dataset registry (`src.vibnet.data.registry`)
    matches the expected LOCO fold counts for all target datasets:
      CWRU_12k=4, CWRU_48k=12, HUST_Gearbox=30, UOEMD=16, PU=4 (Total=66).
    """
    verified: Dict[str, int] = {}
    for ds_name in datasets:
        if ds_name not in CANONICAL_DATASETS:
            raise KeyError(f"Dataset '{ds_name}' is not registered in CANONICAL_DATASETS.")
        spec = CANONICAL_DATASETS[ds_name]
        if not spec.loco_eligible:
            raise ValueError(f"Dataset '{ds_name}' is not marked as LOCO-eligible.")
        expected = EXPECTED_LOCO_FOLDS.get(ds_name)
        if expected is not None and spec.num_conditions != expected:
            raise ValueError(
                f"Fold count mismatch for '{ds_name}': registry has "
                f"{spec.num_conditions}, expected {expected}."
            )
        verified[ds_name] = spec.num_conditions

    if set(datasets) == set(TARGET_DATASETS):
        total = sum(verified.values())
        if total != EXPECTED_TOTAL_FOLDS:
            raise ValueError(
                f"Expected total LOCO folds across 5 targets to be {EXPECTED_TOTAL_FOLDS}, "
                f"got {total}."
            )
    return verified


def validate_data_root_is_detrended_rms(data_root: str | Path) -> Path:
    """
    Validate that `data_root` points to a `detrended_rms` directory and never
    silently substitutes `data/processed/`.
    """
    resolved = Path(data_root).resolve()
    parts = [p.lower() for p in resolved.parts]

    # Explicitly forbid legacy `data/processed` (without `processed_tl/detrended_rms`)
    if "processed" in parts and "processed_tl" not in parts:
        raise ValueError(
            f"Invalid data_root '{resolved}': legacy 'data/processed/' is disallowed "
            f"for '{EXPERIMENT_ID}'. Must use 'data/processed_tl/{PREPROCESSING_VARIANT}/'."
        )
    if resolved.name != PREPROCESSING_VARIANT:
        raise ValueError(
            f"Invalid data_root '{resolved}': directory basename must be "
            f"'{PREPROCESSING_VARIANT}' to guarantee representation parity with TL."
        )
    return resolved


# =============================================================================
# 3. DATASET LOADING & 141-DIMENSIONAL FEATURE EXTRACTION
# =============================================================================


def default_141_feature_extractor(X_windows: np.ndarray, fs_hz: int) -> np.ndarray:
    """
    Extract the canonical 141-dimensional TabVibNet feature matrix using
    `extract_fusion_features(X_windows, fs_hz, extract_advanced_features)` and
    sanitize NaN/Inf values identically to `scripts/exp_tl_cross_dataset_mlp_tabnet.py`.
    """
    X_feat = extract_fusion_features(X_windows, fs_hz, extract_advanced_features)
    X_feat = np.nan_to_num(
        np.asarray(X_feat, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if X_feat.ndim != 2 or X_feat.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"Canonical feature extractor must return shape (N, {FEATURE_DIMENSION}), "
            f"got {X_feat.shape}."
        )
    return X_feat


def load_and_extract_target_dataset(
    dataset_name: str,
    data_root: str | Path,
    feature_extractor: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    logger: Optional[logging.Logger] = None,
    enforce_expected_folds: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load pre-normalized 1-second `.npy` windows from
    `<data_root>/<dataset_name>/<Condition>/<Class>/*.npy` in deterministic sorted
    order and extract the 141-dimensional feature representation.

    Returns:
      `(X_features, y_labels_str, conditions_str, sample_ids_str, audit_dict)`
    """
    root_path = validate_data_root_is_detrended_rms(data_root)
    ds_dir = root_path / dataset_name
    if not ds_dir.exists() or not ds_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found for '{dataset_name}': {ds_dir}. "
            f"Ensure 'data/processed_tl/{PREPROCESSING_VARIANT}/{dataset_name}' is available."
        )

    fs_hz = get_sampling_rate_hz(dataset_name)
    extractor_fn = feature_extractor or default_141_feature_extractor

    X_windows_list: List[np.ndarray] = []
    y_labels_list: List[str] = []
    conditions_list: List[str] = []
    sample_ids_list: List[str] = []

    cond_dirs = sorted([d for d in ds_dir.iterdir() if d.is_dir()])
    for cond_dir in cond_dirs:
        cond_name = cond_dir.name
        class_dirs = sorted([d for d in cond_dir.iterdir() if d.is_dir()])
        for class_dir in class_dirs:
            raw_cls = class_dir.name
            cls_label = (
                raw_cls
                if raw_cls.startswith("Class_")
                else f"Class_{raw_cls}"
            )
            npy_files = sorted(class_dir.glob("*.npy"))
            for fpath in npy_files:
                arr = np.load(fpath)
                arr_1d = np.asarray(arr, dtype=np.float32).reshape(-1)
                X_windows_list.append(arr_1d)
                y_labels_list.append(cls_label)
                conditions_list.append(cond_name)
                sample_ids_list.append(f"{dataset_name}/{cond_name}/{raw_cls}/{fpath.name}")

    if not X_windows_list:
        raise ValueError(f"No .npy window files found under {ds_dir}.")

    # Homogenize window length if minor padding/truncation differences exist
    lengths = {len(w) for w in X_windows_list}
    if len(lengths) == 1:
        X_windows = np.stack(X_windows_list, axis=0)
    else:
        min_len = min(lengths)
        X_windows = np.stack([w[:min_len] for w in X_windows_list], axis=0)

    if logger is not None:
        logger.info(
            f"[{dataset_name}] Loaded {len(X_windows)} windows across "
            f"{len(set(conditions_list))} conditions (fs={fs_hz} Hz). Extracting {FEATURE_DIMENSION} features..."
        )

    X_features = extractor_fn(X_windows, fs_hz)
    X_features = np.nan_to_num(
        np.asarray(X_features, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if X_features.ndim != 2 or X_features.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"Feature matrix for '{dataset_name}' must have shape (N, {FEATURE_DIMENSION}), "
            f"got {X_features.shape}."
        )

    y_arr = np.asarray(y_labels_list, dtype=str)
    conds_arr = np.asarray(conditions_list, dtype=str)
    sample_ids_arr = np.asarray(sample_ids_list, dtype=str)

    unique_conds = sorted(np.unique(conds_arr).tolist())
    unique_classes = sorted(np.unique(y_arr).tolist())

    if enforce_expected_folds and dataset_name in EXPECTED_LOCO_FOLDS:
        expected_k = EXPECTED_LOCO_FOLDS[dataset_name]
        if len(unique_conds) != expected_k:
            raise ValueError(
                f"Dataset '{dataset_name}' on disk has {len(unique_conds)} conditions "
                f"({unique_conds}), but protocol requires {expected_k} LOCO folds."
            )

    cls_counts = {str(k): int(v) for k, v in sorted(Counter(y_arr).items())}
    audit = {
        "dataset": dataset_name,
        "fs_hz": int(fs_hz),
        "data_dir": str(ds_dir),
        "n_windows": int(len(X_features)),
        "feature_dim": int(X_features.shape[1]),
        "n_conditions": int(len(unique_conds)),
        "conditions": unique_conds,
        "n_classes": int(len(unique_classes)),
        "classes": unique_classes,
        "class_distribution": cls_counts,
    }
    return X_features, y_arr, conds_arr, sample_ids_arr, audit


# =============================================================================
# 4. STRICT LOCO ISOLATION, SCALING, SMOTE & INTERNAL GROUPED VALIDATION
# =============================================================================


def compute_sample_split_signature(
    train_sample_ids: Sequence[str],
    test_sample_ids: Sequence[str],
) -> str:
    """
    Compute a deterministic SHA-256 signature of the exact ordered training and
    test sample identifiers for a LOCO fold, verifying that all models in a fold
    are evaluated on the exact same sample partition.
    """
    payload = (
        "TRAIN:"
        + "|".join(str(s) for s in train_sample_ids)
        + "||TEST:"
        + "|".join(str(s) for s in test_sample_ids)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def scale_loco_fold_strict(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit `StandardScaler` exclusively on `X_train` (target training conditions)
    and transform both `X_train` and `X_test`.
    """
    if len(X_train) == 0:
        raise ValueError("Cannot fit StandardScaler on empty X_train.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    if len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test).astype(np.float32)
    else:
        X_test_scaled = np.empty((0, X_train.shape[1]), dtype=np.float32)
    return X_train_scaled, X_test_scaled, scaler


def apply_smote_with_audit(
    X_train_scaled: np.ndarray,
    y_train_encoded: np.ndarray,
    class_names: Sequence[str],
    strategy: str,
    seed: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Apply class balancing (`"none"` or `"smote"`) strictly to the scaled training
    partition (`X_train_scaled`).

    Rules (`Section 6`):
      - `strategy == "none"`: returns `X_train_scaled, y_train_encoded` unchanged.
      - `strategy == "smote"`:
        - Uses `k_neighbors = min(5, minority_class_count - 1)`.
        - Seeds `SMOTE` with `random_state=seed`.
        - If any training class has `< 2` samples (or `< 2` classes exist),
          does NOT silently fake or replace SMOTE; returns `(None, None, audit)`
          with `balance_status = "FAILED_SMOTE_MIN_SAMPLES"`.
    """
    if strategy not in ("none", "smote"):
        raise ValueError(
            f"Unsupported balance_strategy '{strategy}'. Expected one of {BALANCE_STRATEGIES}."
        )

    raw_counts = Counter(int(c) for c in y_train_encoded)
    counts_before = {
        str(class_names[idx]): int(raw_counts.get(idx, 0))
        for idx in sorted(raw_counts.keys())
    }
    n_before = int(len(y_train_encoded))

    audit: Dict[str, Any] = {
        "strategy": strategy,
        "n_train_before": n_before,
        "n_train_after": n_before,
        "counts_before": counts_before,
        "counts_after": dict(counts_before),
        "smote_params": {},
        "unbalanced_classes": {},
        "balance_status": "OK",
        "balance_reason": "OK",
    }

    if strategy == "none":
        audit["smote_params"] = {"strategy": "none"}
        return X_train_scaled, y_train_encoded, audit

    present_classes = sorted(raw_counts.keys())
    if len(present_classes) < 2:
        audit["balance_status"] = "FAILED_SINGLE_CLASS_TRAIN"
        audit["balance_reason"] = (
            f"Training fold contains only {len(present_classes)} class: {counts_before}"
        )
        return None, None, audit

    max_count = max(raw_counts.values())
    min_count = min(raw_counts.values())

    if min_count == max_count:
        audit["balance_status"] = "OK_ALREADY_BALANCED"
        audit["balance_reason"] = (
            f"Training classes already equally distributed ({max_count} samples/class)."
        )
        audit["smote_params"] = {
            "strategy": "smote",
            "applied": False,
            "reason": "already_balanced",
            "random_state": int(seed),
        }
        return X_train_scaled, y_train_encoded, audit

    ineligible = {
        str(class_names[c]): int(cnt)
        for c, cnt in raw_counts.items()
        if cnt < 2
    }
    if ineligible:
        audit["unbalanced_classes"] = ineligible
        audit["balance_status"] = "FAILED_SMOTE_MIN_SAMPLES"
        audit["balance_reason"] = (
            f"Classes with < 2 samples in training fold prevents SMOTE interpolation: {ineligible}"
        )
        audit["smote_params"] = {
            "strategy": "smote",
            "applied": False,
            "random_state": int(seed),
            "ineligible_classes": ineligible,
        }
        return None, None, audit

    k_neighbors = min(5, min_count - 1)
    smote = SMOTE(
        sampling_strategy="auto",
        random_state=int(seed),
        k_neighbors=int(k_neighbors),
    )
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train_encoded)
    X_res = np.asarray(X_res, dtype=np.float32)
    y_res = np.asarray(y_res, dtype=np.int64)

    after_counts = Counter(int(c) for c in y_res)
    counts_after = {
        str(class_names[idx]): int(after_counts.get(idx, 0))
        for idx in sorted(after_counts.keys())
    }

    audit["n_train_after"] = int(len(y_res))
    audit["counts_after"] = counts_after
    audit["smote_params"] = {
        "strategy": "smote",
        "applied": True,
        "random_state": int(seed),
        "k_neighbors": int(k_neighbors),
        "target_majority_count": int(max_count),
    }
    return X_res, y_res, audit


def create_internal_validation_split(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    conds_train: np.ndarray,
    seed: int,
    val_fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Create an internal `(inner_train_idx, inner_val_idx, split_mode)` partition
    exclusively within the target training set (`X_train_raw`, `y_train`, `conds_train`).

    Priority:
      1. Condition-grouped split (`GroupShuffleSplit` on `conds_train`) holding out
         `max(1, round(val_fraction * n_train_conditions))` training conditions for
         internal validation, verifying that `inner_train` retains all classes
         present in `y_train`.
      2. Deterministic condition search across candidate held-out training conditions
         if the random draw happens to drop a class.
      3. Stratified split (`StratifiedShuffleSplit`) strictly within `X_train_raw`
         if holding out any entire training condition would remove a class from
         `inner_train` (e.g., when a class occurs in only 1 training condition).

    Guarantees:
      - `inner_train_idx` and `inner_val_idx` are strictly disjoint subsets of
        `0 .. len(X_train_raw) - 1`.
      - The held-out LOCO test condition is never present in `X_train_raw` or
        `conds_train`.
    """
    n_samples = len(y_train)
    all_indices = np.arange(n_samples, dtype=np.int64)
    full_classes = set(np.unique(y_train).tolist())

    unique_train_conds = sorted(np.unique(conds_train).tolist())
    if len(unique_train_conds) >= 2:
        n_val_groups = max(1, int(round(val_fraction * len(unique_train_conds))))
        n_val_groups = min(n_val_groups, len(unique_train_conds) - 1)

        gss = GroupShuffleSplit(
            n_splits=min(10, len(unique_train_conds)),
            test_size=n_val_groups,
            random_state=int(seed),
        )
        for inner_tr_idx, inner_val_idx in gss.split(X_train_raw, y_train, groups=conds_train):
            if set(np.unique(y_train[inner_tr_idx]).tolist()) == full_classes and len(inner_val_idx) > 0:
                return (
                    np.asarray(inner_tr_idx, dtype=np.int64),
                    np.asarray(inner_val_idx, dtype=np.int64),
                    "grouped_condition",
                )

        # Deterministic fallback over single training conditions ordered by seed-shuffled list
        rng = random.Random(int(seed))
        candidates = list(unique_train_conds)
        rng.shuffle(candidates)
        for val_cond in candidates:
            val_mask = conds_train == val_cond
            tr_mask = ~val_mask
            inner_tr_idx = all_indices[tr_mask]
            inner_val_idx = all_indices[val_mask]
            if (
                len(inner_tr_idx) > 0
                and len(inner_val_idx) > 0
                and set(np.unique(y_train[inner_tr_idx]).tolist()) == full_classes
            ):
                return inner_tr_idx, inner_val_idx, "grouped_condition"

    # Stratified split strictly within the training partition if grouped split is not feasible
    min_class_count = min(Counter(int(c) for c in y_train).values())
    if min_class_count >= 2 and n_samples >= 2 * len(full_classes):
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=max(len(full_classes), int(round(val_fraction * n_samples))),
            random_state=int(seed),
        )
        for inner_tr_idx, inner_val_idx in sss.split(X_train_raw, y_train):
            return (
                np.asarray(inner_tr_idx, dtype=np.int64),
                np.asarray(inner_val_idx, dtype=np.int64),
                "stratified_within_train",
            )

    # Tiny-sample fallback (strictly within training data)
    split_point = max(1, int(round((1.0 - val_fraction) * n_samples)))
    if split_point >= n_samples:
        split_point = max(1, n_samples - 1)
    return (
        all_indices[:split_point],
        all_indices[split_point:],
        "sequential_within_train",
    )


# =============================================================================
# 5. PROBABILITY PROJECTION & FOLD METRICS
# =============================================================================


def project_predictions_to_full_vocabulary(
    preds_local: np.ndarray,
    probs_local: np.ndarray,
    present_train_classes: np.ndarray,
    n_total_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map predictions and probability columns from a model trained on
    `present_train_classes` (`0 .. len(present_train_classes) - 1`) back into the
    dataset's canonical class vocabulary (`0 .. n_total_classes - 1`).
    """
    present_arr = np.asarray(present_train_classes, dtype=np.int64)
    preds_mapped = present_arr[np.asarray(preds_local, dtype=np.int64)]

    probs_local_arr = np.asarray(probs_local, dtype=np.float64)
    if probs_local_arr.ndim == 1:
        probs_local_arr = np.column_stack([1.0 - probs_local_arr, probs_local_arr])

    n_samples = len(preds_mapped)
    probs_full = np.zeros((n_samples, int(n_total_classes)), dtype=np.float64)
    for local_col, global_cls in enumerate(present_arr):
        if local_col < probs_local_arr.shape[1] and 0 <= global_cls < n_total_classes:
            probs_full[:, int(global_cls)] = probs_local_arr[:, local_col]

    row_sums = probs_full.sum(axis=1, keepdims=True)
    nonzero_rows = row_sums.squeeze(-1) > 0.0
    if np.any(nonzero_rows):
        probs_full[nonzero_rows] = probs_full[nonzero_rows] / row_sums[nonzero_rows]

    return preds_mapped, probs_full


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    Compute fold-level classification metrics aligned with
    `scripts/exp_tl_cross_dataset_mlp_tabnet.py`:
      - `Accuracy`
      - `Balanced_Accuracy`
      - `Macro_F1` (over classes present/relevant in the fold)
      - `ROC_AUC` (One-vs-Rest macro; supports binary subset `len(present_true) == 2`
        such as `CWRU_48k` fold `Load_0HP_Sev_0.021` as well as `len(present_true) > 2`)
      - `ROC_AUC_Valid` (`False` with `ROC_AUC = np.nan` when `< 2` classes are present
        in `y_true` or probabilities are degenerate — never substituting `0.0`).
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    present_true = np.unique(y_true_arr)
    present_pred = np.unique(y_pred_arr)
    relevant_classes = np.union1d(present_true, present_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            labels=relevant_classes,
            average="macro",
            zero_division=0,
        )
    )

    roc_auc = float("nan")
    roc_auc_valid = False
    roc_auc_note = "OK"

    if len(present_true) < 2:
        roc_auc_note = (
            f"Undefined: test fold contains only {len(present_true)} class "
            f"({present_true.tolist()})"
        )
    elif probs is None:
        roc_auc_note = "Probabilities not returned by model"
    else:
        try:
            probs_arr = np.asarray(probs, dtype=np.float64)
            if probs_arr.ndim != 2 or len(probs_arr) != len(y_true_arr):
                roc_auc_note = f"Invalid probability shape: {probs_arr.shape}"
            elif np.any(np.isnan(probs_arr)) or np.any(np.isinf(probs_arr)):
                roc_auc_note = "NaN or Inf detected in predicted probabilities"
            elif np.max(present_true) >= probs_arr.shape[1]:
                roc_auc_note = (
                    f"Test class index {int(np.max(present_true))} exceeds "
                    f"probability columns ({probs_arr.shape[1]})"
                )
            else:
                probs_sub = probs_arr[:, present_true]
                row_sums = probs_sub.sum(axis=1, keepdims=True)
                if np.any(row_sums <= 0.0):
                    roc_auc_note = "Zero probability mass on present test classes"
                else:
                    probs_sub = probs_sub / row_sums
                    y_true_remapped = np.searchsorted(present_true, y_true_arr)
                    if len(present_true) == 2:
                        roc_auc = float(roc_auc_score(y_true_remapped, probs_sub[:, 1]))
                        roc_auc_valid = True
                        roc_auc_note = (
                            "OK_binary_subset"
                            if probs_arr.shape[1] > 2
                            else "OK_binary"
                        )
                    else:
                        roc_auc = float(
                            roc_auc_score(
                                y_true_remapped,
                                probs_sub,
                                multi_class="ovr",
                                average="macro",
                            )
                        )
                        roc_auc_valid = True
                        roc_auc_note = (
                            "OK_multiclass_subset_ovr"
                            if len(present_true) < probs_arr.shape[1]
                            else "OK_multiclass_ovr"
                        )
        except Exception as exc:
            roc_auc = float("nan")
            roc_auc_valid = False
            roc_auc_note = f"ROC-AUC calculation error: {exc}"

    return {
        "Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "Macro_F1": macro_f1,
        "ROC_AUC": roc_auc,
        "ROC_AUC_Valid": roc_auc_valid,
        "ROC_AUC_Note": roc_auc_note,
        "Test_Sample_Count": int(len(y_true_arr)),
        "Test_Class_Count": int(len(present_true)),
    }


# =============================================================================
# 6. FIVE TARGET-ONLY BASELINE MODELS (NO PRETRAINING / NO SOURCE DATA)
# =============================================================================


def _prepare_contiguous_labels(
    y_train_bal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map `y_train_bal` onto contiguous `0 .. C_present - 1` labels and return
    `(y_local, present_train_classes)`.
    """
    present_train_classes = np.unique(np.asarray(y_train_bal, dtype=np.int64))
    y_local = np.searchsorted(present_train_classes, y_train_bal).astype(np.int64)
    return y_local, present_train_classes


def train_eval_random_forest(
    X_train_bal: np.ndarray,
    y_train_bal: np.ndarray,
    X_test_scaled: np.ndarray,
    n_total_classes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Train Random Forest from scratch on target training fold."""
    set_global_seed(seed)
    y_local, present_classes = _prepare_contiguous_labels(y_train_bal)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=int(seed),
        n_jobs=-1,
    )
    clf.fit(X_train_bal, y_local)
    preds_local = clf.predict(X_test_scaled)
    probs_local = clf.predict_proba(X_test_scaled)

    preds, probs = project_predictions_to_full_vocabulary(
        preds_local=preds_local,
        probs_local=probs_local,
        present_train_classes=present_classes,
        n_total_classes=n_total_classes,
    )
    return preds, probs, {"validation_split_mode": "none_fixed_hyperparams", "selected_epochs": 0}


def train_eval_svm_rbf(
    X_train_bal: np.ndarray,
    y_train_bal: np.ndarray,
    X_test_scaled: np.ndarray,
    n_total_classes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Train SVM (RBF kernel) from scratch on target training fold."""
    set_global_seed(seed)
    y_local, present_classes = _prepare_contiguous_labels(y_train_bal)

    clf = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        random_state=int(seed),
    )
    clf.fit(X_train_bal, y_local)
    preds_local = clf.predict(X_test_scaled)
    probs_local = clf.predict_proba(X_test_scaled)

    preds, probs = project_predictions_to_full_vocabulary(
        preds_local=preds_local,
        probs_local=probs_local,
        present_train_classes=present_classes,
        n_total_classes=n_total_classes,
    )
    return preds, probs, {"validation_split_mode": "none_fixed_hyperparams", "selected_epochs": 0}


def train_eval_xgboost(
    X_train_bal: np.ndarray,
    y_train_bal: np.ndarray,
    X_test_scaled: np.ndarray,
    n_total_classes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Train XGBoost from scratch on target training fold."""
    from xgboost import XGBClassifier

    set_global_seed(seed)
    y_local, present_classes = _prepare_contiguous_labels(y_train_bal)
    n_present = len(present_classes)

    xgb_kwargs: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss" if n_present > 2 else "logloss",
        "random_state": int(seed),
        "n_jobs": -1,
        "verbosity": 0,
    }
    if n_present > 2:
        xgb_kwargs["objective"] = "multi:softprob"
        xgb_kwargs["num_class"] = int(n_present)
    else:
        xgb_kwargs["objective"] = "binary:logistic"

    clf = XGBClassifier(**xgb_kwargs)
    clf.fit(X_train_bal, y_local)
    preds_local = clf.predict(X_test_scaled)
    probs_local = clf.predict_proba(X_test_scaled)

    preds, probs = project_predictions_to_full_vocabulary(
        preds_local=preds_local,
        probs_local=probs_local,
        present_train_classes=present_classes,
        n_total_classes=n_total_classes,
    )
    return preds, probs, {"validation_split_mode": "none_fixed_hyperparams", "selected_epochs": 200}


def train_eval_mlp_scratch(
    X_train_bal: np.ndarray,
    y_train_bal: np.ndarray,
    X_test_scaled: np.ndarray,
    n_total_classes: int,
    seed: int,
    epochs: int = 30,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Train MLP Scratch from random initialization on the target training fold.

    Matches the exact encoder + bottleneck + domain-head capacity of `HybridDLModel`
    in `scripts/exp_tl_cross_dataset_mlp_tabnet.py` (`141 -> 256 -> 128 -> 64 -> 128 -> 64 -> C_target`),
    trained exclusively on target training data without any source domain or pretrained weights.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.build_tabnet_resnet_balance import HybridDLModel

    set_global_seed(seed)

    # Protect BatchNorm1d against remainder-1 final mini-batch
    X_work = np.asarray(X_train_bal, dtype=np.float32)
    y_work = np.asarray(y_train_bal, dtype=np.int64)
    eff_batch = min(int(batch_size), max(2, len(X_work)))
    if len(X_work) > eff_batch and (len(X_work) % eff_batch == 1):
        X_trim, y_trim, _ = trim_single_remainder_sample(
            X_work.tolist(),
            y_work.tolist(),
            batch_size=eff_batch,
            seed=seed,
        )
        X_work = np.asarray(X_trim, dtype=np.float32)
        y_work = np.asarray(y_trim, dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_key = "target"
    model = HybridDLModel(
        num_features=int(X_work.shape[1]),
        dataset_classes_dict={target_key: int(n_total_classes)},
        encoder_type="mlp",
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    gen = torch.Generator()
    gen.manual_seed(int(seed))

    ds_train = TensorDataset(
        torch.tensor(X_work, dtype=torch.float32),
        torch.tensor(y_work, dtype=torch.long),
    )
    drop_last = len(ds_train) > eff_batch and (len(ds_train) % eff_batch == 1)
    loader = DataLoader(
        ds_train,
        batch_size=eff_batch,
        shuffle=True,
        drop_last=drop_last,
        generator=gen,
    )

    model.train()
    for _ in range(int(epochs)):
        for bx, by in loader:
            if len(bx) <= 1:
                continue
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            logits, _ = model(bx, dataset_name=target_key)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        logits_te, _ = model(X_te_t, dataset_name=target_key)
        probs = torch.softmax(logits_te, dim=1).cpu().numpy().astype(np.float64)
        preds = np.argmax(probs, axis=1).astype(np.int64)

    return (
        preds,
        probs,
        {
            "validation_split_mode": "none_fixed_epochs",
            "selected_epochs": int(epochs),
        },
    )


def train_eval_tabnet_scratch(
    X_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    conds_train_raw: np.ndarray,
    X_train_bal: np.ndarray,
    y_train_bal: np.ndarray,
    X_test_scaled: np.ndarray,
    class_names: Sequence[str],
    n_total_classes: int,
    balance_strategy: str,
    seed: int,
    max_epochs: int = 30,
    patience: int = 15,
    batch_size: int = 64,
    refit_full_train: bool = True,
    eval_set_spy: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Train TabNet Scratch from random initialization (`from_unsupervised=None`),
    using ONLY an internal validation split created within the target training
    conditions for early stopping.

    Two-stage leak-free protocol (`Section 4 & Section 5`):
      1. Internal validation for epoch selection:
         - Split `(X_train_raw, y_train_raw, conds_train_raw)` into `inner_train`
           and `inner_val` using condition-grouped splitting (`GroupShuffleSplit`
           on `conds_train_raw`, or stratified split within `X_train_raw` as fallback).
         - Fit an inner `StandardScaler` strictly on `X_inner_train` and transform
           `X_inner_train` and `X_inner_val`.
         - Apply `balance_strategy` (`"none"` or `"smote"`) strictly to
           `X_inner_train_scaled` (fallback to un-oversampled `X_inner_train_scaled`
           only if the inner subset has `< 2` samples in a minority class).
         - Fit `TabNetClassifier` with `eval_set=[(X_inner_val_scaled, y_inner_val)]`
           and `patience=patience` to select `selected_epochs = max(1, best_epoch + 1)`.
           Notice `X_test_scaled` is NEVER passed to `eval_set`.
      2. Final training on complete allowed training data (`refit_full_train=True`):
         - Refit a fresh `TabNetClassifier` from scratch (same `seed`) on the full
           balanced target training set `(X_train_bal, y_train_bal)` for
           `max_epochs=selected_epochs` with `eval_set=[]` (`patience=0`).
    """
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier

    set_global_seed(seed)

    def _make_tabnet_classifier(run_seed: int) -> TabNetClassifier:
        return TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=4,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 10, "gamma": 0.9},
            mask_type="sparsemax",
            seed=int(run_seed),
            verbose=0,
        )

    def _safe_batch_params(X_arr: np.ndarray, y_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        n_s = len(X_arr)
        eff_b = min(int(batch_size), max(2, n_s))
        if n_s > eff_b and (n_s % eff_b == 1):
            X_trm, y_trm, _ = trim_single_remainder_sample(
                X_arr.tolist(),
                y_arr.tolist(),
                batch_size=eff_b,
                seed=seed,
            )
            X_arr = np.asarray(X_trm, dtype=np.float32)
            y_arr = np.asarray(y_trm, dtype=np.int64)
            n_s = len(X_arr)
            eff_b = min(int(batch_size), max(2, n_s))
        eff_vb = min(32, max(2, eff_b // 2))
        return X_arr, y_arr, eff_b, eff_vb

    # Stage 1: Internal validation split exclusively within target training conditions
    inner_tr_idx, inner_val_idx, split_mode = create_internal_validation_split(
        X_train_raw=X_train_raw,
        y_train=y_train_raw,
        conds_train=conds_train_raw,
        seed=seed,
        val_fraction=0.2,
    )

    X_inner_tr_raw = X_train_raw[inner_tr_idx]
    y_inner_tr_raw = y_train_raw[inner_tr_idx]
    X_inner_val_raw = X_train_raw[inner_val_idx]
    y_inner_val_raw = y_train_raw[inner_val_idx]

    X_inner_tr_scaled, X_inner_val_scaled, _ = scale_loco_fold_strict(
        X_inner_tr_raw, X_inner_val_raw
    )

    X_inner_tr_bal, y_inner_tr_bal, _ = apply_smote_with_audit(
        X_train_scaled=X_inner_tr_scaled,
        y_train_encoded=y_inner_tr_raw,
        class_names=class_names,
        strategy=balance_strategy,
        seed=seed,
    )
    if X_inner_tr_bal is None or y_inner_tr_bal is None:
        # If the inner subset has only 1 sample of a rare class, keep inner_tr un-oversampled for epoch selection
        X_inner_tr_bal = X_inner_tr_scaled
        y_inner_tr_bal = y_inner_tr_raw

    y_inner_tr_local, inner_present_classes = _prepare_contiguous_labels(y_inner_tr_bal)

    # Filter inner_val to classes present in inner_train so TabNet eval_metric never crashes
    val_keep_mask = np.isin(y_inner_val_raw, inner_present_classes)
    X_inner_val_eff = X_inner_val_scaled[val_keep_mask]
    y_inner_val_eff = np.searchsorted(
        inner_present_classes, y_inner_val_raw[val_keep_mask]
    ).astype(np.int64)

    X_inner_tr_safe, y_inner_tr_safe, eff_b_inner, eff_vb_inner = _safe_batch_params(
        X_inner_tr_bal, y_inner_tr_local
    )

    selected_epochs = int(max_epochs)
    clf_inner = _make_tabnet_classifier(seed)

    if len(X_inner_val_eff) > 0 and len(inner_present_classes) >= 2:
        eval_set_arg = [(X_inner_val_eff, y_inner_val_eff)]
        if eval_set_spy is not None:
            eval_set_spy.append(
                {
                    "stage": "inner_validation",
                    "eval_set_shapes": [
                        (arr_x.shape, arr_y.shape) for arr_x, arr_y in eval_set_arg
                    ],
                    "eval_set_X": [arr_x.copy() for arr_x, _ in eval_set_arg],
                    "inner_val_indices": inner_val_idx.copy(),
                    "split_mode": split_mode,
                }
            )
        clf_inner.fit(
            X_train=X_inner_tr_safe,
            y_train=y_inner_tr_safe,
            eval_set=eval_set_arg,
            eval_name=["inner_val"],
            eval_metric=["logloss"],
            max_epochs=int(max_epochs),
            patience=int(patience),
            batch_size=eff_b_inner,
            virtual_batch_size=eff_vb_inner,
            num_workers=0,
            drop_last=False,
        )
        best_epoch = getattr(clf_inner, "best_epoch", None)
        if best_epoch is not None and int(best_epoch) >= 0:
            selected_epochs = max(1, min(int(max_epochs), int(best_epoch) + 1))

    if not refit_full_train and len(inner_present_classes) == len(np.unique(y_train_bal)):
        preds_local = clf_inner.predict(X_test_scaled)
        probs_local = clf_inner.predict_proba(X_test_scaled)
        preds, probs = project_predictions_to_full_vocabulary(
            preds_local=preds_local,
            probs_local=probs_local,
            present_train_classes=inner_present_classes,
            n_total_classes=n_total_classes,
        )
        return (
            preds,
            probs,
            {
                "validation_split_mode": split_mode,
                "selected_epochs": int(selected_epochs),
            },
        )

    # Stage 2: Refit on full allowed balanced target training set for `selected_epochs` (no eval_set!)
    set_global_seed(seed)
    y_full_local, full_present_classes = _prepare_contiguous_labels(y_train_bal)
    X_full_safe, y_full_safe, eff_b_full, eff_vb_full = _safe_batch_params(
        X_train_bal, y_full_local
    )
    clf_final = _make_tabnet_classifier(seed)
    if eval_set_spy is not None:
        eval_set_spy.append(
            {
                "stage": "refit_full_train",
                "eval_set_shapes": [],
                "eval_set_X": [],
                "selected_epochs": int(selected_epochs),
                "split_mode": split_mode,
            }
        )
    clf_final.fit(
        X_train=X_full_safe,
        y_train=y_full_safe,
        eval_set=[],
        max_epochs=int(selected_epochs),
        patience=0,
        batch_size=eff_b_full,
        virtual_batch_size=eff_vb_full,
        num_workers=0,
        drop_last=False,
    )
    preds_local = clf_final.predict(X_test_scaled)
    probs_local = clf_final.predict_proba(X_test_scaled)

    preds, probs = project_predictions_to_full_vocabulary(
        preds_local=preds_local,
        probs_local=probs_local,
        present_train_classes=full_present_classes,
        n_total_classes=n_total_classes,
    )
    return (
        preds,
        probs,
        {
            "validation_split_mode": split_mode,
            "selected_epochs": int(selected_epochs),
        },
    )


# =============================================================================
# 7. CORE BENCHMARK RUNNER & SUMMARY GENERATION
# =============================================================================


def evaluate_single_model_on_fold(
    model_name: str,
    X_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    conds_train_raw: np.ndarray,
    X_train_bal: np.ndarray,
    y_train_bal: np.ndarray,
    X_test_scaled: np.ndarray,
    class_names: Sequence[str],
    n_total_classes: int,
    balance_strategy: str,
    seed: int,
    epochs: int = 30,
    batch_size: int = 64,
    tabnet_patience: int = 15,
    eval_set_spy: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Dispatch a single model training & evaluation on an isolated LOCO fold."""
    canonical = normalize_model_name(model_name)
    if canonical == "Random Forest":
        return train_eval_random_forest(
            X_train_bal=X_train_bal,
            y_train_bal=y_train_bal,
            X_test_scaled=X_test_scaled,
            n_total_classes=n_total_classes,
            seed=seed,
        )
    if canonical == "SVM (RBF)":
        return train_eval_svm_rbf(
            X_train_bal=X_train_bal,
            y_train_bal=y_train_bal,
            X_test_scaled=X_test_scaled,
            n_total_classes=n_total_classes,
            seed=seed,
        )
    if canonical == "XGBoost":
        return train_eval_xgboost(
            X_train_bal=X_train_bal,
            y_train_bal=y_train_bal,
            X_test_scaled=X_test_scaled,
            n_total_classes=n_total_classes,
            seed=seed,
        )
    if canonical == "MLP Scratch":
        return train_eval_mlp_scratch(
            X_train_bal=X_train_bal,
            y_train_bal=y_train_bal,
            X_test_scaled=X_test_scaled,
            n_total_classes=n_total_classes,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
        )
    if canonical == "TabNet Scratch":
        return train_eval_tabnet_scratch(
            X_train_raw=X_train_raw,
            y_train_raw=y_train_raw,
            conds_train_raw=conds_train_raw,
            X_train_bal=X_train_bal,
            y_train_bal=y_train_bal,
            X_test_scaled=X_test_scaled,
            class_names=class_names,
            n_total_classes=n_total_classes,
            balance_strategy=balance_strategy,
            seed=seed,
            max_epochs=epochs,
            patience=tabnet_patience,
            batch_size=batch_size,
            refit_full_train=True,
            eval_set_spy=eval_set_spy,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def run_dataset_loco_evaluations(
    dataset_name: str,
    X_all: np.ndarray,
    y_all_str: np.ndarray,
    conds_all: np.ndarray,
    sample_ids_all: np.ndarray,
    models: Sequence[str] = CANONICAL_MODELS,
    balance_strategies: Sequence[str] = BALANCE_STRATEGIES,
    seeds: Sequence[int] = SEEDS,
    max_folds: Optional[int] = None,
    epochs: int = 30,
    batch_size: int = 64,
    tabnet_patience: int = 15,
    results_csv_path: Optional[Path] = None,
    existing_results: Optional[List[Dict[str, Any]]] = None,
    save_predictions: bool = False,
    predictions_list: Optional[List[Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None,
    model_runner_override: Optional[Callable[..., Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]] = None,
    eval_set_spy: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Execute strict LOCO evaluation for `dataset_name` across `seeds`,
    `balance_strategies`, `folds`, and `models`.
    """
    results_accum = existing_results if existing_results is not None else []
    canonical_models = [normalize_model_name(m) for m in models]

    le = LabelEncoder()
    y_all_enc = le.fit_transform(np.asarray(y_all_str, dtype=str)).astype(np.int64)
    class_names = [str(c) for c in le.classes_]
    n_total_classes = len(class_names)

    conds_arr = np.asarray(conds_all, dtype=str)
    sample_ids_arr = np.asarray(sample_ids_all, dtype=str)
    unique_conds = sorted(np.unique(conds_arr).tolist())
    if len(unique_conds) < 2:
        raise ValueError(
            f"Dataset '{dataset_name}' must have at least 2 operating conditions for LOCO, "
            f"got {len(unique_conds)}."
        )
    if max_folds is not None:
        unique_conds = unique_conds[: int(max_folds)]

    runner_fn = model_runner_override or evaluate_single_model_on_fold

    for fold_idx, test_cond in enumerate(unique_conds):
        test_mask = conds_arr == test_cond
        train_mask = ~test_mask

        train_indices = np.where(train_mask)[0].tolist()
        test_indices = np.where(test_mask)[0].tolist()

        # Invariant 1 & 2: Verify disjoint train/test conditions and sample indices
        assert_no_condition_leakage(conds_arr.tolist(), train_indices, test_indices)

        X_train_raw = X_all[train_mask]
        y_train_raw = y_all_enc[train_mask]
        conds_train_raw = conds_arr[train_mask]
        train_sample_ids = sample_ids_arr[train_mask]

        X_test_raw = X_all[test_mask]
        y_test = y_all_enc[test_mask]
        test_sample_ids = sample_ids_arr[test_mask]

        split_signature = compute_sample_split_signature(
            train_sample_ids=train_sample_ids.tolist(),
            test_sample_ids=test_sample_ids.tolist(),
        )

        # Invariant 3: Fit StandardScaler strictly on X_train_raw
        X_train_scaled, X_test_scaled, _ = scale_loco_fold_strict(
            X_train_raw, X_test_raw
        )

        for seed in seeds:
            for balance_strategy in balance_strategies:
                t_bal_start = time.perf_counter()
                # Invariant 4: Apply SMOTE strictly to X_train_scaled
                X_train_bal, y_train_bal, smote_audit = apply_smote_with_audit(
                    X_train_scaled=X_train_scaled,
                    y_train_encoded=y_train_raw,
                    class_names=class_names,
                    strategy=balance_strategy,
                    seed=seed,
                )
                bal_elapsed = time.perf_counter() - t_bal_start

                if X_train_bal is None or y_train_bal is None:
                    if logger is not None:
                        logger.warning(
                            f"  [{dataset_name} | Fold {test_cond} | Seed {seed} | Balance {balance_strategy}] "
                            f"Fold skipped prior to model training: {smote_audit['balance_status']} - "
                            f"{smote_audit['balance_reason']}"
                        )
                    for m_name in canonical_models:
                        row = {
                            "Dataset": dataset_name,
                            "Task": TASK_NAME,
                            "Test_Condition": str(test_cond),
                            "Fold": int(fold_idx),
                            "Seed": int(seed),
                            "Model": m_name,
                            "Balance_Strategy": str(balance_strategy),
                            "Preprocessing": PREPROCESSING_VARIANT,
                            "Feature_Dimension": int(X_all.shape[1]),
                            "Accuracy": float("nan"),
                            "Balanced_Accuracy": float("nan"),
                            "Macro_F1": float("nan"),
                            "ROC_AUC": float("nan"),
                            "ROC_AUC_Valid": False,
                            "Test_Sample_Count": int(len(y_test)),
                            "Test_Class_Count": int(len(np.unique(y_test))),
                            "Train_Sample_Count": int(smote_audit["n_train_after"]),
                            "Train_Class_Count": int(len(np.unique(y_train_raw))),
                            "Status": str(smote_audit["balance_status"]),
                            "Error_Message": str(smote_audit["balance_reason"]),
                            "Runtime_Seconds": float(round(bal_elapsed, 6)),
                            "Train_Sample_Count_Before_Balance": int(smote_audit["n_train_before"]),
                            "Class_Counts_Before": json.dumps(smote_audit["counts_before"], ensure_ascii=False),
                            "Class_Counts_After": json.dumps(smote_audit["counts_after"], ensure_ascii=False),
                            "SMOTE_Params": json.dumps(smote_audit["smote_params"], ensure_ascii=False),
                            "Unbalanced_Classes": json.dumps(smote_audit["unbalanced_classes"], ensure_ascii=False),
                            "ROC_AUC_Note": "Fold failed prior to model training",
                            "Validation_Split_Mode": "none",
                            "Selected_Epochs": 0,
                            "Sample_Split_Signature": split_signature,
                        }
                        results_accum.append(row)
                    if results_csv_path is not None:
                        pd.DataFrame(results_accum).to_csv(results_csv_path, index=False)
                    continue

                # Evaluate all requested models on the exact same balanced training partition
                for m_name in canonical_models:
                    t0 = time.perf_counter()
                    try:
                        preds, probs, model_meta = runner_fn(
                            model_name=m_name,
                            X_train_raw=X_train_raw,
                            y_train_raw=y_train_raw,
                            conds_train_raw=conds_train_raw,
                            X_train_bal=X_train_bal,
                            y_train_bal=y_train_bal,
                            X_test_scaled=X_test_scaled,
                            class_names=class_names,
                            n_total_classes=n_total_classes,
                            balance_strategy=balance_strategy,
                            seed=seed,
                            epochs=epochs,
                            batch_size=batch_size,
                            tabnet_patience=tabnet_patience,
                            eval_set_spy=eval_set_spy,
                        )
                        runtime_sec = time.perf_counter() - t0
                        metrics = compute_fold_metrics(
                            y_true=y_test,
                            y_pred=preds,
                            probs=probs,
                        )

                        if logger is not None:
                            auc_fmt = (
                                f"{metrics['ROC_AUC']:.4f}"
                                if metrics["ROC_AUC_Valid"]
                                else f"NaN ({metrics['ROC_AUC_Note']})"
                            )
                            logger.info(
                                f"  [{dataset_name} | Fold {fold_idx}:{test_cond} | Seed {seed} | "
                                f"Bal={balance_strategy} | {m_name}] "
                                f"Train {smote_audit['n_train_before']}->{len(y_train_bal)} | "
                                f"Macro_F1={metrics['Macro_F1']:.4f} | Acc={metrics['Accuracy']:.4f} | "
                                f"BalAcc={metrics['Balanced_Accuracy']:.4f} | AUC={auc_fmt} "
                                f"({runtime_sec:.2f}s)"
                            )

                        row = {
                            "Dataset": dataset_name,
                            "Task": TASK_NAME,
                            "Test_Condition": str(test_cond),
                            "Fold": int(fold_idx),
                            "Seed": int(seed),
                            "Model": m_name,
                            "Balance_Strategy": str(balance_strategy),
                            "Preprocessing": PREPROCESSING_VARIANT,
                            "Feature_Dimension": int(X_all.shape[1]),
                            "Accuracy": float(metrics["Accuracy"]),
                            "Balanced_Accuracy": float(metrics["Balanced_Accuracy"]),
                            "Macro_F1": float(metrics["Macro_F1"]),
                            "ROC_AUC": float(metrics["ROC_AUC"]),
                            "ROC_AUC_Valid": bool(metrics["ROC_AUC_Valid"]),
                            "Test_Sample_Count": int(metrics["Test_Sample_Count"]),
                            "Test_Class_Count": int(metrics["Test_Class_Count"]),
                            "Train_Sample_Count": int(len(y_train_bal)),
                            "Train_Class_Count": int(len(np.unique(y_train_bal))),
                            "Status": "OK",
                            "Error_Message": "",
                            "Runtime_Seconds": float(round(runtime_sec, 6)),
                            "Train_Sample_Count_Before_Balance": int(smote_audit["n_train_before"]),
                            "Class_Counts_Before": json.dumps(smote_audit["counts_before"], ensure_ascii=False),
                            "Class_Counts_After": json.dumps(smote_audit["counts_after"], ensure_ascii=False),
                            "SMOTE_Params": json.dumps(smote_audit["smote_params"], ensure_ascii=False),
                            "Unbalanced_Classes": json.dumps(smote_audit["unbalanced_classes"], ensure_ascii=False),
                            "ROC_AUC_Note": str(metrics["ROC_AUC_Note"]),
                            "Validation_Split_Mode": str(model_meta.get("validation_split_mode", "none")),
                            "Selected_Epochs": int(model_meta.get("selected_epochs", 0)),
                            "Sample_Split_Signature": split_signature,
                        }
                        results_accum.append(row)

                        if save_predictions and predictions_list is not None:
                            for idx_s in range(len(y_test)):
                                predictions_list.append(
                                    {
                                        "Dataset": dataset_name,
                                        "Task": TASK_NAME,
                                        "Test_Condition": str(test_cond),
                                        "Fold": int(fold_idx),
                                        "Seed": int(seed),
                                        "Balance_Strategy": str(balance_strategy),
                                        "Model": m_name,
                                        "Sample_ID": str(test_sample_ids[idx_s]),
                                        "y_true": int(y_test[idx_s]),
                                        "y_true_label": str(class_names[int(y_test[idx_s])]),
                                        "y_pred": int(preds[idx_s]),
                                        "y_pred_label": str(class_names[int(preds[idx_s])]),
                                        "y_prob_json": json.dumps(
                                            [round(float(p), 6) for p in probs[idx_s]]
                                        ),
                                    }
                                )
                    except Exception as exc:
                        runtime_sec = time.perf_counter() - t0
                        if logger is not None:
                            logger.error(
                                f"  [ERROR] {dataset_name} | Fold {test_cond} | Seed {seed} | "
                                f"Balance {balance_strategy} | {m_name}: {exc}"
                            )
                        row = {
                            "Dataset": dataset_name,
                            "Task": TASK_NAME,
                            "Test_Condition": str(test_cond),
                            "Fold": int(fold_idx),
                            "Seed": int(seed),
                            "Model": m_name,
                            "Balance_Strategy": str(balance_strategy),
                            "Preprocessing": PREPROCESSING_VARIANT,
                            "Feature_Dimension": int(X_all.shape[1]),
                            "Accuracy": float("nan"),
                            "Balanced_Accuracy": float("nan"),
                            "Macro_F1": float("nan"),
                            "ROC_AUC": float("nan"),
                            "ROC_AUC_Valid": False,
                            "Test_Sample_Count": int(len(y_test)),
                            "Test_Class_Count": int(len(np.unique(y_test))),
                            "Train_Sample_Count": int(len(y_train_bal)),
                            "Train_Class_Count": int(len(np.unique(y_train_bal))),
                            "Status": "ERROR_MODEL_EXECUTION",
                            "Error_Message": f"{type(exc).__name__}: {exc}",
                            "Runtime_Seconds": float(round(runtime_sec, 6)),
                            "Train_Sample_Count_Before_Balance": int(smote_audit["n_train_before"]),
                            "Class_Counts_Before": json.dumps(smote_audit["counts_before"], ensure_ascii=False),
                            "Class_Counts_After": json.dumps(smote_audit["counts_after"], ensure_ascii=False),
                            "SMOTE_Params": json.dumps(smote_audit["smote_params"], ensure_ascii=False),
                            "Unbalanced_Classes": json.dumps(smote_audit["unbalanced_classes"], ensure_ascii=False),
                            "ROC_AUC_Note": f"Execution error: {exc}",
                            "Validation_Split_Mode": "error",
                            "Selected_Epochs": 0,
                            "Sample_Split_Signature": split_signature,
                        }
                        results_accum.append(row)

                    if results_csv_path is not None:
                        pd.DataFrame(results_accum).to_csv(results_csv_path, index=False)

    return results_accum


def generate_summaries(
    df_results: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Generate the three required summary CSVs in `output_dir`:
      1. `summary.csv` (grouped by `Dataset`, `Model`, `Balance_Strategy`, `Seed` ->
         mean/std across LOCO folds)
      2. `summary_by_dataset_model.csv` (grouped by `Dataset`, `Model`, `Balance_Strategy` ->
         overall mean/std across all valid fold-seed evaluations)
      3. `summary_across_seeds.csv` (first averages across LOCO folds per seed, then
         computes mean and std across seeds for each `Dataset`, `Model`, `Balance_Strategy`)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    summary_cols = [
        "Dataset",
        "Model",
        "Balance_Strategy",
        "Seed",
        "Preprocessing",
        "Total_Folds",
        "Valid_Folds",
        "Failed_Folds",
        "Valid_ROC_AUC_Folds",
        "Macro_F1_Mean",
        "Macro_F1_Std",
        "Accuracy_Mean",
        "Accuracy_Std",
        "Balanced_Accuracy_Mean",
        "Balanced_Accuracy_Std",
        "ROC_AUC_Mean",
        "ROC_AUC_Std",
        "Runtime_Seconds_Total",
    ]

    if df_results.empty:
        for fname in ("summary.csv", "summary_by_dataset_model.csv", "summary_across_seeds.csv"):
            p = output_dir / fname
            pd.DataFrame(columns=summary_cols).to_csv(p, index=False)
            paths[fname.replace(".csv", "")] = p
        return paths

    def _agg_metrics(g: pd.DataFrame) -> Dict[str, Any]:
        valid = g[g["Status"] == "OK"]
        valid_auc = valid[valid["ROC_AUC_Valid"] == True]  # noqa: E712

        def _m_s(series: pd.Series) -> Tuple[float, float]:
            s = series.dropna()
            if len(s) == 0:
                return float("nan"), float("nan")
            mean_val = float(s.mean())
            std_val = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            return mean_val, std_val

        f1_m, f1_s = _m_s(valid["Macro_F1"])
        acc_m, acc_s = _m_s(valid["Accuracy"])
        bacc_m, bacc_s = _m_s(valid["Balanced_Accuracy"])
        auc_m, auc_s = _m_s(valid_auc["ROC_AUC"])

        return {
            "Total_Folds": int(len(g)),
            "Valid_Folds": int(len(valid)),
            "Failed_Folds": int(len(g) - len(valid)),
            "Valid_ROC_AUC_Folds": int(len(valid_auc)),
            "Macro_F1_Mean": f1_m,
            "Macro_F1_Std": f1_s,
            "Accuracy_Mean": acc_m,
            "Accuracy_Std": acc_s,
            "Balanced_Accuracy_Mean": bacc_m,
            "Balanced_Accuracy_Std": bacc_s,
            "ROC_AUC_Mean": auc_m,
            "ROC_AUC_Std": auc_s,
            "Runtime_Seconds_Total": float(g["Runtime_Seconds"].sum()),
        }

    # 1. summary.csv (per Dataset, Model, Balance_Strategy, Seed across LOCO folds)
    rows_per_seed: List[Dict[str, Any]] = []
    for (ds, model, bal, seed), g in df_results.groupby(
        ["Dataset", "Model", "Balance_Strategy", "Seed"], sort=False
    ):
        row = {
            "Dataset": ds,
            "Model": model,
            "Balance_Strategy": bal,
            "Seed": int(seed),
            "Preprocessing": PREPROCESSING_VARIANT,
        }
        row.update(_agg_metrics(g))
        rows_per_seed.append(row)

    df_summary = pd.DataFrame(rows_per_seed)
    p_summary = output_dir / "summary.csv"
    df_summary.to_csv(p_summary, index=False)
    paths["summary"] = p_summary

    # 2. summary_by_dataset_model.csv (pooled across folds and seeds per Dataset, Model, Balance_Strategy)
    rows_by_ds_model: List[Dict[str, Any]] = []
    for (ds, model, bal), g in df_results.groupby(
        ["Dataset", "Model", "Balance_Strategy"], sort=False
    ):
        row = {
            "Dataset": ds,
            "Model": model,
            "Balance_Strategy": bal,
            "Preprocessing": PREPROCESSING_VARIANT,
            "N_Seeds": int(g["Seed"].nunique()),
        }
        row.update(_agg_metrics(g))
        rows_by_ds_model.append(row)

    df_by_ds_model = pd.DataFrame(rows_by_ds_model).sort_values(
        by=["Dataset", "Balance_Strategy", "Macro_F1_Mean"],
        ascending=[True, True, False],
    )
    p_by_ds_model = output_dir / "summary_by_dataset_model.csv"
    df_by_ds_model.to_csv(p_by_ds_model, index=False)
    paths["summary_by_dataset_model"] = p_by_ds_model

    # 3. summary_across_seeds.csv (mean & std of seed-level LOCO averages)
    rows_across_seeds: List[Dict[str, Any]] = []
    for (ds, model, bal), g_seed in df_summary.groupby(
        ["Dataset", "Model", "Balance_Strategy"], sort=False
    ):
        def _seed_ms(col: str) -> Tuple[float, float]:
            s = g_seed[col].dropna()
            if len(s) == 0:
                return float("nan"), float("nan")
            return float(s.mean()), (float(s.std(ddof=1)) if len(s) > 1 else 0.0)

        f1_m, f1_s = _seed_ms("Macro_F1_Mean")
        acc_m, acc_s = _seed_ms("Accuracy_Mean")
        bacc_m, bacc_s = _seed_ms("Balanced_Accuracy_Mean")
        auc_m, auc_s = _seed_ms("ROC_AUC_Mean")

        rows_across_seeds.append(
            {
                "Dataset": ds,
                "Model": model,
                "Balance_Strategy": bal,
                "Preprocessing": PREPROCESSING_VARIANT,
                "N_Seeds": int(len(g_seed)),
                "Folds_Per_Seed": int(g_seed["Total_Folds"].iloc[0]) if len(g_seed) > 0 else 0,
                "Total_Evaluations": int(g_seed["Total_Folds"].sum()),
                "Valid_Evaluations": int(g_seed["Valid_Folds"].sum()),
                "Failed_Evaluations": int(g_seed["Failed_Folds"].sum()),
                "Macro_F1_Mean": f1_m,
                "Macro_F1_Std_Across_Seeds": f1_s,
                "Accuracy_Mean": acc_m,
                "Accuracy_Std_Across_Seeds": acc_s,
                "Balanced_Accuracy_Mean": bacc_m,
                "Balanced_Accuracy_Std_Across_Seeds": bacc_s,
                "ROC_AUC_Mean": auc_m,
                "ROC_AUC_Std_Across_Seeds": auc_s,
                "Runtime_Seconds_Total": float(g_seed["Runtime_Seconds_Total"].sum()),
            }
        )

    df_across_seeds = pd.DataFrame(rows_across_seeds).sort_values(
        by=["Dataset", "Balance_Strategy", "Macro_F1_Mean"],
        ascending=[True, True, False],
    )
    p_across_seeds = output_dir / "summary_across_seeds.csv"
    df_across_seeds.to_csv(p_across_seeds, index=False)
    paths["summary_across_seeds"] = p_across_seeds

    return paths


# =============================================================================
# 8. CLI ENTRY POINT
# =============================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the leakage-free, paired Leave-One-Condition-Out (LOCO) "
            "supervised baseline benchmark (baseline_loco_v2)."
        )
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(TARGET_DATASETS),
        choices=TARGET_DATASETS,
        help="Target datasets to evaluate under LOCO.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(CANONICAL_MODELS),
        help=(
            "Models to evaluate: 'Random Forest', 'SVM (RBF)', 'XGBoost', "
            "'MLP Scratch', 'TabNet Scratch' (aliases RF, SVM, XGB, MLP, TabNet also accepted)."
        ),
    )
    parser.add_argument(
        "--balance-strategies",
        nargs="+",
        default=list(BALANCE_STRATEGIES),
        choices=BALANCE_STRATEGIES,
        help="Training class balance strategies ('none', 'smote').",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS),
        help="Random seeds for full reproducibility (default: 42 123 2026).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs for MLP Scratch and max_epochs for TabNet Scratch (default: 30).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for MLP Scratch and TabNet Scratch (default: 64).",
    )
    parser.add_argument(
        "--tabnet-patience",
        type=int,
        default=15,
        help="Early-stopping patience for TabNet Scratch on internal training-only validation (default: 15).",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional cap on the number of LOCO folds per dataset (for smoke testing).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_PROCESSED_TL_DIR,
        help=f"Path to data/processed_tl/{PREPROCESSING_VARIANT} directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Base output directory (default: results/{EXPERIMENT_ID}).",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save fold-level test predictions and probabilities to predictions.csv for paired statistical tests.",
    )
    return parser.parse_args(argv)


def run_benchmark(
    args: argparse.Namespace,
    feature_extractor: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    enforce_expected_folds: bool = True,
) -> Path:
    """
    Execute the `baseline_loco_v2` benchmark and write all required artifacts to
    `<output_root>/run_YYYYMMDD_HHMMSS/`.
    """
    verified_folds = verify_canonical_fold_metadata(args.targets)
    canonical_models = [normalize_model_name(m) for m in args.models]
    data_root = validate_data_root_is_detrended_rms(args.data_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    logger = logging.getLogger(f"{EXPERIMENT_ID}_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("=" * 80)
    logger.info(f"STARTING BENCHMARK: {EXPERIMENT_ID} (task={TASK_NAME})")
    logger.info(f"Data root         : {data_root}")
    logger.info(f"Output directory  : {run_dir}")
    logger.info(f"Targets           : {args.targets} (expected folds: {verified_folds})")
    logger.info(f"Models            : {canonical_models}")
    logger.info(f"Balance strategies: {args.balance_strategies}")
    logger.info(f"Seeds             : {args.seeds}")
    logger.info(f"Max folds cap     : {args.max_folds}")
    logger.info("=" * 80)

    config_payload: Dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "task": TASK_NAME,
        "timestamp": timestamp,
        "preprocessing": PREPROCESSING_VARIANT,
        "feature_dimension": FEATURE_DIMENSION,
        "data_root": str(data_root),
        "output_dir": str(run_dir),
        "targets": list(args.targets),
        "expected_loco_folds": verified_folds,
        "models": canonical_models,
        "balance_strategies": list(args.balance_strategies),
        "seeds": [int(s) for s in args.seeds],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "tabnet_patience": int(args.tabnet_patience),
        "max_folds": args.max_folds,
        "save_predictions": bool(args.save_predictions),
        "leakage_prevention_contract": {
            "loco_isolation": "1 operating condition held out as test; K-1 conditions used for training",
            "scaler_fit_scope": "target_training_fold_only",
            "smote_scope": "scaled_target_training_fold_only",
            "tabnet_early_stopping_scope": "internal_validation_within_target_training_conditions_only",
            "tabnet_refit_strategy": "refit_full_train_with_selected_epoch",
            "source_data_used": False,
            "pretrained_weights_used": False,
        },
        "model_hyperparameters": MODEL_HYPERPARAMETERS_DOC,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    results_list: List[Dict[str, Any]] = []
    predictions_list: List[Dict[str, Any]] = []
    dataset_audits: List[Dict[str, Any]] = []
    results_csv_path = run_dir / "results.csv"

    for ds_name in args.targets:
        X_all, y_all_str, conds_all, sample_ids_all, ds_audit = load_and_extract_target_dataset(
            dataset_name=ds_name,
            data_root=data_root,
            feature_extractor=feature_extractor,
            logger=logger,
            enforce_expected_folds=enforce_expected_folds,
        )
        dataset_audits.append(ds_audit)

        run_dataset_loco_evaluations(
            dataset_name=ds_name,
            X_all=X_all,
            y_all_str=y_all_str,
            conds_all=conds_all,
            sample_ids_all=sample_ids_all,
            models=canonical_models,
            balance_strategies=args.balance_strategies,
            seeds=args.seeds,
            max_folds=args.max_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            tabnet_patience=args.tabnet_patience,
            results_csv_path=results_csv_path,
            existing_results=results_list,
            save_predictions=args.save_predictions,
            predictions_list=predictions_list,
            logger=logger,
        )

    df_final = pd.DataFrame(results_list)
    if df_final.empty:
        df_final = pd.DataFrame(columns=REQUIRED_RESULTS_COLUMNS + AUDIT_EXTRA_COLUMNS)
    df_final.to_csv(results_csv_path, index=False)

    if dataset_audits:
        pd.DataFrame(dataset_audits).to_csv(run_dir / "dataset_audit.csv", index=False)

    if args.save_predictions and predictions_list:
        pd.DataFrame(predictions_list).to_csv(run_dir / "predictions.csv", index=False)

    summary_paths = generate_summaries(df_final, run_dir)

    status_counts = (
        df_final["Status"].value_counts().to_dict() if not df_final.empty else {}
    )
    valid_count = int((df_final["Status"] == "OK").sum()) if not df_final.empty else 0
    failed_count = int(len(df_final) - valid_count)
    error_breakdown = {
        str(k): int(v) for k, v in status_counts.items() if str(k) != "OK"
    }

    fold_accounting = {
        "expected_evaluations": int(len(df_final)),
        "valid_evaluations": valid_count,
        "failed_evaluations": failed_count,
        "status_breakdown": {str(k): int(v) for k, v in status_counts.items()},
        "error_breakdown": error_breakdown,
    }

    provenance = collect_run_provenance(
        project_root=PROJECT_ROOT,
        script_paths=[
            Path(__file__),
            PROJECT_ROOT / "src" / "features" / "extractors_v2.py",
            PROJECT_ROOT / "src" / "features" / "signalai_wrapper.py",
            PROJECT_ROOT / "src" / "models" / "build_tabnet_resnet_balance.py",
            PROJECT_ROOT / "src" / "vibnet" / "data" / "registry.py",
        ],
        seeds=args.seeds,
        extra_config=config_payload,
        fold_accounting=fold_accounting,
    )
    write_run_metadata(run_dir, provenance, filename="run_metadata.json")

    logger.info("=" * 80)
    logger.info(f"BENCHMARK COMPLETED: {EXPERIMENT_ID}")
    logger.info(f"Detailed results   : {results_csv_path}")
    for s_name, s_path in summary_paths.items():
        logger.info(f"Summary [{s_name}] : {s_path}")
    logger.info(f"Run metadata       : {run_dir / 'run_metadata.json'}")
    logger.info("=" * 80)
    return run_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
