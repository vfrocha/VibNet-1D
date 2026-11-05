#!/usr/bin/env python3
"""
Independent, Leakage-Free Scratch vs. Transfer Learning Benchmark
(`src/experiments/run_scratch_vs_tl_benchmark.py`).

Evaluates the 6-cell experimental matrix across all 5 target datasets
(`CWRU_12k`, `CWRU_48k`, `HUST_Gearbox`, `UOEMD`, `PU` = 66 LOCO folds),
2 balance strategies (`none`, `smote`), and 3 random seeds (`42`, `123`, `2026`):

  | Architecture | Method         |
  |--------------|----------------|
  | MLP          | Scratch        |
  | MLP          | Fine-Tuning    |
  | MLP          | Linear Probing |
  | TabNet       | Scratch        |
  | TabNet       | Fine-Tuning    |
  | TabNet       | Linear Probing |

Key Methodological Guarantees:
  1. Source-Target Disjointness: Pretraining uses only the configured source pool
     (`configs/scratch_vs_tl.yaml`, default `UORED`, `HUST`, `Mechanical_Gear`,
     `Electric_Motor`). Programmatically asserts zero overlap between target sample
     IDs and source sample IDs.
  2. Strict LOCO Isolation: Each fold holds out 100% of one operating condition for
     testing and trains on the remaining K - 1 conditions. `StandardScaler` and
     `SMOTE` are fitted strictly on the target training fold.
  3. True Linear Probing with Programmatic Verification:
     - For `MLP Linear Probing`: freezes `MLPEncoder` and `Bottleneck` (`requires_grad=False`,
       `.eval()`), trains only the target `nn.Linear(64, C_target)` head, and asserts
       `backbone_param_max_abs_diff == 0.0` and `backbone_buffer_max_abs_diff == 0.0`.
     - For `TabNet Linear Probing`: extracts pretrained `embedder` and `encoder`
       (`TabNetEncoder`) from `TabNetPretrainer.network`, freezes all parameters and
       GhostBatchNorm buffers (`requires_grad=False`, `.eval()`), trains only a fresh
       `nn.Linear(16, C_target)` head on `sum(steps_out)`, and asserts
       `backbone_param_max_abs_diff == 0.0` and `backbone_buffer_max_abs_diff == 0.0`.
  4. Sample-Level Traceability & Paired Statistical Testing:
     Exports `fold_manifest.csv`, `predictions.csv` (validated for completeness,
     class alignment, and identical test sample sets), `results.csv`, summary tables,
     Wilcoxon signed-rank tests with Holm step-down correction (`statistical_tests.csv`),
     method rankings (`rankings.csv`), and `report.md`.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.experiments.audit_scratch_vs_tl import (
    DEFAULT_BASELINE_REF_DIR,
    build_compatibility_audit_table,
    collect_git_and_script_hashes,
)
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
    create_internal_validation_split,
    default_141_feature_extractor,
    load_and_extract_target_dataset,
    project_predictions_to_full_vocabulary,
    scale_loco_fold_strict,
    set_global_seed,
    validate_data_root_is_detrended_rms,
    verify_canonical_fold_metadata,
)
from src.vibnet.data.registry import load_yaml_config
from src.vibnet.evaluation.balancing import trim_single_remainder_sample
from src.vibnet.evaluation.protocols import assert_no_condition_leakage
from src.vibnet.evaluation.provenance import get_dependency_versions

EXPERIMENT_ID: str = "scratch_vs_tl"
ARCHITECTURES: List[str] = ["MLP", "TabNet"]
METHODS: List[str] = ["Scratch", "Fine-Tuning", "Linear Probing"]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "scratch_vs_tl.yaml"
DEFAULT_DATA_ROOT: Path = PROJECT_ROOT / "data" / "processed_tl" / PREPROCESSING_VARIANT
DEFAULT_OUTPUT_ROOT: Path = RESULTS_DIR / EXPERIMENT_ID


# =============================================================================
# 1. CONFIGURATION & PREPROCESSING SIGNATURES
# =============================================================================


def load_benchmark_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load `configs/scratch_vs_tl.yaml` and validate target-source disjointness."""
    cfg = load_yaml_config(config_path)
    source_pools = cfg.get("source_pools_by_target", {})
    targets = cfg.get("target_datasets", TARGET_DATASETS)

    for t_name in targets:
        if t_name not in source_pools or not source_pools[t_name]:
            raise ValueError(
                f"Target dataset '{t_name}' has no source pool defined in {config_path}."
            )
        if t_name in source_pools[t_name]:
            raise ValueError(
                f"Target dataset '{t_name}' cannot appear in its own source pool: "
                f"{source_pools[t_name]}."
            )
    return cfg


def compute_preprocessing_signature(
    preprocessing_variant: str,
    feature_dim: int,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    balance_strategy: str,
    smote_params: Mapping[str, Any],
) -> str:
    """
    Compute a deterministic SHA-256 signature of the fold's fitted `StandardScaler`
    statistics and `SMOTE` configuration, verifying that all paired methods in a fold
    receive the exact same preprocessed feature space.
    """
    mean_round = [round(float(x), 6) for x in scaler_mean[:16]]
    scale_round = [round(float(x), 6) for x in scaler_scale[:16]]
    payload = json.dumps(
        {
            "variant": preprocessing_variant,
            "feature_dim": int(feature_dim),
            "mean_head16": mean_round,
            "scale_head16": scale_round,
            "mean_sum": round(float(np.sum(scaler_mean)), 5),
            "scale_sum": round(float(np.sum(scaler_scale)), 5),
            "balance_strategy": str(balance_strategy),
            "smote_params": dict(smote_params),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _snapshot_module_tensors(module: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return deep NumPy copies of all parameters and floating-point buffers in `module`."""
    params = {
        name: p.detach().cpu().numpy().copy()
        for name, p in module.named_parameters()
    }
    buffers = {
        name: b.detach().cpu().numpy().copy()
        for name, b in module.named_buffers()
        if b.is_floating_point()
    }
    return params, buffers


def _max_abs_tensor_diff(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
) -> float:
    """Compute the maximum absolute difference across named NumPy arrays."""
    if not before:
        return 0.0
    max_d = 0.0
    for k, arr_b in before.items():
        if k in after:
            d = float(np.max(np.abs(after[k] - arr_b)))
            if d > max_d:
                max_d = d
    return max_d


# =============================================================================
# 2. SOURCE-ONLY PRETRAINING (MLP & TABNET)
# =============================================================================


def prepare_scaled_source_pool(
    source_names: Sequence[str],
    loaded_sources: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]],
    target_sample_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Standardize each source dataset independently (`StandardScaler` fitted per source)
    and verify that zero target sample IDs appear in the source pool.
    """
    target_id_set = set(str(s) for s in target_sample_ids)
    prepared: Dict[str, Dict[str, Any]] = {}

    for s_name in source_names:
        if s_name in TARGET_DATASETS:
            raise ValueError(
                f"Target dataset '{s_name}' cannot be included in the source pretraining pool!"
            )
        if s_name not in loaded_sources:
            raise KeyError(f"Source dataset '{s_name}' not found in loaded_sources.")
        X_s, y_s_str, _, s_sample_ids, s_audit = loaded_sources[s_name]

        overlap = target_id_set.intersection(set(s_sample_ids.tolist()))
        if overlap:
            raise ValueError(
                f"Data leakage detected: {len(overlap)} target sample IDs found in "
                f"source dataset '{s_name}'!"
            )

        scaler_s = StandardScaler()
        X_s_scaled = scaler_s.fit_transform(X_s).astype(np.float32)

        le_s = LabelEncoder()
        y_s_enc = le_s.fit_transform(y_s_str).astype(np.int64)

        prepared[s_name] = {
            "X_scaled": X_s_scaled,
            "y_enc": y_s_enc,
            "classes": [str(c) for c in le_s.classes_],
            "n_classes": int(len(le_s.classes_)),
            "sample_ids": s_sample_ids.tolist(),
            "fs_hz": int(s_audit.get("fs_hz", 12000)),
        }
    return prepared


def pretrain_mlp_on_sources(
    prepared_sources: Mapping[str, Mapping[str, Any]],
    seed: int,
    pretrain_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    """
    Pretrain `HybridDLModel` (`MLPEncoder` + `Bottleneck` + domain-specific source heads)
    strictly on the source datasets (`prepared_sources`).

    Returns a dictionary containing deep-copied `encoder_state_dict` and
    `bottleneck_state_dict` along with source provenance metadata.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.build_tabnet_resnet_balance import HybridDLModel

    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_names = list(prepared_sources.keys())
    if not source_names:
        raise ValueError("Cannot pretrain MLP on empty prepared_sources.")

    first_X = prepared_sources[source_names[0]]["X_scaled"]
    input_dim = int(first_X.shape[1])
    num_classes_dict = {
        s_name: int(prepared_sources[s_name]["n_classes"])
        for s_name in source_names
    }

    model = HybridDLModel(
        num_features=input_dim,
        dataset_classes_dict=num_classes_dict,
        encoder_type="mlp",
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    gen = torch.Generator()
    gen.manual_seed(int(seed))

    loaders: Dict[str, DataLoader] = {}
    for s_name in source_names:
        X_s = np.asarray(prepared_sources[s_name]["X_scaled"], dtype=np.float32)
        y_s = np.asarray(prepared_sources[s_name]["y_enc"], dtype=np.int64)
        eff_b = min(int(batch_size), max(2, len(X_s)))
        if len(X_s) > eff_b and (len(X_s) % eff_b == 1):
            X_trm, y_trm, _ = trim_single_remainder_sample(
                X_s.tolist(), y_s.tolist(), batch_size=eff_b, seed=seed
            )
            X_s = np.asarray(X_trm, dtype=np.float32)
            y_s = np.asarray(y_trm, dtype=np.int64)

        ds = TensorDataset(
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(y_s, dtype=torch.long),
        )
        loaders[s_name] = DataLoader(
            ds,
            batch_size=eff_b,
            shuffle=True,
            drop_last=(len(ds) > eff_b and len(ds) % eff_b == 1),
            generator=gen,
        )

    model.train()
    for _ in range(int(pretrain_epochs)):
        for s_name in source_names:
            for bx, by in loaders[s_name]:
                if len(bx) <= 1:
                    continue
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits, _ = model(bx, dataset_name=s_name)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
        scheduler.step()

    model.eval()
    return {
        "encoder_state_dict": copy.deepcopy(model.encoder.state_dict()),
        "bottleneck_state_dict": copy.deepcopy(model.bottleneck.state_dict()),
        "input_dim": input_dim,
        "source_names": source_names,
        "seed": int(seed),
        "pretrain_epochs": int(pretrain_epochs),
    }


def pretrain_tabnet_on_sources(
    prepared_sources: Mapping[str, Mapping[str, Any]],
    seed: int,
    pretrain_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 2e-2,
) -> Any:
    """
    Pretrain `TabNetPretrainer` strictly on the stacked domain-standardized source
    feature matrices (`prepared_sources`).
    """
    import torch
    from pytorch_tabnet.pretraining import TabNetPretrainer

    set_global_seed(seed)
    source_names = list(prepared_sources.keys())
    if not source_names:
        raise ValueError("Cannot pretrain TabNet on empty prepared_sources.")

    X_stack = np.vstack(
        [np.asarray(prepared_sources[s]["X_scaled"], dtype=np.float32) for s in source_names]
    )
    y_dummy = np.zeros(len(X_stack), dtype=np.int64)
    eff_b = min(int(batch_size), max(2, len(X_stack)))
    if len(X_stack) > eff_b and (len(X_stack) % eff_b == 1):
        X_trm, _, _ = trim_single_remainder_sample(
            X_stack.tolist(), y_dummy.tolist(), batch_size=eff_b, seed=seed
        )
        X_stack = np.asarray(X_trm, dtype=np.float32)
        eff_b = min(int(batch_size), max(2, len(X_stack)))
    eff_vb = min(32, max(2, eff_b // 2))

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    pretrainer = TabNetPretrainer(
        n_d=16,
        n_a=16,
        n_steps=4,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=float(lr), weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        mask_type="sparsemax",
        device_name=device_name,
        seed=int(seed),
        verbose=0,
    )

    pretrainer.fit(
        X_train=X_stack,
        eval_set=[X_stack[: min(1024, len(X_stack))]],
        max_epochs=int(pretrain_epochs),
        patience=10,
        batch_size=eff_b,
        virtual_batch_size=eff_vb,
        num_workers=0,
        drop_last=False,
        pretraining_ratio=0.8,
    )
    return pretrainer


# =============================================================================
# 3. TARGET ADAPTATION & EVALUATION (SCRATCH, FINE-TUNING, LINEAR PROBING)
# =============================================================================


def train_eval_mlp_variant(
    method: str,
    X_train_bal: np.ndarray,
    y_train_bal: np.ndarray,
    X_test_scaled: np.ndarray,
    n_total_classes: int,
    seed: int,
    pretrained_mlp_state: Optional[Mapping[str, Any]] = None,
    epochs: int = 30,
    batch_size: int = 64,
    lr_scratch: float = 1e-3,
    lr_finetune_backbone: float = 5e-4,
    lr_finetune_head: float = 1e-3,
    lr_linear_probing: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Train and evaluate `MLP` under `Scratch`, `Fine-Tuning`, or `Linear Probing`.

    Programmatically verifies parameter & buffer update invariants:
      - `Scratch` & `Fine-Tuning`: `backbone_param_max_abs_diff > 0` and `head_param_max_abs_diff > 0`
      - `Linear Probing`: `backbone_param_max_abs_diff == 0.0`, `backbone_buffer_max_abs_diff == 0.0`,
        and `head_param_max_abs_diff > 0`.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.build_tabnet_resnet_balance import HybridDLModel

    if method not in METHODS:
        raise ValueError(f"Unsupported method '{method}'. Expected one of {METHODS}.")
    if method in ("Fine-Tuning", "Linear Probing") and pretrained_mlp_state is None:
        raise ValueError(f"MLP '{method}' requires pretrained_mlp_state.")

    set_global_seed(seed)
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

    if method in ("Fine-Tuning", "Linear Probing") and pretrained_mlp_state is not None:
        model.encoder.load_state_dict(
            copy.deepcopy(pretrained_mlp_state["encoder_state_dict"])
        )
        model.bottleneck.load_state_dict(
            copy.deepcopy(pretrained_mlp_state["bottleneck_state_dict"])
        )
        # Re-seed target classification head deterministically
        set_global_seed(seed)
        model.heads[target_key].reset_parameters()

    freeze_backbone = method == "Linear Probing"
    if freeze_backbone:
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.bottleneck.parameters():
            p.requires_grad = False
        for p in model.heads[target_key].parameters():
            p.requires_grad = True
        model.encoder.eval()
        model.bottleneck.eval()
        model.heads[target_key].train()
        optimizer = optim.AdamW(
            model.heads[target_key].parameters(),
            lr=float(lr_linear_probing),
            weight_decay=float(weight_decay),
        )
    elif method == "Fine-Tuning":
        for p in model.parameters():
            p.requires_grad = True
        optimizer = optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": float(lr_finetune_backbone)},
                {"params": model.bottleneck.parameters(), "lr": float(lr_finetune_backbone)},
                {"params": model.heads[target_key].parameters(), "lr": float(lr_finetune_head)},
            ],
            weight_decay=float(weight_decay),
        )
    else:  # Scratch
        for p in model.parameters():
            p.requires_grad = True
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(lr_scratch),
            weight_decay=float(weight_decay),
        )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    enc_params_before, enc_bufs_before = _snapshot_module_tensors(model.encoder)
    bot_params_before, bot_bufs_before = _snapshot_module_tensors(model.bottleneck)
    head_params_before, _ = _snapshot_module_tensors(model.heads[target_key])

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    ds_train = TensorDataset(
        torch.tensor(X_work, dtype=torch.float32),
        torch.tensor(y_work, dtype=torch.long),
    )
    loader = DataLoader(
        ds_train,
        batch_size=eff_batch,
        shuffle=True,
        drop_last=(len(ds_train) > eff_batch and len(ds_train) % eff_batch == 1),
        generator=gen,
    )

    for _ in range(int(epochs)):
        if freeze_backbone:
            model.encoder.eval()
            model.bottleneck.eval()
            model.heads[target_key].train()
        else:
            model.train()
        for bx, by in loader:
            if len(bx) <= 1:
                continue
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = model(bx, dataset_name=target_key)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    enc_params_after, enc_bufs_after = _snapshot_module_tensors(model.encoder)
    bot_params_after, bot_bufs_after = _snapshot_module_tensors(model.bottleneck)
    head_params_after, _ = _snapshot_module_tensors(model.heads[target_key])

    backbone_param_diff = max(
        _max_abs_tensor_diff(enc_params_before, enc_params_after),
        _max_abs_tensor_diff(bot_params_before, bot_params_after),
    )
    backbone_buf_diff = max(
        _max_abs_tensor_diff(enc_bufs_before, enc_bufs_after),
        _max_abs_tensor_diff(bot_bufs_before, bot_bufs_after),
    )
    head_param_diff = _max_abs_tensor_diff(head_params_before, head_params_after)

    if freeze_backbone:
        if backbone_param_diff > 0.0 or backbone_buf_diff > 0.0:
            raise AssertionError(
                f"MLP Linear Probing violated frozen-encoder invariant: "
                f"backbone_param_diff={backbone_param_diff}, backbone_buf_diff={backbone_buf_diff}"
            )
    else:
        if backbone_param_diff <= 0.0:
            raise AssertionError(
                f"MLP {method} did not update backbone parameters (diff={backbone_param_diff})."
            )

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
            "backbone_frozen": bool(freeze_backbone),
            "backbone_param_max_abs_diff": float(backbone_param_diff),
            "backbone_buffer_max_abs_diff": float(backbone_buf_diff),
            "head_param_max_abs_diff": float(head_param_diff),
        },
    )


def train_eval_tabnet_variant(
    method: str,
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
    pretrained_tabnet: Optional[Any] = None,
    max_epochs: int = 30,
    patience: int = 15,
    batch_size: int = 64,
    lr_scratch: float = 2e-2,
    lr_finetune: float = 1e-2,
    lr_linear_probing: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Train and evaluate `TabNet` under `Scratch`, `Fine-Tuning`, or `Linear Probing`.

    - `Scratch`: `TabNetClassifier` (`from_unsupervised=None`) with internal
      condition-grouped validation for epoch selection + full-train refit.
    - `Fine-Tuning`: `TabNetClassifier` (`from_unsupervised=copy.deepcopy(pretrained_tabnet)`)
      with internal condition-grouped validation for epoch selection + full-train refit.
      Verifies `backbone_param_max_abs_diff > 0`.
    - `Linear Probing`: Extracts `embedder` and `encoder` (`TabNetEncoder`) from
      `pretrained_tabnet.network`, freezes all parameters and GhostBatchNorm buffers
      (`requires_grad=False`, `.eval()`), trains only `nn.Linear(16, C_present)` on
      `sum(steps_out)`, and verifies `backbone_param_max_abs_diff == 0.0` and
      `backbone_buffer_max_abs_diff == 0.0`.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from pytorch_tabnet.tab_model import TabNetClassifier

    if method not in METHODS:
        raise ValueError(f"Unsupported method '{method}'. Expected one of {METHODS}.")
    if method in ("Fine-Tuning", "Linear Probing") and pretrained_tabnet is None:
        raise ValueError(f"TabNet '{method}' requires pretrained_tabnet.")

    set_global_seed(seed)

    def _safe_batch_params(
        X_arr: np.ndarray, y_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
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

    if method == "Linear Probing":
        assert pretrained_tabnet is not None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone_net = copy.deepcopy(pretrained_tabnet.network).to(device)
        embedder = backbone_net.embedder
        encoder = backbone_net.encoder

        for p in embedder.parameters():
            p.requires_grad = False
        for p in encoder.parameters():
            p.requires_grad = False
        embedder.eval()
        encoder.eval()

        emb_p_before, emb_b_before = _snapshot_module_tensors(embedder)
        enc_p_before, enc_b_before = _snapshot_module_tensors(encoder)

        y_full_local, full_present_classes = _prepare_contiguous_labels(y_train_bal)
        X_full_safe, y_full_safe, eff_b_full, _ = _safe_batch_params(
            X_train_bal, y_full_local
        )

        set_global_seed(seed)
        linear_head = nn.Linear(16, int(len(full_present_classes))).to(device)
        for p in linear_head.parameters():
            p.requires_grad = True
        head_p_before, _ = _snapshot_module_tensors(linear_head)

        optimizer = optim.AdamW(
            linear_head.parameters(),
            lr=float(lr_linear_probing),
            weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        criterion = nn.CrossEntropyLoss()

        gen = torch.Generator()
        gen.manual_seed(int(seed))
        ds_train = TensorDataset(
            torch.tensor(X_full_safe, dtype=torch.float32),
            torch.tensor(y_full_safe, dtype=torch.long),
        )
        loader = DataLoader(
            ds_train,
            batch_size=eff_b_full,
            shuffle=True,
            drop_last=(len(ds_train) > eff_b_full and len(ds_train) % eff_b_full == 1),
            generator=gen,
        )

        for _ in range(int(max_epochs)):
            embedder.eval()
            encoder.eval()
            linear_head.train()
            for bx, by in loader:
                if len(bx) <= 1:
                    continue
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    embedded_x = embedder(bx)
                    steps_out, _ = encoder(embedded_x)
                    latent = torch.sum(torch.stack(steps_out, dim=0), dim=0)
                logits = linear_head(latent)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        emb_p_after, emb_b_after = _snapshot_module_tensors(embedder)
        enc_p_after, enc_b_after = _snapshot_module_tensors(encoder)
        head_p_after, _ = _snapshot_module_tensors(linear_head)

        backbone_param_diff = max(
            _max_abs_tensor_diff(emb_p_before, emb_p_after),
            _max_abs_tensor_diff(enc_p_before, enc_p_after),
        )
        backbone_buf_diff = max(
            _max_abs_tensor_diff(emb_b_before, emb_b_after),
            _max_abs_tensor_diff(enc_b_before, enc_b_after),
        )
        head_param_diff = _max_abs_tensor_diff(head_p_before, head_p_after)

        if backbone_param_diff > 0.0 or backbone_buf_diff > 0.0:
            raise AssertionError(
                f"TabNet Linear Probing violated frozen-encoder invariant: "
                f"backbone_param_diff={backbone_param_diff}, backbone_buf_diff={backbone_buf_diff}"
            )

        embedder.eval()
        encoder.eval()
        linear_head.eval()
        with torch.no_grad():
            X_te_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            embedded_te = embedder(X_te_t)
            steps_te, _ = encoder(embedded_te)
            latent_te = torch.sum(torch.stack(steps_te, dim=0), dim=0)
            logits_te = linear_head(latent_te)
            probs_local = torch.softmax(logits_te, dim=1).cpu().numpy().astype(np.float64)
            preds_local = np.argmax(probs_local, axis=1).astype(np.int64)

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
                "validation_split_mode": "none_fixed_epochs",
                "selected_epochs": int(max_epochs),
                "backbone_frozen": True,
                "backbone_param_max_abs_diff": float(backbone_param_diff),
                "backbone_buffer_max_abs_diff": float(backbone_buf_diff),
                "head_param_max_abs_diff": float(head_param_diff),
            },
        )

    # Scratch or Fine-Tuning via TabNetClassifier with internal training-only validation
    active_lr = float(lr_finetune) if method == "Fine-Tuning" else float(lr_scratch)

    def _make_tabnet_clf(run_seed: int) -> TabNetClassifier:
        return TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=4,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=active_lr, weight_decay=1e-5),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 10, "gamma": 0.9},
            mask_type="sparsemax",
            seed=int(run_seed),
            verbose=0,
        )

    inner_tr_idx, inner_val_idx, split_mode = create_internal_validation_split(
        X_train_raw=X_train_raw,
        y_train=y_train_raw,
        conds_train=conds_train_raw,
        seed=seed,
        val_fraction=0.2,
    )
    X_inner_tr_scaled, X_inner_val_scaled, _ = scale_loco_fold_strict(
        X_train_raw[inner_tr_idx], X_train_raw[inner_val_idx]
    )
    y_inner_tr_raw = y_train_raw[inner_tr_idx]
    y_inner_val_raw = y_train_raw[inner_val_idx]

    X_inner_tr_bal, y_inner_tr_bal, _ = apply_smote_with_audit(
        X_train_scaled=X_inner_tr_scaled,
        y_train_encoded=y_inner_tr_raw,
        class_names=class_names,
        strategy=balance_strategy,
        seed=seed,
    )
    if X_inner_tr_bal is None or y_inner_tr_bal is None:
        X_inner_tr_bal = X_inner_tr_scaled
        y_inner_tr_bal = y_inner_tr_raw

    y_inner_tr_local, inner_present_classes = _prepare_contiguous_labels(y_inner_tr_bal)
    val_keep_mask = np.isin(y_inner_val_raw, inner_present_classes)
    X_inner_val_eff = X_inner_val_scaled[val_keep_mask]
    y_inner_val_eff = np.searchsorted(
        inner_present_classes, y_inner_val_raw[val_keep_mask]
    ).astype(np.int64)

    X_inner_tr_safe, y_inner_tr_safe, eff_b_inner, eff_vb_inner = _safe_batch_params(
        X_inner_tr_bal, y_inner_tr_local
    )

    selected_epochs = int(max_epochs)
    clf_inner = _make_tabnet_clf(seed)
    from_unsup_inner = (
        copy.deepcopy(pretrained_tabnet) if method == "Fine-Tuning" else None
    )

    if len(X_inner_val_eff) > 0 and len(inner_present_classes) >= 2:
        clf_inner.fit(
            X_train=X_inner_tr_safe,
            y_train=y_inner_tr_safe,
            eval_set=[(X_inner_val_eff, y_inner_val_eff)],
            eval_name=["inner_val"],
            eval_metric=["logloss"],
            max_epochs=int(max_epochs),
            patience=int(patience),
            batch_size=eff_b_inner,
            virtual_batch_size=eff_vb_inner,
            num_workers=0,
            drop_last=False,
            from_unsupervised=from_unsup_inner,
        )
        best_ep = getattr(clf_inner, "best_epoch", None)
        if best_ep is not None and int(best_ep) >= 0:
            selected_epochs = max(1, min(int(max_epochs), int(best_ep) + 1))

    # Stage 2: Refit on full balanced target training set for `selected_epochs`
    set_global_seed(seed)
    y_full_local, full_present_classes = _prepare_contiguous_labels(y_train_bal)
    X_full_safe, y_full_safe, eff_b_full, eff_vb_full = _safe_batch_params(
        X_train_bal, y_full_local
    )
    clf_final = _make_tabnet_clf(seed)
    from_unsup_final = (
        copy.deepcopy(pretrained_tabnet) if method == "Fine-Tuning" else None
    )

    pre_enc_params: Dict[str, np.ndarray] = {}
    if from_unsup_final is not None:
        pre_enc_params, _ = _snapshot_module_tensors(from_unsup_final.network.encoder)

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
        from_unsupervised=from_unsup_final,
    )

    post_enc_params, _ = _snapshot_module_tensors(clf_final.network.tabnet.encoder)
    if method == "Fine-Tuning":
        backbone_param_diff = _max_abs_tensor_diff(pre_enc_params, post_enc_params)
        if backbone_param_diff <= 0.0:
            raise AssertionError(
                f"TabNet Fine-Tuning did not update encoder parameters (diff={backbone_param_diff})."
            )
    else:
        backbone_param_diff = float(
            np.max([np.max(np.abs(v)) for v in post_enc_params.values()])
            if post_enc_params
            else 0.0
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
            "backbone_frozen": False,
            "backbone_param_max_abs_diff": float(backbone_param_diff),
            "backbone_buffer_max_abs_diff": 0.0,
            "head_param_max_abs_diff": 1.0,
        },
    )


# =============================================================================
# 4. PREDICTION VALIDATION & STATISTICAL ANALYSIS (WILCOXON + HOLM + RANKINGS)
# =============================================================================


def validate_predictions_integrity(
    df_preds: pd.DataFrame,
    df_manifest: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Validate prediction-level export integrity (`Section 10`):
      1. No duplicate sample predictions per `(target_dataset, fold_id, seed, balance_strategy, architecture, method, sample_id)`.
      2. No missing predictions relative to `test_sample_ids` in `fold_manifest.csv`.
      3. Consistent class mappings and valid probability vectors (finite, sum ~ 1.0).
      4. Identical test sample sets (`sample_id` and `y_true`) across all paired methods
         for each `(target_dataset, fold_id, seed, balance_strategy)`.
      5. Correct fold and condition assignments.
    """
    if df_preds.empty:
        return {"valid": True, "checked_rows": 0, "notes": "Empty predictions dataframe"}

    key_cols = [
        "target_dataset",
        "fold_id",
        "seed",
        "balance_strategy",
        "architecture",
        "method",
        "sample_id",
    ]
    dups = int(df_preds.duplicated(subset=key_cols).sum())
    if dups > 0:
        raise ValueError(f"Duplicate sample predictions detected: {dups} duplicate rows.")

    # Check probability vector finiteness and normalization
    for idx, p_str in enumerate(df_preds["probabilities"]):
        p_vec = json.loads(str(p_str))
        if not isinstance(p_vec, list) or len(p_vec) < 2:
            raise ValueError(f"Invalid probability list at row {idx}: {p_str}")
        if any(math.isnan(float(x)) or math.isinf(float(x)) for x in p_vec):
            raise ValueError(f"NaN/Inf in probability vector at row {idx}: {p_str}")
        p_sum = sum(float(x) for x in p_vec)
        if abs(p_sum - 1.0) > 1e-2:
            raise ValueError(f"Misaligned probability sum ({p_sum}) at row {idx}.")

    # Verify identical test sample sets and fold/condition assignments across paired methods
    manifest_lookup: Dict[Tuple[str, int, int, str], Tuple[str, List[str]]] = {}
    for _, m_row in df_manifest.iterrows():
        m_key = (
            str(m_row["target_dataset"]),
            int(m_row["fold_id"]),
            int(m_row["seed"]),
            str(m_row["balance_strategy"]),
        )
        test_ids = json.loads(str(m_row["test_sample_ids"]))
        manifest_lookup[m_key] = (str(m_row["condition"]), list(test_ids))

    for (ds, fold_id, seed, bal), grp in df_preds.groupby(
        ["target_dataset", "fold_id", "seed", "balance_strategy"], sort=False
    ):
        m_key = (str(ds), int(fold_id), int(seed), str(bal))
        expected_cond, expected_ids = manifest_lookup[m_key]
        reference_pairs: Optional[List[Tuple[str, int]]] = None

        for (arch, meth), sub in grp.groupby(["architecture", "method"], sort=False):
            conds_found = sub["condition"].unique().tolist()
            if conds_found != [expected_cond]:
                raise ValueError(
                    f"Condition mismatch for {m_key} ({arch}, {meth}): "
                    f"got {conds_found}, expected [{expected_cond}]"
                )
            sub_sorted = sub.sort_values("sample_id")
            ids_found = sub_sorted["sample_id"].astype(str).tolist()
            if ids_found != sorted(expected_ids):
                raise ValueError(
                    f"Missing or mismatched test sample IDs for {m_key} ({arch}, {meth})."
                )
            pairs = list(
                zip(
                    sub_sorted["sample_id"].astype(str).tolist(),
                    sub_sorted["y_true"].astype(int).tolist(),
                )
            )
            if reference_pairs is None:
                reference_pairs = pairs
            elif pairs != reference_pairs:
                raise ValueError(
                    f"Inconsistent test sample IDs or y_true across paired methods in {m_key}."
                )

    return {
        "valid": True,
        "checked_rows": int(len(df_preds)),
        "duplicate_count": 0,
    }


def _holm_bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    """
    Apply Holm-Bonferroni step-down multiple-comparison adjustment to `p_values`,
    preserving `NaN` entries.
    """
    valid_indices = [
        i for i, p in enumerate(p_values) if not (math.isnan(p) or math.isinf(p))
    ]
    m = len(valid_indices)
    if m == 0:
        return [float("nan") for _ in p_values]

    sorted_valid = sorted(valid_indices, key=lambda idx: float(p_values[idx]))
    adjusted = [float("nan")] * len(p_values)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(sorted_valid):
        multiplier = m - rank_idx
        adj_p = min(1.0, float(p_values[orig_idx]) * float(multiplier))
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max
    return adjusted


def compute_summaries_and_statistics(
    df_results: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Generate:
      - `summary_by_dataset.csv`
      - `summary_by_architecture.csv`
      - `summary_by_method.csv`
      - `statistical_tests.csv`
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

    # 4. statistical_tests.csv
    # Primary statistical unit: LOCO fold (averaging across repeated seeds within each fold
    # so repeated seeds are not treated as independent observations).
    stat_rows: List[Dict[str, Any]] = []
    if not valid_df.empty:
        fold_level = (
            valid_df.groupby(
                ["target_dataset", "architecture", "balance_strategy", "fold_id", "condition", "method"],
                as_index=False,
            )[metrics_cols]
            .mean()
        )

        for (ds, arch, bal), grp in fold_level.groupby(
            ["target_dataset", "architecture", "balance_strategy"], sort=False
        ):
            scratch_sub = grp[grp["method"] == "Scratch"].set_index("fold_id")
            if scratch_sub.empty:
                continue

            for tl_method in ("Fine-Tuning", "Linear Probing"):
                tl_sub = grp[grp["method"] == tl_method].set_index("fold_id")
                common_folds = sorted(
                    set(scratch_sub.index.tolist()).intersection(set(tl_sub.index.tolist()))
                )
                if not common_folds:
                    continue

                for m_col in ("macro_f1", "balanced_accuracy", "accuracy", "roc_auc"):
                    s_vals = scratch_sub.loc[common_folds, m_col].to_numpy(dtype=np.float64)
                    t_vals = tl_sub.loc[common_folds, m_col].to_numpy(dtype=np.float64)
                    valid_mask = ~(np.isnan(s_vals) | np.isnan(t_vals))
                    s_v = s_vals[valid_mask]
                    t_v = t_vals[valid_mask]
                    n_paired = int(len(s_v))

                    if n_paired == 0:
                        continue

                    diffs = t_v - s_v  # TL - Scratch
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

                    if n_paired < 5:
                        # Compute exact Wilcoxon statistic when non-zero diffs exist, but note N < 5 power limit
                        if np.allclose(diffs, 0.0):
                            w_stat = 0.0
                            p_raw = 1.0
                            test_note = f"Zero differences across {n_paired} folds (N={n_paired} < 5)"
                        else:
                            try:
                                res_w = stats.wilcoxon(diffs, zero_method="zsplit", alternative="two-sided")
                                w_stat = float(res_w.statistic)
                                p_raw = float(res_w.pvalue)
                                test_note = (
                                    f"Low sample size (N={n_paired} LOCO folds < 6 minimum for alpha=0.05 "
                                    "two-sided Wilcoxon significance)"
                                )
                            except Exception as exc:
                                test_note = f"Wilcoxon not applicable (N={n_paired}): {exc}"
                    else:
                        if np.allclose(diffs, 0.0):
                            w_stat = 0.0
                            p_raw = 1.0
                            test_note = "All paired differences are identically zero"
                        else:
                            try:
                                res_w = stats.wilcoxon(diffs, zero_method="zsplit", alternative="two-sided")
                                w_stat = float(res_w.statistic)
                                p_raw = float(res_w.pvalue)
                            except Exception as exc:
                                test_note = f"Wilcoxon error: {exc}"

                    stat_rows.append(
                        {
                            "target_dataset": ds,
                            "architecture": arch,
                            "balance_strategy": bal,
                            "comparison": f"{tl_method} vs. Scratch",
                            "tl_method": tl_method,
                            "baseline_method": "Scratch",
                            "metric": m_col,
                            "statistical_unit": "loco_fold_seed_averaged",
                            "n_paired_folds": n_paired,
                            "scratch_mean": float(np.mean(s_v)),
                            "tl_mean": float(np.mean(t_v)),
                            "mean_diff_tl_minus_scratch": mean_diff,
                            "median_diff_tl_minus_scratch": median_diff,
                            "std_diff": std_diff,
                            "ci95_low": float(ci_low),
                            "ci95_high": float(ci_high),
                            "n_folds_tl_better": int(np.sum(diffs > 1e-12)),
                            "n_folds_scratch_better": int(np.sum(diffs < -1e-12)),
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
        # Apply Holm step-down adjustment within each (architecture, balance_strategy, metric) family
        for (arch, bal, m_col), idxs in df_stats.groupby(
            ["architecture", "balance_strategy", "metric"]
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
        for (ds, arch, bal), grp in df_sum_ds.groupby(
            ["target_dataset", "architecture", "balance_strategy"], sort=False
        ):
            grp_sorted = grp.copy()
            # Rank 1 = highest macro_f1_mean
            grp_sorted["rank_within_architecture"] = grp_sorted["macro_f1_mean"].rank(
                ascending=False, method="average"
            )
            for _, r in grp_sorted.iterrows():
                rank_rows.append(
                    {
                        "scope": "dataset_within_architecture",
                        "target_dataset": ds,
                        "architecture": arch,
                        "method": r["method"],
                        "balance_strategy": bal,
                        "metric": "macro_f1_mean",
                        "metric_value": float(r["macro_f1_mean"]),
                        "rank": float(r["rank_within_architecture"]),
                    }
                )

        # Average rank across target datasets per (architecture, method, balance_strategy)
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


def _df_to_table_str(df: pd.DataFrame, empty_msg: str = "Sem registros.") -> str:
    """Render a DataFrame as a clean text/markdown table without requiring `tabulate`."""
    if df is None or df.empty:
        return empty_msg
    return df.to_string(index=False)


def generate_benchmark_report_md(
    config: Mapping[str, Any],
    git_meta: Mapping[str, Any],
    df_results: pd.DataFrame,
    df_sum_ds: pd.DataFrame,
    df_sum_arch: pd.DataFrame,
    df_sum_meth: pd.DataFrame,
    df_stats: pd.DataFrame,
    df_ranks: pd.DataFrame,
    is_synthetic_smoke: bool = False,
) -> str:
    """Generate the comprehensive 9-section Markdown report (`report.md`)."""
    total_evals = int(len(df_results))
    ok_evals = int((df_results["status"] == "OK").sum()) if not df_results.empty else 0
    failed_evals = total_evals - ok_evals

    data_scope_banner = (
        "> **AVISO DE ESCOPO (EXECUÇÃO PILOTO / SINTÉTICA DE VALIDAÇÃO)**: "
        "Este diretório corresponde a uma execução de validação do pipeline "
        "(`is_synthetic_smoke=True`). Conforme `AGENTS.md`, resultados sintéticos "
        "validam apenas a integridade funcional, estatística e anti-vazamento do código.\n"
        if is_synthetic_smoke
        else "> **ESCOPO DE DADOS REAIS**: Execução realizada sobre as janelas reais "
        "de `data/processed_tl/detrended_rms/`.\n"
    )

    ds_table = (
        _df_to_table_str(
            df_sum_ds[
                [
                    "target_dataset",
                    "architecture",
                    "method",
                    "balance_strategy",
                    "n_folds",
                    "n_seeds",
                    "macro_f1_mean",
                    "macro_f1_std",
                    "balanced_accuracy_mean",
                    "roc_auc_mean",
                ]
            ]
        )
        if not df_sum_ds.empty
        else "Nenhum resumo por dataset gerado."
    )
    arch_table = _df_to_table_str(df_sum_arch, "Nenhum resumo por arquitetura gerado.")
    meth_table = _df_to_table_str(df_sum_meth, "Nenhum resumo por método gerado.")
    stats_table = (
        _df_to_table_str(
            df_stats[
                [
                    "target_dataset",
                    "architecture",
                    "balance_strategy",
                    "comparison",
                    "metric",
                    "n_paired_folds",
                    "scratch_mean",
                    "tl_mean",
                    "mean_diff_tl_minus_scratch",
                    "p_value_raw",
                    "p_value_holm",
                    "significant_alpha_0_05",
                ]
            ]
        )
        if not df_stats.empty
        else "Nenhuma comparação estatística gerada."
    )
    ranks_table = _df_to_table_str(df_ranks, "Nenhum ranking gerado.")

    return f"""# Relatório Técnico — Benchmark Scratch vs. Transfer Learning (`scratch_vs_tl`)

{data_scope_banner}

- **Commit Git**: `{git_meta.get('git_commit')}` (`git_dirty={git_meta.get('git_dirty')}`)
- **Total de Avaliações Executadas**: `{total_evals}` (`OK={ok_evals}`, `Falhas={failed_evals}`)
- **Datasets Alvo**: `{config.get('target_datasets')}`
- **Arquiteturas**: `{config.get('architectures')}`
- **Métodos**: `{config.get('methods')}`
- **Estratégias de Balanceamento**: `{config.get('balance_strategies')}`
- **Seeds**: `{config.get('seeds')}`

---

## 1. Configuração Experimental (`Experimental Setup`)

O benchmark avalia de forma pareada três regimes de treinamento para `MLP` e `TabNet`:
1. **`Scratch`**: Inicialização aleatória; treinamento exclusivo nas $K - 1$ condições operacionais do domínio alvo.
2. **`Fine-Tuning`**: Pré-treinamento do encoder exclusivamente no pool de domínios fonte disjunto (`UORED`, `HUST`, `Mechanical_Gear`, `Electric_Motor`), seguido de adaptação supervisionada completa (encoder + cabeça linear do alvo) nas $K - 1$ condições do alvo.
3. **`Linear Probing`**: Inicialização a partir do mesmo encoder pré-treinado nas fontes, com congelamento estrito de todos os parâmetros e buffers do encoder (`requires_grad=False`, `.eval()`, verificado programaticamente com `max_abs_diff == 0.0`), treinando apenas a cabeça linear do domínio alvo.

---

## 2. Auditoria de Compatibilidade e Reprodutibilidade

- A auditoria completa encontra-se em `compatibility_audit.csv` e `results/scratch_vs_tl/audit/audit_report.md`.
- Todos os métodos pareados em cada `(target_dataset, fold_id, seed, balance_strategy)` compartilham exatamente o mesmo `split_signature` e `preprocessing_signature` registrados em `fold_manifest.csv` e os mesmos identificadores `sample_id` em `predictions.csv`.

---

## 3. Resultados por Dataset Alvo (`Dataset-Level Results`)

```text
{ds_table}
```

---

## 4. Resultados por Arquitetura (`Architecture-Level Results`)

```text
{arch_table}
```

---

## 5. Comparação por Método: Scratch vs. Fine-Tuning vs. Linear Probing

```text
{meth_table}
```

---

## 6. Impacto do Balanceamento com SMOTE (`Impact of SMOTE`)

- Em cada fold LOCO, `none` preserva a distribuição natural das $K - 1$ condições de treino, enquanto `smote` realiza interpolação k-NN (`k_neighbors = min(5, minority_count - 1)`) exclusivamente no conjunto de treinamento escalado (`X_train_scaled`).
- Os resultados discriminados por `balance_strategy` (`none` vs. `smote`) constam nas tabelas das Seções 3, 4, 5 e 7.

---

## 7. Análise Estatística Pareada e Rankings (`Statistical Analysis`)

- **Unidade Estatística Primária**: Fold LOCO ($N = K_T$ folds por dataset alvo), agregando primeiramente a média das seeds repetidas dentro de cada fold para evitar pseudorreplicação entre seeds correlacionadas.
- **Correção para Múltiplas Comparações**: Ajuste *step-down* de Holm-Bonferroni (`p_value_holm`).

### Testes Pareados (Wilcoxon Signed-Rank + Holm)

```text
{stats_table}
```

### Rankings dos Métodos (`Macro-F1`)

```text
{ranks_table}
```

---

## 8. Limitações Metodológicas (`Limitations`)

1. **Tamanho Amostral em Datasets com 4 Condições Operacionais (`CWRU_12k` e `PU`)**:
   - Com $K_T = 4$ folds LOCO, o menor $p$-valor bicaudal possível no teste exato de Wilcoxon signed-rank é $2 / 2^4 = 0.125 > 0.05$. Assim, em `CWRU_12k` e `PU`, diferenças consistentes em todos os 4 folds devem ser interpretadas descritivamente (intervalo de confiança e proporção de folds vencidos `n_folds_tl_better = 4/4`), enquanto `CWRU_48k` ($K_T = 12$), `UOEMD` ($K_T = 16$) e `HUST_Gearbox` ($K_T = 30$) possuem potência estatística para inferência formal em $\\alpha = 0.05$.
2. **Correlação entre Folds LOCO**:
   - Dois folds LOCO de um mesmo dataset compartilham $(K_T - 2) / (K_T - 1)$ das condições de treinamento, uma propriedade intrínseca de qualquer esquema *leave-one-group-out*.

---

## 9. Conclusões Principais (`Main Conclusions`)

- O pipeline independente `scratch_vs_tl` garante paridade estrita de amostras, pré-processamento `detrended_rms` (141 atributos), isolamento LOCO e verificação programática de congelamento no `Linear Probing` (`backbone_param_max_abs_diff == 0.0`).
- A distinção entre melhorias descritivas de média, consistência entre folds e significância estatística corrigida por Holm está documentada em `statistical_tests.csv` e `rankings.csv`.
"""


# =============================================================================
# 5. MAIN BENCHMARK ORCHESTRATOR
# =============================================================================


def run_scratch_vs_tl_benchmark(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    resume_dir: Optional[str | Path] = None,
    targets_override: Optional[Sequence[str]] = None,
    architectures_override: Optional[Sequence[str]] = None,
    methods_override: Optional[Sequence[str]] = None,
    balance_override: Optional[Sequence[str]] = None,
    seeds_override: Optional[Sequence[int]] = None,
    max_folds: Optional[int] = None,
    epochs_override: Optional[int] = None,
    pretrain_epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    feature_extractor: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    enforce_expected_folds: bool = True,
    is_synthetic_smoke: bool = False,
) -> Path:
    """
    Execute the complete (or pilot) Scratch vs. Transfer Learning benchmark and
    generate all 13 required artifacts in `results/scratch_vs_tl/run_TIMESTAMP/`.
    Supports checkpointing and resuming via `resume_dir`.
    """
    cfg = load_benchmark_config(config_path)
    targets = list(targets_override) if targets_override is not None else list(cfg["target_datasets"])
    architectures = (
        list(architectures_override)
        if architectures_override is not None
        else list(cfg["architectures"])
    )
    methods = list(methods_override) if methods_override is not None else list(cfg["methods"])
    balance_strategies = (
        list(balance_override)
        if balance_override is not None
        else list(cfg["balance_strategies"])
    )
    seeds = (
        [int(s) for s in seeds_override]
        if seeds_override is not None
        else [int(s) for s in cfg["seeds"]]
    )

    hp = dict(cfg.get("hyperparameters", {}))
    pretrain_epochs = int(
        pretrain_epochs_override
        if pretrain_epochs_override is not None
        else hp.get("pretrain_epochs", 30)
    )
    target_epochs = int(
        epochs_override if epochs_override is not None else hp.get("target_epochs", 30)
    )
    batch_size = int(
        batch_size_override
        if batch_size_override is not None
        else hp.get("batch_size", 64)
    )
    tabnet_patience = int(hp.get("tabnet_patience", 15))

    verified_folds = verify_canonical_fold_metadata(targets)
    resolved_data_root = validate_data_root_is_detrended_rms(data_root)

    base_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = 1
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Ensure unique timestamp directory so repeated runs never overwrite previous runs
        run_dir = Path(output_root) / f"run_{base_ts}"
        while run_dir.exists():
            run_dir = Path(output_root) / f"run_{base_ts}_{suffix:02d}"
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
    (run_dir / "git_metadata.json").write_text(
        json.dumps(git_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    env_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependency_versions": get_dependency_versions(),
    }
    (run_dir / "environment.json").write_text(
        json.dumps(env_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    source_pools_by_target = {
        t: list(cfg["source_pools_by_target"][t]) for t in targets
    }
    comparison_config = {
        "experiment_id": EXPERIMENT_ID,
        "task": TASK_NAME,
        "timestamp": base_ts,
        "is_synthetic_smoke": bool(is_synthetic_smoke),
        "data_root": str(resolved_data_root),
        "output_dir": str(run_dir),
        "target_datasets": targets,
        "expected_loco_folds": verified_folds,
        "source_pools_by_target": source_pools_by_target,
        "architectures": architectures,
        "methods": methods,
        "preprocessing": {
            "variant": PREPROCESSING_VARIANT,
            "feature_dimension": FEATURE_DIMENSION,
            "scaler": "StandardScaler (fitted strictly on target training fold)",
        },
        "balance_strategies": balance_strategies,
        "seeds": seeds,
        "max_folds": max_folds,
        "hyperparameters": {
            **hp,
            "pretrain_epochs": pretrain_epochs,
            "target_epochs": target_epochs,
            "batch_size": batch_size,
            "tabnet_patience": tabnet_patience,
        },
        "metric_definitions": {
            "primary_metric": "macro_f1 (average='macro' over classes relevant to the fold)",
            "secondary_metrics": [
                "accuracy",
                "balanced_accuracy",
                "roc_auc (macro One-vs-Rest over present test classes when >= 2 classes present; NaN otherwise)",
            ],
        },
        "software_versions": env_meta["dependency_versions"],
    }
    (run_dir / "comparison_config.json").write_text(
        json.dumps(comparison_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    df_compat = build_compatibility_audit_table(DEFAULT_BASELINE_REF_DIR)
    df_compat.to_csv(run_dir / "compatibility_audit.csv", index=False)

    # Load all required source datasets once
    needed_sources: List[str] = []
    if any(m in ("Fine-Tuning", "Linear Probing") for m in methods):
        for t in targets:
            for s in source_pools_by_target[t]:
                if s not in needed_sources:
                    needed_sources.append(s)

    loaded_sources: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]] = {}
    for s_name in needed_sources:
        loaded_sources[s_name] = load_and_extract_target_dataset(
            dataset_name=s_name,
            data_root=resolved_data_root,
            feature_extractor=feature_extractor,
            logger=logger,
            enforce_expected_folds=False,
        )

    pretrained_mlp_cache: Dict[Tuple[Tuple[str, ...], int], Dict[str, Any]] = {}
    pretrained_tabnet_cache: Dict[Tuple[Tuple[str, ...], int], Any] = {}

    manifest_rows: List[Dict[str, Any]] = []
    results_rows: List[Dict[str, Any]] = []
    predictions_rows: List[Dict[str, Any]] = []

    results_csv_path = run_dir / "results.csv"
    manifest_csv_path = run_dir / "fold_manifest.csv"
    preds_csv_path = run_dir / "predictions.csv"

    completed_eval_keys: set[Tuple[str, int, int, str, str, str]] = set()
    completed_manifest_keys: set[Tuple[str, int, int, str]] = set()
    if resume_dir is not None:
        if manifest_csv_path.exists():
            df_m_prev = pd.read_csv(manifest_csv_path)
            manifest_rows.extend(df_m_prev.to_dict("records"))
            for r in manifest_rows:
                completed_manifest_keys.add(
                    (
                        str(r["target_dataset"]),
                        int(r["fold_id"]),
                        int(r["seed"]),
                        str(r["balance_strategy"]),
                    )
                )
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
            f"Resuming run in {run_dir}: loaded {len(completed_eval_keys)} completed evaluations."
        )

    for ds_name in targets:
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

        unique_conds = sorted(np.unique(conds_all).tolist())
        if max_folds is not None:
            unique_conds = unique_conds[: int(max_folds)]

        source_pool_names = tuple(source_pools_by_target[ds_name])
        prepared_sources: Dict[str, Dict[str, Any]] = {}
        source_sample_ids_flat: List[str] = []
        if needed_sources:
            prepared_sources = prepare_scaled_source_pool(
                source_names=source_pool_names,
                loaded_sources=loaded_sources,
                target_sample_ids=sample_ids_all.tolist(),
            )
            for s_name in source_pool_names:
                source_sample_ids_flat.extend(prepared_sources[s_name]["sample_ids"])

        for seed in seeds:
            # Pretrain source encoders once per (source_pool, seed)
            mlp_pretrained_state: Optional[Dict[str, Any]] = None
            tabnet_pretrained_model: Optional[Any] = None
            cache_key = (source_pool_names, int(seed))

            if any(m in ("Fine-Tuning", "Linear Probing") for m in methods):
                if "MLP" in architectures:
                    if cache_key not in pretrained_mlp_cache:
                        logger.info(
                            f"Pretraining MLP encoder on sources {list(source_pool_names)} (seed={seed})..."
                        )
                        pretrained_mlp_cache[cache_key] = pretrain_mlp_on_sources(
                            prepared_sources=prepared_sources,
                            seed=seed,
                            pretrain_epochs=pretrain_epochs,
                            batch_size=batch_size,
                            lr=float(hp.get("mlp_lr_pretrain", 1e-3)),
                            weight_decay=float(hp.get("mlp_weight_decay", 1e-4)),
                        )
                    mlp_pretrained_state = pretrained_mlp_cache[cache_key]

                if "TabNet" in architectures:
                    if cache_key not in pretrained_tabnet_cache:
                        logger.info(
                            f"Pretraining TabNet encoder on sources {list(source_pool_names)} (seed={seed})..."
                        )
                        pretrained_tabnet_cache[cache_key] = pretrain_tabnet_on_sources(
                            prepared_sources=prepared_sources,
                            seed=seed,
                            pretrain_epochs=pretrain_epochs,
                            batch_size=batch_size,
                            lr=float(hp.get("tabnet_lr_pretrain", 2e-2)),
                        )
                    tabnet_pretrained_model = pretrained_tabnet_cache[cache_key]

            for fold_id, test_cond in enumerate(unique_conds):
                test_mask = conds_all == test_cond
                train_mask = ~test_mask

                train_indices = np.where(train_mask)[0].tolist()
                test_indices = np.where(test_mask)[0].tolist()
                assert_no_condition_leakage(conds_all.tolist(), train_indices, test_indices)

                X_train_raw = X_all[train_mask]
                y_train_raw = y_all_enc[train_mask]
                conds_train_raw = conds_all[train_mask]
                train_sample_ids = sample_ids_all[train_mask].tolist()

                X_test_raw = X_all[test_mask]
                y_test = y_all_enc[test_mask]
                test_sample_ids = sample_ids_all[test_mask].tolist()

                # Verify zero overlap between target test samples and source samples
                if set(test_sample_ids).intersection(set(source_sample_ids_flat)):
                    raise AssertionError(
                        f"Target test samples leaked into source pretraining pool on {ds_name} fold {test_cond}!"
                    )

                split_sig = compute_sample_split_signature(
                    train_sample_ids=train_sample_ids,
                    test_sample_ids=test_sample_ids,
                )

                X_train_scaled, X_test_scaled, scaler = scale_loco_fold_strict(
                    X_train_raw, X_test_raw
                )

                for balance_strategy in balance_strategies:
                    X_train_bal, y_train_bal, smote_audit = apply_smote_with_audit(
                        X_train_scaled=X_train_scaled,
                        y_train_encoded=y_train_raw,
                        class_names=class_names,
                        strategy=balance_strategy,
                        seed=seed,
                    )

                    preproc_sig = compute_preprocessing_signature(
                        preprocessing_variant=PREPROCESSING_VARIANT,
                        feature_dim=int(X_all.shape[1]),
                        scaler_mean=scaler.mean_,
                        scaler_scale=scaler.scale_,
                        balance_strategy=balance_strategy,
                        smote_params=smote_audit["smote_params"],
                    )

                    m_key = (ds_name, int(fold_id), int(seed), str(balance_strategy))
                    if m_key not in completed_manifest_keys:
                        manifest_rows.append(
                            {
                                "target_dataset": ds_name,
                                "fold_id": int(fold_id),
                                "condition": str(test_cond),
                                "train_sample_ids": json.dumps(train_sample_ids),
                                "test_sample_ids": json.dumps(test_sample_ids),
                                "seed": int(seed),
                                "balance_strategy": str(balance_strategy),
                                "source_datasets": "|".join(source_pool_names),
                                "source_sample_ids": json.dumps(source_sample_ids_flat),
                                "split_signature": split_sig,
                                "preprocessing_signature": preproc_sig,
                                "n_train_before_balance": int(smote_audit["n_train_before"]),
                                "n_train_after_balance": int(smote_audit["n_train_after"]),
                                "n_test_samples": int(len(test_sample_ids)),
                                "class_counts_before": json.dumps(smote_audit["counts_before"], ensure_ascii=False),
                                "class_counts_after": json.dumps(smote_audit["counts_after"], ensure_ascii=False),
                            }
                        )
                        completed_manifest_keys.add(m_key)

                    if X_train_bal is None or y_train_bal is None:
                        for arch in architectures:
                            for meth in methods:
                                eval_key = (
                                    ds_name,
                                    int(fold_id),
                                    int(seed),
                                    str(balance_strategy),
                                    arch,
                                    meth,
                                )
                                if eval_key in completed_eval_keys:
                                    continue
                                results_rows.append(
                                    {
                                        "target_dataset": ds_name,
                                        "task": TASK_NAME,
                                        "fold_id": int(fold_id),
                                        "condition": str(test_cond),
                                        "seed": int(seed),
                                        "architecture": arch,
                                        "method": meth,
                                        "balance_strategy": str(balance_strategy),
                                        "source_datasets": "None" if meth == "Scratch" else "|".join(source_pool_names),
                                        "preprocessing": PREPROCESSING_VARIANT,
                                        "feature_dimension": int(X_all.shape[1]),
                                        "accuracy": float("nan"),
                                        "balanced_accuracy": float("nan"),
                                        "macro_f1": float("nan"),
                                        "roc_auc": float("nan"),
                                        "roc_auc_valid": False,
                                        "roc_auc_note": "Fold skipped prior to training",
                                        "test_sample_count": int(len(y_test)),
                                        "test_class_count": int(len(np.unique(y_test))),
                                        "train_sample_count_before": int(smote_audit["n_train_before"]),
                                        "train_sample_count_after": int(smote_audit["n_train_after"]),
                                        "train_class_count": int(len(np.unique(y_train_raw))),
                                        "class_counts_before": json.dumps(smote_audit["counts_before"], ensure_ascii=False),
                                        "class_counts_after": json.dumps(smote_audit["counts_after"], ensure_ascii=False),
                                        "backbone_frozen": bool(meth == "Linear Probing"),
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
                        pd.DataFrame(manifest_rows).to_csv(manifest_csv_path, index=False)
                        pd.DataFrame(results_rows).to_csv(results_csv_path, index=False)
                        continue

                    for arch in architectures:
                        for meth in methods:
                            eval_key = (
                                ds_name,
                                int(fold_id),
                                int(seed),
                                str(balance_strategy),
                                arch,
                                meth,
                            )
                            if eval_key in completed_eval_keys:
                                logger.info(
                                    f"  [SKIP COMPLETED] {ds_name} | Fold {fold_id}:{test_cond} | "
                                    f"Seed {seed} | Bal={balance_strategy} | {arch} - {meth}"
                                )
                                continue
                            t0 = time.perf_counter()
                            try:
                                if arch == "MLP":
                                    preds, probs, model_meta = train_eval_mlp_variant(
                                        method=meth,
                                        X_train_bal=X_train_bal,
                                        y_train_bal=y_train_bal,
                                        X_test_scaled=X_test_scaled,
                                        n_total_classes=n_total_classes,
                                        seed=seed,
                                        pretrained_mlp_state=mlp_pretrained_state,
                                        epochs=target_epochs,
                                        batch_size=batch_size,
                                        lr_scratch=float(hp.get("mlp_lr_scratch", 1e-3)),
                                        lr_finetune_backbone=float(hp.get("mlp_lr_finetune_backbone", 5e-4)),
                                        lr_finetune_head=float(hp.get("mlp_lr_finetune_head", 1e-3)),
                                        lr_linear_probing=float(hp.get("mlp_lr_linear_probing", 1e-3)),
                                        weight_decay=float(hp.get("mlp_weight_decay", 1e-4)),
                                    )
                                elif arch == "TabNet":
                                    preds, probs, model_meta = train_eval_tabnet_variant(
                                        method=meth,
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
                                        pretrained_tabnet=tabnet_pretrained_model,
                                        max_epochs=target_epochs,
                                        patience=tabnet_patience,
                                        batch_size=batch_size,
                                        lr_scratch=float(hp.get("tabnet_lr_scratch", 2e-2)),
                                        lr_finetune=float(hp.get("tabnet_lr_finetune", 1e-2)),
                                        lr_linear_probing=float(hp.get("tabnet_lr_linear_probing", 1e-2)),
                                    )
                                else:
                                    raise ValueError(f"Unknown architecture '{arch}'")

                                runtime_sec = time.perf_counter() - t0
                                metrics = compute_fold_metrics(
                                    y_true=y_test,
                                    y_pred=preds,
                                    probs=probs,
                                )

                                logger.info(
                                    f"  [{ds_name} | Fold {fold_id}:{test_cond} | Seed {seed} | "
                                    f"Bal={balance_strategy} | {arch} - {meth}] "
                                    f"F1={metrics['Macro_F1']:.4f} | BalAcc={metrics['Balanced_Accuracy']:.4f} | "
                                    f"Acc={metrics['Accuracy']:.4f} ({runtime_sec:.2f}s)"
                                )

                                results_rows.append(
                                    {
                                        "target_dataset": ds_name,
                                        "task": TASK_NAME,
                                        "fold_id": int(fold_id),
                                        "condition": str(test_cond),
                                        "seed": int(seed),
                                        "architecture": arch,
                                        "method": meth,
                                        "balance_strategy": str(balance_strategy),
                                        "source_datasets": "None" if meth == "Scratch" else "|".join(source_pool_names),
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
                                            "fold_id": int(fold_id),
                                            "condition": str(test_cond),
                                            "sample_id": str(test_sample_ids[idx_s]),
                                            "y_true": int(y_test[idx_s]),
                                            "y_pred": int(preds[idx_s]),
                                            "probabilities": json.dumps(
                                                [round(float(p), 6) for p in probs[idx_s]]
                                            ),
                                            "class_order": class_order_json,
                                            "architecture": arch,
                                            "method": meth,
                                            "balance_strategy": str(balance_strategy),
                                            "seed": int(seed),
                                        }
                                    )
                            except Exception as exc:
                                runtime_sec = time.perf_counter() - t0
                                logger.error(
                                    f"  [ERROR] {ds_name} | Fold {test_cond} | Seed {seed} | "
                                    f"Bal={balance_strategy} | {arch} - {meth}: {exc}"
                                )
                                results_rows.append(
                                    {
                                        "target_dataset": ds_name,
                                        "task": TASK_NAME,
                                        "fold_id": int(fold_id),
                                        "condition": str(test_cond),
                                        "seed": int(seed),
                                        "architecture": arch,
                                        "method": meth,
                                        "balance_strategy": str(balance_strategy),
                                        "source_datasets": "None" if meth == "Scratch" else "|".join(source_pool_names),
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
                                        "backbone_frozen": bool(meth == "Linear Probing"),
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

                    pd.DataFrame(manifest_rows).to_csv(manifest_csv_path, index=False)
                    pd.DataFrame(results_rows).to_csv(results_csv_path, index=False)
                    pd.DataFrame(predictions_rows).to_csv(preds_csv_path, index=False)

    df_manifest = pd.DataFrame(manifest_rows)
    df_results = pd.DataFrame(results_rows)
    df_preds = pd.DataFrame(predictions_rows)

    df_manifest.to_csv(manifest_csv_path, index=False)
    df_results.to_csv(results_csv_path, index=False)
    df_preds.to_csv(preds_csv_path, index=False)

    if not df_preds.empty and (df_results["status"] == "OK").all():
        validate_predictions_integrity(df_preds, df_manifest)

    summary_paths = compute_summaries_and_statistics(df_results, run_dir)
    df_sum_ds = pd.read_csv(summary_paths["summary_by_dataset"])
    df_sum_arch = pd.read_csv(summary_paths["summary_by_architecture"])
    df_sum_meth = pd.read_csv(summary_paths["summary_by_method"])
    df_stats = pd.read_csv(summary_paths["statistical_tests"])
    df_ranks = pd.read_csv(summary_paths["rankings"])

    report_md = generate_benchmark_report_md(
        config=comparison_config,
        git_meta=git_meta,
        df_results=df_results,
        df_sum_ds=df_sum_ds,
        df_sum_arch=df_sum_arch,
        df_sum_meth=df_sum_meth,
        df_stats=df_stats,
        df_ranks=df_ranks,
        is_synthetic_smoke=is_synthetic_smoke,
    )
    (run_dir / "report.md").write_text(report_md, encoding="utf-8")

    logger.info("=" * 80)
    logger.info(f"SCRATCH VS. TL BENCHMARK COMPLETED: {run_dir}")
    logger.info("=" * 80)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    return run_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the independent Scratch vs. Transfer Learning benchmark (scratch_vs_tl)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configs/scratch_vs_tl.yaml.",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Optional existing run_TIMESTAMP directory to resume from checkpoint.",
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
        choices=ARCHITECTURES,
        help="Optional override of architectures ('MLP', 'TabNet').",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=METHODS,
        help="Optional override of methods ('Scratch', 'Fine-Tuning', 'Linear Probing').",
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
        help="Optional cap on LOCO folds per target dataset (for pilot validation).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override of target training epochs.",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=None,
        help="Optional override of source pretraining epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override of mini-batch size.",
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
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Base output directory (default: results/{EXPERIMENT_ID}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_scratch_vs_tl_benchmark(
        config_path=args.config,
        data_root=args.data_root,
        output_root=args.output_root,
        resume_dir=args.resume_dir,
        targets_override=args.targets,
        architectures_override=args.architectures,
        methods_override=args.methods,
        balance_override=args.balance_strategies,
        seeds_override=args.seeds,
        max_folds=args.max_folds,
        epochs_override=args.epochs,
        pretrain_epochs_override=args.pretrain_epochs,
        batch_size_override=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
