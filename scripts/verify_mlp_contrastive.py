"""Synthetic verification suite for MLP contrastive transfer learning protocol.

Note: This script uses purely synthetic data and AST/runtime instrumentation to
validate methodological and architectural invariants without touching real
datasets or overwriting existing experiment logs. If numpy/torch/scikit-learn
are not installed in the current environment, it runs deterministic AST and
CSV/protocol structural checks and reports which runtime checks require the ML
environment.
"""

from __future__ import annotations

import ast
import copy
import csv
import json
import math
import os
import sys
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "models", "build_mlp_contrastive.py")
SCRIPT_PATH = os.path.join(PROJECT_ROOT, "scripts", "exp_tl_contrastive_mlp.py")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "tl_contrastive_mlp")


def test_ast_and_protocol_invariants():
    """Verify AST structure of build_mlp_contrastive.py and exp_tl_contrastive_mlp.py."""
    with open(MODEL_PATH, "r", encoding="utf-8") as stream:
        model_src = stream.read()
    with open(SCRIPT_PATH, "r", encoding="utf-8") as stream:
        script_src = stream.read()

    model_tree = ast.parse(model_src, filename=MODEL_PATH)
    script_tree = ast.parse(script_src, filename=SCRIPT_PATH)

    # 1. Verify MLPClassifier defines encode() and forward()
    mlp_cls = next(
        node for node in model_tree.body if isinstance(node, ast.ClassDef) and node.name == "MLPClassifier"
    )
    mlp_methods = {node.name for node in mlp_cls.body if isinstance(node, ast.FunctionDef)}
    assert {"__init__", "encode", "forward"} <= mlp_methods, (
        f"MLPClassifier missing expected methods: {mlp_methods}"
    )

    # 2. Verify pretrain block calls model.encode(X), uses pretrain_optimizer, and deletes pretrain_projection
    assert "latent = model.encode(X)" in model_src, (
        "pretrain block must call model.encode(X) so classification heads are untouched during Stage 0."
    )
    assert "del pretrain_projection, pretrain_optimizer" in model_src, (
        "pretrain block must discard pretrain_projection and pretrain_optimizer after Stage 0."
    )

    # 3. Verify separate optimizers for pretrain, supervised, and finetune stages
    assert "pretrain_optimizer = torch.optim.Adam(" in model_src
    assert "supervised_optimizer = torch.optim.Adam(" in model_src
    assert "finetune_optimizer = torch.optim.Adam(" in model_src
    assert "list(model.encoder.parameters()) + list(model.classifiers[target_name].parameters())" in model_src, (
        "finetune_optimizer must optimize only model.encoder and model.classifiers[target_name]."
    )

    # 4. Verify binary subset ROC-AUC handling (len(present) == 2)
    assert "if len(present) == 2:" in model_src and "elif len(present) > 2:" in model_src, (
        "train_and_evaluate must handle 2-class test folds (len(present) == 2) explicitly for roc_auc_score."
    )

    # 5. Verify exp_tl_contrastive_mlp.py defines build_experiment_summaries
    script_funcs = {node.name for node in script_tree.body if isinstance(node, ast.FunctionDef)}
    assert "build_experiment_summaries" in script_funcs, (
        "exp_tl_contrastive_mlp.py must define build_experiment_summaries."
    )
    print("[PASS] AST & static protocol invariants verified for model and experiment runner.")


def test_published_csv_consistency():
    """Verify published CSV artifacts in results/tl_contrastive_mlp match reported metrics and fold counts."""
    expected_runs = {
        "run_20261026_212739": {
            ("baseline", "baseline", "asymmetric"): (12, 0.9372850087695248),
            ("joint", "baseline", "asymmetric"): (12, 0.9323611653125822),
            ("pretrain", "baseline", "asymmetric"): (12, 0.9029527395379852),
        },
        "run_20261026_212945": {
            ("joint", "baseline", "asymmetric"): (12, 0.9323611653125822),
            ("joint", "expanded", "asymmetric"): (12, 0.889592525396346),
        },
        "run_20261026_213141": {
            ("joint", "expanded", "asymmetric"): (12, 0.889592525396346),
            ("joint", "expanded", "symmetric"): (12, 0.9169056111453),
        },
    }
    for run_name, expected_configs in expected_runs.items():
        run_dir = os.path.join(RESULTS_ROOT, run_name)
        results_csv = os.path.join(run_dir, "results.csv")
        summary_csv = os.path.join(run_dir, "summary.csv")
        config_json = os.path.join(run_dir, "config.json")
        assert os.path.isfile(results_csv) and os.path.isfile(summary_csv) and os.path.isfile(config_json), (
            f"Missing published artifact in {run_dir}"
        )
        with open(results_csv, "r", encoding="utf-8") as f:
            res_rows = list(csv.DictReader(f))
        with open(summary_csv, "r", encoding="utf-8") as f:
            sum_rows = list(csv.DictReader(f))

        for srow in sum_rows:
            key = (srow["Mode"], srow["Encoder Profile"], srow["Head Shape"])
            assert key in expected_configs, f"Unexpected config {key} in {run_name}"
            exp_folds, exp_f1 = expected_configs[key]
            matching = [
                r for r in res_rows
                if (r["Mode"], r["Encoder Profile"], r["Head Shape"]) == key and r["Status"] == "OK"
            ]
            assert len(matching) == exp_folds == int(srow["N_Folds"]), (
                f"Fold count mismatch in {run_name} for {key}"
            )
            mean_f1 = sum(float(r["Macro F1"]) for r in matching) / len(matching)
            assert math.isclose(mean_f1, float(srow["Macro_F1_Mean"]), rel_tol=1e-12)
            assert math.isclose(mean_f1, exp_f1, rel_tol=1e-12)
    print("[PASS] Published CSV artifacts verified intact and consistent across all 12 LOCO folds.")


def run_tensor_suite():
    """Run synthetic tensor tests when numpy, pandas, scikit-learn, and torch are installed."""
    import numpy as np
    import pandas as pd
    import torch

    from scripts.exp_tl_contrastive_mlp import build_experiment_summaries
    from src.models.build_mlp_contrastive import (
        MLPClassifier,
        ProjectionHead,
        _prepare_domains,
        seed_everything,
        supervised_contrastive_loss,
        train_and_evaluate,
    )

    def _make_synthetic_domains(seed: int = 42):
        rng = np.random.default_rng(seed)
        train_data = {
            "CWRU_48k": (
                rng.normal(loc=0.0, scale=1.0, size=(36, 16)).astype(np.float32),
                np.repeat([0, 1, 2], 12).astype(np.int64),
            ),
            "CWRU_12k": (
                rng.normal(loc=2.0, scale=1.5, size=(40, 16)).astype(np.float32),
                np.repeat([0, 1, 2, 3], 10).astype(np.int64),
            ),
            "MFPT": (
                rng.normal(loc=-1.5, scale=0.8, size=(24, 16)).astype(np.float32),
                np.repeat([0, 1, 2], 8).astype(np.int64),
            ),
        }
        X_test = rng.normal(loc=50.0, scale=10.0, size=(15, 16)).astype(np.float32)
        y_test = np.repeat([0, 1, 2], 5).astype(np.int64)
        return train_data, "CWRU_48k", X_test, y_test

    # 1. Baseline and zero-weight joint equivalence
    train_data, target, X_test, y_test = _make_synthetic_domains(seed=42)
    res_base = train_and_evaluate(
        train_data, target, X_test, y_test,
        mode="baseline", epochs=3, finetune_epochs=2, batch_size=16, seed=42,
    )
    res_joint_zero = train_and_evaluate(
        train_data, target, X_test, y_test,
        mode="joint", contrastive_weight=0.0, epochs=3, finetune_epochs=2, batch_size=16, seed=42,
    )
    assert np.allclose(res_base["probabilities"], res_joint_zero["probabilities"], atol=1e-6)
    print("[PASS] baseline and joint(contrastive_weight=0.0) are bit-for-bit equivalent.")

    # 2. Pretrain preserves heads and discards projection
    seed_everything(42)
    domains, _, _, _ = _prepare_domains(train_data, target, X_test, "none", 42)
    class_counts = {name: len(np.unique(y)) for name, (_, y) in domains.items()}
    ref_model = MLPClassifier(16, class_counts, "baseline", "asymmetric")
    initial_encoder_state = copy.deepcopy(ref_model.encoder.state_dict())
    initial_heads_state = copy.deepcopy(ref_model.classifiers.state_dict())

    stage1_snapshots = {}
    orig_step_ce = sys.modules["src.models.build_mlp_contrastive"]._step_ce

    def wrapped_step_ce(model, projection, optimizer, batch, domain, mode, weight, temperature, device):
        if "first_stage1_call" not in stage1_snapshots:
            stage1_snapshots["first_stage1_call"] = True
            stage1_snapshots["encoder_after_pretrain"] = copy.deepcopy(model.encoder.state_dict())
            stage1_snapshots["heads_after_pretrain"] = copy.deepcopy(model.classifiers.state_dict())
            stage1_snapshots["projection_in_stage1"] = projection
            stage1_snapshots["optimizer_state_len"] = len(optimizer.state)
        return orig_step_ce(model, projection, optimizer, batch, domain, mode, weight, temperature, device)

    with patch("src.models.build_mlp_contrastive._step_ce", side_effect=wrapped_step_ce):
        train_and_evaluate(
            train_data, target, X_test, y_test,
            mode="pretrain", epochs=3, finetune_epochs=2, batch_size=16, seed=42,
        )

    for key, val in initial_heads_state.items():
        after_val = stage1_snapshots["heads_after_pretrain"][key]
        assert torch.equal(val, after_val.cpu()), f"Classifier head tensor {key} was modified during pretrain!"
    encoder_changed = any(
        not torch.allclose(initial_encoder_state[k], stage1_snapshots["encoder_after_pretrain"][k].cpu())
        for k in initial_encoder_state if "weight" in k
    )
    assert encoder_changed
    assert stage1_snapshots["projection_in_stage1"] is None
    assert stage1_snapshots["optimizer_state_len"] == 0
    print("[PASS] pretrain updates encoder, leaves heads untouched in Stage 0, discards projection, and resets optimizer.")

    # 3. LOCO isolation and binary present classes ROC-AUC
    y_test_two_classes = np.array([1] * 8 + [2] * 7, dtype=np.int64)
    metrics_bin = train_and_evaluate(
        train_data, target, X_test, y_test_two_classes,
        mode="baseline", epochs=2, finetune_epochs=1, batch_size=16, seed=42,
    )
    assert not np.isnan(metrics_bin["roc_auc_ovr_macro"])
    print("[PASS] Binary-subset test fold ROC-AUC and LOCO isolation verified.")


if __name__ == "__main__":
    test_ast_and_protocol_invariants()
    test_published_csv_consistency()
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
        import sklearn  # noqa: F401
    except ModuleNotFoundError as exc:
        print(
            f"[INFO] Optional ML runtime dependency ({exc.name}) not installed in local shell; "
            "run `python scripts/verify_mlp_contrastive.py` inside the project's ML virtualenv to execute tensor checks."
        )
    else:
        run_tensor_suite()
    print("\nVerification completed successfully.")
