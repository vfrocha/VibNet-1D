"""
Unit test suite for the independent Scratch vs. Transfer Learning benchmark
(`src.experiments.run_scratch_vs_tl_benchmark`).

Verifies all 10 mandatory scientific and engineering invariants specified in Section 13:
  1. Scratch and TL use identical LOCO test sample IDs (`split_signature` and `predictions.csv` sample IDs).
  2. Target test samples never enter pretraining (`prepare_scaled_source_pool` and target test overlap guard).
  3. Preprocessing (`StandardScaler`) is fitted only on permitted training data.
  4. SMOTE is applied only to target training data.
  5. Linear Probing does not update encoder parameters or buffers (`backbone_param_max_abs_diff == 0.0`
     and `backbone_buffer_max_abs_diff == 0.0` for both `MLP` and `TabNet`).
  6. Fine-Tuning does update the intended backbone and head parameters (`backbone_param_max_abs_diff > 0`
     for both `MLP` and `TabNet`).
  7. Class mappings are consistent across train, validation, and test (including missing classes).
  8. Predictions align correctly by `sample_id` (`validate_predictions_integrity` detects duplicates,
     missing samples, or misaligned probabilities).
  9. All expected metric values are finite or explicitly marked undefined (`NaN` + `roc_auc_valid=False`,
     never `0.0`).
 10. Repeated runs create distinct `run_TIMESTAMP` directories and never overwrite previous benchmark
     results (while `--resume-dir` resumes cleanly from checkpoints).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.experiments.run_scratch_vs_tl_benchmark import (
    ARCHITECTURES,
    BALANCE_STRATEGIES,
    FEATURE_DIMENSION,
    METHODS,
    PREPROCESSING_VARIANT,
    apply_smote_with_audit,
    compute_fold_metrics,
    compute_preprocessing_signature,
    compute_sample_split_signature,
    load_benchmark_config,
    prepare_scaled_source_pool,
    pretrain_mlp_on_sources,
    pretrain_tabnet_on_sources,
    project_predictions_to_full_vocabulary,
    run_scratch_vs_tl_benchmark,
    scale_loco_fold_strict,
    train_eval_mlp_variant,
    train_eval_tabnet_variant,
    validate_predictions_integrity,
)


def _create_synthetic_window_tree(
    base_dir: Path,
    datasets_spec: Dict[str, Tuple[List[str], List[str], int]],
    seed: int = 42,
) -> Path:
    """
    Create a temporary `detrended_rms/` directory containing `.npy` window files
    for target and source datasets so `run_scratch_vs_tl_benchmark` can be tested
    end-to-end with a fast 141-dim feature extractor stub.
    """
    rng = np.random.default_rng(seed)
    detrended_dir = base_dir / PREPROCESSING_VARIANT
    detrended_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, (conds, classes, n_per_cell) in datasets_spec.items():
        for c_idx, cond in enumerate(conds):
            for cls_idx, cls_name in enumerate(classes):
                cell_dir = detrended_dir / ds_name / cond / cls_name
                cell_dir.mkdir(parents=True, exist_ok=True)
                for s_idx in range(n_per_cell):
                    # Encode class and condition information into the signal so features are informative
                    sig = rng.normal(
                        loc=float(cls_idx + 1) * 0.8 + float(c_idx) * 0.1,
                        scale=float(cls_idx + 1) * 0.3,
                        size=256,
                    ).astype(np.float32)
                    np.save(cell_dir / f"win_{s_idx:03d}.npy", sig)

    return detrended_dir


def _fast_141_feature_extractor(X_raw_list: List[np.ndarray], fs: int) -> np.ndarray:
    """Deterministic 141-dimensional batch feature extractor for fast unit tests."""
    rows: List[np.ndarray] = []
    for sig in X_raw_list:
        sig_f = np.asarray(sig, dtype=np.float64).ravel()
        mean_v = float(np.mean(sig_f))
        std_v = float(np.std(sig_f)) + 1e-6
        rms_v = float(np.sqrt(np.mean(sig_f**2)))
        feats = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
        feats[0] = mean_v
        feats[1] = std_v
        feats[2] = rms_v
        for i in range(3, FEATURE_DIMENSION):
            idx = (i * 7) % len(sig_f)
            feats[i] = float(sig_f[idx]) * 0.5 + mean_v * 0.5
        rows.append(feats)
    return np.stack(rows, axis=0)


def _make_synthetic_source_pool(
    n_samples: int = 48,
    feature_dim: int = FEATURE_DIMENSION,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]]:
    """Create a synthetic loaded_sources dictionary for two source domains."""
    rng = np.random.default_rng(seed)
    loaded = {}
    for s_idx, (s_name, n_cls) in enumerate([("UORED", 3), ("HUST", 4)]):
        X = rng.normal(loc=float(s_idx), scale=1.0, size=(n_samples, feature_dim)).astype(np.float32)
        y = np.array([f"SrcCls_{i % n_cls}" for i in range(n_samples)], dtype=str)
        # Distinguish classes in feature space
        for i in range(n_samples):
            c = i % n_cls
            X[i, c * 4 : (c + 1) * 4] += 2.5
        conds = np.array([f"SrcCond_{i % 2}" for i in range(n_samples)], dtype=str)
        ids = np.array([f"{s_name}/SrcCond_{i % 2}/SrcCls_{i % n_cls}/w_{i:03d}.npy" for i in range(n_samples)], dtype=str)
        loaded[s_name] = (X, y, conds, ids, {"n_samples": n_samples, "fs_hz": 42000})
    return loaded


class TestScratchVsTLBenchmark(unittest.TestCase):
    """Comprehensive unit tests for all 10 requirements in Section 13."""

    def test_01_scratch_and_tl_use_identical_loco_test_sample_ids(self) -> None:
        """1. Scratch, Fine-Tuning, and Linear Probing use identical LOCO test sample IDs."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = _create_synthetic_window_tree(
                tmp_path / "data",
                {
                    "CWRU_12k": (["0hp_1797rpm", "1hp_1772rpm", "2hp_1750rpm", "3hp_1730rpm"], ["Normal", "Inner", "Outer"], 6),
                    "UORED": (["S1", "S2"], ["C0", "C1", "C2"], 8),
                    "HUST": (["S1", "S2"], ["C0", "C1", "C2"], 8),
                    "Mechanical_Gear": (["S1", "S2"], ["C0", "C1"], 8),
                    "Electric_Motor": (["S1", "S2"], ["C0", "C1"], 8),
                },
            )
            out_root = tmp_path / "results"
            run_dir = run_scratch_vs_tl_benchmark(
                data_root=data_root,
                output_root=out_root,
                targets_override=["CWRU_12k"],
                architectures_override=["MLP", "TabNet"],
                methods_override=["Scratch", "Fine-Tuning", "Linear Probing"],
                balance_override=["none"],
                seeds_override=[42],
                max_folds=2,
                epochs_override=2,
                pretrain_epochs_override=2,
                batch_size_override=16,
                feature_extractor=_fast_141_feature_extractor,
                enforce_expected_folds=True,
                is_synthetic_smoke=True,
            )

            df_res = pd.read_csv(run_dir / "results.csv")
            df_preds = pd.read_csv(run_dir / "predictions.csv")
            df_man = pd.read_csv(run_dir / "fold_manifest.csv")

            self.assertTrue((df_res["status"] == "OK").all())
            # Every (fold_id, seed, balance_strategy) has identical split_signature across all 6 (arch, method) pairs
            for (fold_id, seed, bal), grp in df_res.groupby(["fold_id", "seed", "balance_strategy"]):
                self.assertEqual(len(grp), 6)
                self.assertEqual(grp["split_signature"].nunique(), 1)
                self.assertEqual(grp["preprocessing_signature"].nunique(), 1)

            # Check predictions sample_id sets match identically across Scratch, Fine-Tuning, Linear Probing
            val_report = validate_predictions_integrity(df_preds, df_man)
            self.assertTrue(val_report["valid"])

    def test_02_target_test_samples_never_enter_pretraining(self) -> None:
        """2. Target test samples never enter source pretraining (`prepare_scaled_source_pool` guard)."""
        loaded_sources = _make_synthetic_source_pool(n_samples=20)
        # Disjoint target sample IDs -> succeeds
        prepared = prepare_scaled_source_pool(
            source_names=["UORED", "HUST"],
            loaded_sources=loaded_sources,
            target_sample_ids=["CWRU_12k/Cond0/Normal/w_001.npy"],
        )
        self.assertEqual(set(prepared.keys()), {"UORED", "HUST"})

        # Overlapping target sample ID -> raises ValueError immediately
        leaked_id = loaded_sources["UORED"][3][0]
        with self.assertRaises(ValueError):
            prepare_scaled_source_pool(
                source_names=["UORED", "HUST"],
                loaded_sources=loaded_sources,
                target_sample_ids=[str(leaked_id)],
            )

        # Target dataset accidentally listed in source_names -> raises ValueError
        loaded_with_target = dict(loaded_sources)
        loaded_with_target["CWRU_12k"] = loaded_sources["UORED"]
        with self.assertRaises(ValueError):
            prepare_scaled_source_pool(
                source_names=["UORED", "CWRU_12k"],
                loaded_sources=loaded_with_target,
                target_sample_ids=[],
            )

    def test_03_preprocessing_fitted_only_on_permitted_training_data(self) -> None:
        """3. StandardScaler is fitted strictly on target training data, unaffected by test data."""
        rng = np.random.default_rng(101)
        X_train = rng.normal(loc=3.0, scale=2.0, size=(40, FEATURE_DIMENSION)).astype(np.float32)
        X_test_a = rng.normal(loc=-50.0, scale=10.0, size=(12, FEATURE_DIMENSION)).astype(np.float32)
        X_test_b = rng.normal(loc=5000.0, scale=999.0, size=(12, FEATURE_DIMENSION)).astype(np.float32)

        X_tr_a, _, scaler_a = scale_loco_fold_strict(X_train, X_test_a)
        X_tr_b, _, scaler_b = scale_loco_fold_strict(X_train, X_test_b)

        np.testing.assert_allclose(scaler_a.mean_, np.mean(X_train, axis=0), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(scaler_a.mean_, scaler_b.mean_, rtol=0, atol=0)
        np.testing.assert_allclose(scaler_a.scale_, scaler_b.scale_, rtol=0, atol=0)
        np.testing.assert_allclose(X_tr_a, X_tr_b, rtol=0, atol=0)

    def test_04_smote_applied_only_to_training_data(self) -> None:
        """4. SMOTE is applied strictly to training data and never alters test samples."""
        rng = np.random.default_rng(202)
        X_train = rng.normal(size=(26, FEATURE_DIMENSION)).astype(np.float32)
        y_train = np.array([0] * 20 + [1] * 6, dtype=np.int64)
        X_test = rng.normal(size=(10, FEATURE_DIMENSION)).astype(np.float32)
        X_test_snapshot = X_test.copy()

        X_bal, y_bal, audit = apply_smote_with_audit(
            X_train_scaled=X_train,
            y_train_encoded=y_train,
            class_names=["Normal", "Fault"],
            strategy="smote",
            seed=42,
        )
        self.assertIsNotNone(X_bal)
        self.assertEqual(audit["balance_status"], "OK")
        self.assertTrue(audit["smote_params"]["applied"])
        self.assertEqual(len(y_bal), 40)
        np.testing.assert_array_equal(X_test, X_test_snapshot)

    def test_05_linear_probing_does_not_update_encoder_parameters(self) -> None:
        """5. Linear Probing freezes encoder parameters and buffers (`max_abs_diff == 0.0`) for MLP and TabNet."""
        loaded_sources = _make_synthetic_source_pool(n_samples=40, seed=42)
        prepared_sources = prepare_scaled_source_pool(
            source_names=["UORED", "HUST"],
            loaded_sources=loaded_sources,
            target_sample_ids=[],
        )

        # Target data
        rng = np.random.default_rng(42)
        X_train_raw = rng.normal(size=(48, FEATURE_DIMENSION)).astype(np.float32)
        y_train_raw = np.array([i % 3 for i in range(48)], dtype=np.int64)
        conds_train_raw = np.array([f"Cond_{i % 3}" for i in range(48)], dtype=str)
        X_test_raw = rng.normal(size=(16, FEATURE_DIMENSION)).astype(np.float32)
        X_train_scaled, X_test_scaled, _ = scale_loco_fold_strict(X_train_raw, X_test_raw)

        # 5a. MLP Linear Probing
        mlp_pre = pretrain_mlp_on_sources(
            prepared_sources=prepared_sources,
            seed=42,
            pretrain_epochs=3,
            batch_size=16,
        )
        _, probs_mlp, meta_mlp = train_eval_mlp_variant(
            method="Linear Probing",
            X_train_bal=X_train_scaled,
            y_train_bal=y_train_raw,
            X_test_scaled=X_test_scaled,
            n_total_classes=3,
            seed=42,
            pretrained_mlp_state=mlp_pre,
            epochs=4,
            batch_size=16,
        )
        self.assertTrue(meta_mlp["backbone_frozen"])
        self.assertEqual(meta_mlp["backbone_param_max_abs_diff"], 0.0)
        self.assertEqual(meta_mlp["backbone_buffer_max_abs_diff"], 0.0)
        self.assertEqual(probs_mlp.shape, (16, 3))

        # 5b. TabNet Linear Probing
        tabnet_pre = pretrain_tabnet_on_sources(
            prepared_sources=prepared_sources,
            seed=42,
            pretrain_epochs=3,
            batch_size=16,
        )
        _, probs_tab, meta_tab = train_eval_tabnet_variant(
            method="Linear Probing",
            X_train_raw=X_train_raw,
            y_train_raw=y_train_raw,
            conds_train_raw=conds_train_raw,
            X_train_bal=X_train_scaled,
            y_train_bal=y_train_raw,
            X_test_scaled=X_test_scaled,
            class_names=["C0", "C1", "C2"],
            n_total_classes=3,
            balance_strategy="none",
            seed=42,
            pretrained_tabnet=tabnet_pre,
            max_epochs=4,
            patience=3,
            batch_size=16,
        )
        self.assertTrue(meta_tab["backbone_frozen"])
        self.assertEqual(meta_tab["backbone_param_max_abs_diff"], 0.0)
        self.assertEqual(meta_tab["backbone_buffer_max_abs_diff"], 0.0)
        self.assertEqual(probs_tab.shape, (16, 3))

    def test_06_fine_tuning_updates_intended_parameters(self) -> None:
        """6. Fine-Tuning updates encoder parameters (`backbone_param_max_abs_diff > 0`) for MLP and TabNet."""
        loaded_sources = _make_synthetic_source_pool(n_samples=40, seed=77)
        prepared_sources = prepare_scaled_source_pool(
            source_names=["UORED", "HUST"],
            loaded_sources=loaded_sources,
            target_sample_ids=[],
        )

        rng = np.random.default_rng(77)
        X_train_raw = rng.normal(size=(48, FEATURE_DIMENSION)).astype(np.float32)
        y_train_raw = np.array([i % 3 for i in range(48)], dtype=np.int64)
        for i in range(48):
            c = y_train_raw[i]
            X_train_raw[i, c * 5 : (c + 1) * 5] += 3.0
        conds_train_raw = np.array([f"Cond_{i % 3}" for i in range(48)], dtype=str)
        X_test_raw = rng.normal(size=(16, FEATURE_DIMENSION)).astype(np.float32)
        X_train_scaled, X_test_scaled, _ = scale_loco_fold_strict(X_train_raw, X_test_raw)

        # 6a. MLP Fine-Tuning
        mlp_pre = pretrain_mlp_on_sources(
            prepared_sources=prepared_sources,
            seed=77,
            pretrain_epochs=3,
            batch_size=16,
        )
        _, _, meta_mlp_ft = train_eval_mlp_variant(
            method="Fine-Tuning",
            X_train_bal=X_train_scaled,
            y_train_bal=y_train_raw,
            X_test_scaled=X_test_scaled,
            n_total_classes=3,
            seed=77,
            pretrained_mlp_state=mlp_pre,
            epochs=4,
            batch_size=16,
        )
        self.assertFalse(meta_mlp_ft["backbone_frozen"])
        self.assertGreater(meta_mlp_ft["backbone_param_max_abs_diff"], 0.0)

        # 6b. TabNet Fine-Tuning
        tabnet_pre = pretrain_tabnet_on_sources(
            prepared_sources=prepared_sources,
            seed=77,
            pretrain_epochs=3,
            batch_size=16,
        )
        _, _, meta_tab_ft = train_eval_tabnet_variant(
            method="Fine-Tuning",
            X_train_raw=X_train_raw,
            y_train_raw=y_train_raw,
            conds_train_raw=conds_train_raw,
            X_train_bal=X_train_scaled,
            y_train_bal=y_train_raw,
            X_test_scaled=X_test_scaled,
            class_names=["C0", "C1", "C2"],
            n_total_classes=3,
            balance_strategy="none",
            seed=77,
            pretrained_tabnet=tabnet_pre,
            max_epochs=4,
            patience=3,
            batch_size=16,
        )
        self.assertFalse(meta_tab_ft["backbone_frozen"])
        self.assertGreater(meta_tab_ft["backbone_param_max_abs_diff"], 0.0)

    def test_07_class_mappings_are_consistent(self) -> None:
        """7. Class mappings and probability projections remain consistent even when classes are absent in train."""
        # Full dataset vocabulary has 4 classes [0, 1, 2, 3], but training fold only has [0, 2, 3]
        raw_preds = np.array([0, 1, 2, 0], dtype=np.int64)
        raw_probs = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.6, 0.3, 0.1],
            ],
            dtype=np.float64,
        )
        model_classes = np.array([0, 2, 3], dtype=np.int64)
        proj_preds, proj_probs = project_predictions_to_full_vocabulary(
            preds_local=raw_preds,
            probs_local=raw_probs,
            present_train_classes=model_classes,
            n_total_classes=4,
        )
        np.testing.assert_array_equal(proj_preds, np.array([0, 2, 3, 0], dtype=np.int64))
        self.assertEqual(proj_probs.shape, (4, 4))
        np.testing.assert_allclose(proj_probs[:, 1], 0.0)
        np.testing.assert_allclose(np.sum(proj_probs, axis=1), 1.0, rtol=1e-6)

    def test_08_predictions_align_correctly_by_sample_id(self) -> None:
        """8. `validate_predictions_integrity` verifies alignment and catches duplicates or missing sample IDs."""
        df_man = pd.DataFrame(
            [
                {
                    "target_dataset": "CWRU_12k",
                    "fold_id": 0,
                    "condition": "0hp",
                    "seed": 42,
                    "balance_strategy": "none",
                    "test_sample_ids": json.dumps(["s1", "s2"]),
                }
            ]
        )
        rows = []
        for arch in ("MLP", "TabNet"):
            for meth in ("Scratch", "Fine-Tuning", "Linear Probing"):
                for sid, yt in [("s1", 0), ("s2", 1)]:
                    rows.append(
                        {
                            "target_dataset": "CWRU_12k",
                            "fold_id": 0,
                            "condition": "0hp",
                            "sample_id": sid,
                            "y_true": yt,
                            "y_pred": yt,
                            "probabilities": json.dumps([0.8, 0.2] if yt == 0 else [0.1, 0.9]),
                            "class_order": json.dumps(["Normal", "Fault"]),
                            "architecture": arch,
                            "method": meth,
                            "balance_strategy": "none",
                            "seed": 42,
                        }
                    )
        df_preds = pd.DataFrame(rows)
        res = validate_predictions_integrity(df_preds, df_man)
        self.assertTrue(res["valid"])
        self.assertEqual(res["checked_rows"], 12)

        # Duplicate sample row -> raises ValueError
        df_dup = pd.concat([df_preds, df_preds.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            validate_predictions_integrity(df_dup, df_man)

        # Mismatched y_true across methods -> raises ValueError
        df_bad_y = df_preds.copy()
        df_bad_y.loc[0, "y_true"] = 99
        with self.assertRaises(ValueError):
            validate_predictions_integrity(df_bad_y, df_man)

    def test_09_metrics_are_finite_or_explicitly_nan_when_undefined(self) -> None:
        """9. Metrics are finite when defined and explicitly NaN (never 0.0) when ROC-AUC is undefined."""
        # Case A: Single class in y_true (e.g., UOEMD single-class fold) -> ROC_AUC is NaN + ROC_AUC_Valid=False
        y_true_single = np.array([0, 0, 0, 0], dtype=np.int64)
        y_pred_single = np.array([0, 0, 1, 0], dtype=np.int64)
        probs_single = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.8, 0.10, 0.10],
                [0.2, 0.70, 0.10],
                [0.85, 0.10, 0.05],
            ],
            dtype=np.float64,
        )
        m_single = compute_fold_metrics(y_true_single, y_pred_single, probs_single)
        self.assertTrue(math.isfinite(m_single["Accuracy"]))
        self.assertTrue(math.isfinite(m_single["Balanced_Accuracy"]))
        self.assertTrue(math.isfinite(m_single["Macro_F1"]))
        self.assertTrue(math.isnan(m_single["ROC_AUC"]))
        self.assertFalse(m_single["ROC_AUC_Valid"])
        self.assertNotEqual(m_single["ROC_AUC"], 0.0)

        # Case B: Subset of classes in y_true (2 of 4 classes present) -> valid OvR ROC_AUC
        y_true_sub = np.array([0, 0, 2, 2], dtype=np.int64)
        y_pred_sub = np.array([0, 0, 2, 2], dtype=np.int64)
        probs_sub = np.array(
            [
                [0.8, 0.05, 0.1, 0.05],
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.05, 0.8, 0.05],
                [0.1, 0.1, 0.7, 0.1],
            ],
            dtype=np.float64,
        )
        m_sub = compute_fold_metrics(y_true_sub, y_pred_sub, probs_sub)
        self.assertTrue(m_sub["ROC_AUC_Valid"])
        self.assertAlmostEqual(m_sub["ROC_AUC"], 1.0, places=5)

    def test_10_repeated_runs_never_overwrite_and_resume_works(self) -> None:
        """10. Repeated runs create distinct `run_TIMESTAMP` directories and `--resume-dir` resumes cleanly."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = _create_synthetic_window_tree(
                tmp_path / "data",
                {
                    "CWRU_12k": (["0hp_1797rpm", "1hp_1772rpm", "2hp_1750rpm", "3hp_1730rpm"], ["Normal", "Fault"], 5),
                    "UORED": (["S1", "S2"], ["C0", "C1"], 6),
                    "HUST": (["S1", "S2"], ["C0", "C1"], 6),
                    "Mechanical_Gear": (["S1", "S2"], ["C0", "C1"], 6),
                    "Electric_Motor": (["S1", "S2"], ["C0", "C1"], 6),
                },
            )
            out_root = tmp_path / "results"

            run_dir_1 = run_scratch_vs_tl_benchmark(
                data_root=data_root,
                output_root=out_root,
                targets_override=["CWRU_12k"],
                architectures_override=["MLP"],
                methods_override=["Scratch", "Linear Probing"],
                balance_override=["none"],
                seeds_override=[42],
                max_folds=1,
                epochs_override=2,
                pretrain_epochs_override=2,
                batch_size_override=16,
                feature_extractor=_fast_141_feature_extractor,
                enforce_expected_folds=True,
                is_synthetic_smoke=True,
            )
            run_dir_2 = run_scratch_vs_tl_benchmark(
                data_root=data_root,
                output_root=out_root,
                targets_override=["CWRU_12k"],
                architectures_override=["MLP"],
                methods_override=["Scratch", "Linear Probing"],
                balance_override=["none"],
                seeds_override=[42],
                max_folds=1,
                epochs_override=2,
                pretrain_epochs_override=2,
                batch_size_override=16,
                feature_extractor=_fast_141_feature_extractor,
                enforce_expected_folds=True,
                is_synthetic_smoke=True,
            )
            self.assertNotEqual(run_dir_1, run_dir_2)
            self.assertTrue(run_dir_1.exists())
            self.assertTrue(run_dir_2.exists())

            # Test resuming run_dir_1 adding Fine-Tuning to the same run directory
            resumed_dir = run_scratch_vs_tl_benchmark(
                data_root=data_root,
                output_root=out_root,
                resume_dir=run_dir_1,
                targets_override=["CWRU_12k"],
                architectures_override=["MLP"],
                methods_override=["Scratch", "Fine-Tuning", "Linear Probing"],
                balance_override=["none"],
                seeds_override=[42],
                max_folds=1,
                epochs_override=2,
                pretrain_epochs_override=2,
                batch_size_override=16,
                feature_extractor=_fast_141_feature_extractor,
                enforce_expected_folds=True,
                is_synthetic_smoke=True,
            )
            self.assertEqual(resumed_dir, run_dir_1)
            df_res = pd.read_csv(run_dir_1 / "results.csv")
            self.assertEqual(len(df_res), 3)
            self.assertEqual(set(df_res["method"]), {"Scratch", "Fine-Tuning", "Linear Probing"})

            # Verify all 13 required output files exist in run_dir_1
            expected_files = [
                "comparison_config.json",
                "environment.json",
                "git_metadata.json",
                "compatibility_audit.csv",
                "fold_manifest.csv",
                "results.csv",
                "predictions.csv",
                "summary_by_dataset.csv",
                "summary_by_architecture.csv",
                "summary_by_method.csv",
                "statistical_tests.csv",
                "rankings.csv",
                "report.md",
            ]
            for fname in expected_files:
                self.assertTrue((run_dir_1 / fname).exists(), f"Missing artifact: {fname}")

    def test_11_stage_d_pilot_validation_all_artifacts_and_statistics(self) -> None:
        """
        Stage D Pilot Validation:
        Covers 2 target datasets (CWRU_12k and UOEMD with single-class test fold),
        both architectures (MLP, TabNet), all 3 methods (Scratch, Fine-Tuning, Linear Probing),
        both balance strategies (none, smote), and 2 seeds (42, 123).
        Verifies all 13 output artifacts, parameter freezing checks, statistical tests, and rankings.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = _create_synthetic_window_tree(
                tmp_path / "data",
                {
                    "CWRU_12k": (
                        ["0hp_1797rpm", "1hp_1772rpm", "2hp_1750rpm", "3hp_1730rpm"],
                        ["Normal", "Inner", "Outer"],
                        4,
                    ),
                    "UOEMD": (
                        [f"U_{i:02d}" for i in range(16)],
                        ["Healthy", "FaultA", "FaultB"],
                        3,
                    ),
                    "UORED": (["S1", "S2"], ["C0", "C1", "C2"], 6),
                    "HUST": (["S1", "S2"], ["C0", "C1", "C2"], 6),
                    "Mechanical_Gear": (["S1", "S2"], ["C0", "C1"], 6),
                    "Electric_Motor": (["S1", "S2"], ["C0", "C1"], 6),
                },
            )
            # Make UOEMD condition U_00 single-class (only "Healthy") to test UOEMD single-class fold handling
            import shutil

            shutil.rmtree(data_root / "UOEMD" / "U_00" / "FaultA")
            shutil.rmtree(data_root / "UOEMD" / "U_00" / "FaultB")

            out_root = tmp_path / "results"
            run_dir = run_scratch_vs_tl_benchmark(
                data_root=data_root,
                output_root=out_root,
                targets_override=["CWRU_12k", "UOEMD"],
                architectures_override=["MLP", "TabNet"],
                methods_override=["Scratch", "Fine-Tuning", "Linear Probing"],
                balance_override=["none", "smote"],
                seeds_override=[42, 123],
                max_folds=2,
                epochs_override=2,
                pretrain_epochs_override=2,
                batch_size_override=16,
                feature_extractor=_fast_141_feature_extractor,
                enforce_expected_folds=True,
                is_synthetic_smoke=True,
            )

            df_res = pd.read_csv(run_dir / "results.csv")
            df_stats = pd.read_csv(run_dir / "statistical_tests.csv")
            df_ranks = pd.read_csv(run_dir / "rankings.csv")

            # 2 targets * 2 folds * 2 seeds * 2 balances * 2 architectures * 3 methods = 96 rows
            self.assertEqual(len(df_res), 96)
            self.assertTrue((df_res["status"] == "OK").all())

            # Linear Probing rows have backbone_frozen == True and diff == 0.0
            lp_rows = df_res[df_res["method"] == "Linear Probing"]
            self.assertTrue((lp_rows["backbone_frozen"] == True).all())  # noqa: E712
            np.testing.assert_allclose(lp_rows["backbone_param_max_abs_diff"], 0.0)
            np.testing.assert_allclose(lp_rows["backbone_buffer_max_abs_diff"], 0.0)

            # Fine-Tuning rows have backbone_frozen == False and diff > 0.0
            ft_rows = df_res[df_res["method"] == "Fine-Tuning"]
            self.assertTrue((ft_rows["backbone_frozen"] == False).all())  # noqa: E712
            self.assertTrue((ft_rows["backbone_param_max_abs_diff"] > 0.0).all())

            # UOEMD fold U_00 has 1 test class -> roc_auc_valid == False and roc_auc is NaN
            u00_rows = df_res[(df_res["target_dataset"] == "UOEMD") & (df_res["condition"] == "U_00")]
            self.assertTrue((u00_rows["test_class_count"] == 1).all())
            self.assertTrue((u00_rows["roc_auc_valid"] == False).all())  # noqa: E712
            self.assertTrue(u00_rows["roc_auc"].isna().all())

            # Statistical tests and rankings are populated
            self.assertFalse(df_stats.empty)
            self.assertFalse(df_ranks.empty)
            self.assertIn("p_value_holm", df_stats.columns)
            self.assertIn("mean_rank_across_datasets", df_ranks["scope"].values)


if __name__ == "__main__":
    unittest.main()
