"""
Unit test suite for the leakage-free, paired Leave-One-Condition-Out (LOCO)
supervised baseline benchmark (`src.experiments.run_baseline_loco_v2`).

Verifies all 10 protocol and anti-leakage invariants specified for `baseline_loco_v2`:
  1. Train and test conditions are strictly disjoint.
  2. The held-out test condition never appears in training or internal validation.
  3. StandardScaler fitting uses only target training data (zero test leakage).
  4. SMOTE never receives test samples and explicitly flags `< 2`-sample classes.
  5. TabNet Scratch never receives test data in `eval_set`.
  6. All 5 models receive identical training and test sample identifiers.
  7. Label encoding remains consistent across train, validation, and test.
  8. Fold counts match expected canonical dataset metadata (4+12+30+16+4 = 66).
  9. Metrics handle missing test classes properly (ROC_AUC is NaN + ROC_AUC_Valid=False
     when < 2 test classes, valid subset OvR when >= 2 test classes, never 0.0).
 10. Failed models remain explicitly identified in `results.csv` with NaN metrics.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.experiments.run_baseline_loco_v2 import (
    BALANCE_STRATEGIES,
    CANONICAL_MODELS,
    EXPECTED_LOCO_FOLDS,
    EXPECTED_TOTAL_FOLDS,
    FEATURE_DIMENSION,
    REQUIRED_RESULTS_COLUMNS,
    SEEDS,
    TARGET_DATASETS,
    apply_smote_with_audit,
    compute_fold_metrics,
    create_internal_validation_split,
    evaluate_single_model_on_fold,
    generate_summaries,
    parse_args,
    project_predictions_to_full_vocabulary,
    run_benchmark,
    run_dataset_loco_evaluations,
    scale_loco_fold_strict,
    train_eval_tabnet_scratch,
    validate_data_root_is_detrended_rms,
    verify_canonical_fold_metadata,
)
from src.vibnet.evaluation.protocols import assert_no_condition_leakage, iter_loco_folds


def _make_synthetic_loco_dataset(
    n_conditions: int = 4,
    n_classes: int = 4,
    samples_per_cond_class: int = 8,
    feature_dim: int = FEATURE_DIMENSION,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a deterministic synthetic 141-feature dataset with operating conditions."""
    rng = np.random.default_rng(seed)
    X_rows: List[np.ndarray] = []
    y_rows: List[str] = []
    cond_rows: List[str] = []
    id_rows: List[str] = []

    for c_idx in range(n_conditions):
        cond_name = f"Cond_{c_idx:02d}"
        cond_shift = float(c_idx) * 0.15
        for cls_idx in range(n_classes):
            cls_name = f"Class_{cls_idx}"
            center = np.zeros(feature_dim, dtype=np.float32)
            center[cls_idx * 5 : (cls_idx + 1) * 5] = 2.5
            for s_idx in range(samples_per_cond_class):
                vec = center + cond_shift + rng.normal(0.0, 0.6, size=feature_dim).astype(np.float32)
                X_rows.append(vec)
                y_rows.append(cls_name)
                cond_rows.append(cond_name)
                id_rows.append(f"SYNTH/{cond_name}/{cls_name}/win_{s_idx:03d}.npy")

    return (
        np.stack(X_rows, axis=0),
        np.asarray(y_rows, dtype=str),
        np.asarray(cond_rows, dtype=str),
        np.asarray(id_rows, dtype=str),
    )


class TestBaselineLOCOV2Protocol(unittest.TestCase):
    """Comprehensive unit tests for `baseline_loco_v2` scientific protocol."""

    def test_01_train_and_test_conditions_are_disjoint(self) -> None:
        """1. Train and test conditions and sample indices are strictly disjoint."""
        _, _, conds, sample_ids = _make_synthetic_loco_dataset(n_conditions=4, n_classes=3)
        folds = list(iter_loco_folds(conds.tolist()))
        self.assertEqual(len(folds), 4)

        for fold in folds:
            train_conds = {conds[i] for i in fold.train_indices}
            test_conds = {conds[i] for i in fold.test_indices}
            self.assertEqual(len(test_conds), 1)
            self.assertEqual(test_conds, {fold.test_condition})
            self.assertTrue(train_conds.isdisjoint(test_conds))

            train_ids = {sample_ids[i] for i in fold.train_indices}
            test_ids = {sample_ids[i] for i in fold.test_indices}
            self.assertTrue(train_ids.isdisjoint(test_ids))
            assert_no_condition_leakage(conds.tolist(), fold.train_indices, fold.test_indices)

    def test_02_held_out_condition_never_in_training_or_internal_validation(self) -> None:
        """2. Held-out test condition never appears in training or internal validation."""
        X, y_str, conds, _ = _make_synthetic_loco_dataset(n_conditions=4, n_classes=4)
        unique_classes = sorted(np.unique(y_str).tolist())
        y_enc = np.searchsorted(unique_classes, y_str)

        for held_out in sorted(np.unique(conds).tolist()):
            test_mask = conds == held_out
            train_mask = ~test_mask
            X_train = X[train_mask]
            y_train = y_enc[train_mask]
            conds_train = conds[train_mask]

            self.assertNotIn(held_out, set(conds_train.tolist()))

            inner_tr_idx, inner_val_idx, split_mode = create_internal_validation_split(
                X_train_raw=X_train,
                y_train=y_train,
                conds_train=conds_train,
                seed=42,
            )
            self.assertEqual(split_mode, "grouped_condition")
            self.assertTrue(set(inner_tr_idx.tolist()).isdisjoint(set(inner_val_idx.tolist())))

            inner_tr_conds = set(conds_train[inner_tr_idx].tolist())
            inner_val_conds = set(conds_train[inner_val_idx].tolist())
            self.assertNotIn(held_out, inner_tr_conds)
            self.assertNotIn(held_out, inner_val_conds)
            self.assertTrue(inner_tr_conds.isdisjoint(inner_val_conds))

    def test_03_scaler_fitting_uses_only_training_data(self) -> None:
        """3. StandardScaler is fitted exclusively on training data without test leakage."""
        rng = np.random.default_rng(123)
        X_train = rng.normal(loc=2.0, scale=3.0, size=(40, 141)).astype(np.float32)
        # Extreme outlier values in X_test that would shift mean/std if leaked
        X_test = np.full((15, 141), fill_value=10000.0, dtype=np.float32)

        X_tr_s, X_te_s, scaler = scale_loco_fold_strict(X_train, X_test)

        expected_mean = X_train.mean(axis=0)
        expected_std = X_train.std(axis=0, ddof=0)

        np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(scaler.scale_, expected_std, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(X_tr_s.mean(axis=0), np.zeros(141), atol=1e-5)
        self.assertTrue(np.all(X_te_s > 1000.0))

    def test_04_smote_never_receives_test_samples_and_audits_min_samples(self) -> None:
        """4. SMOTE operates strictly on scaled training data and fails explicitly if < 2 samples."""
        rng = np.random.default_rng(42)
        # Confine training features to [-1, 1]; test features to [500, 600]
        X_train = rng.uniform(-1.0, 1.0, size=(18, 141)).astype(np.float32)
        y_train = np.array([0] * 10 + [1] * 5 + [2] * 3, dtype=np.int64)
        class_names = ["Class_0", "Class_1", "Class_2"]

        X_bal, y_bal, audit = apply_smote_with_audit(
            X_train_scaled=X_train,
            y_train_encoded=y_train,
            class_names=class_names,
            strategy="smote",
            seed=42,
        )
        self.assertIsNotNone(X_bal)
        self.assertIsNotNone(y_bal)
        assert X_bal is not None and y_bal is not None
        self.assertEqual(len(X_bal), 30)
        self.assertEqual(audit["counts_after"], {"Class_0": 10, "Class_1": 10, "Class_2": 10})
        self.assertEqual(audit["smote_params"]["k_neighbors"], 2)  # min(5, 3 - 1)
        # Convex combinations of [-1, 1] must stay strictly within [-1, 1] (zero test contamination)
        self.assertLessEqual(float(np.max(X_bal)), 1.0 + 1e-6)
        self.assertGreaterEqual(float(np.min(X_bal)), -1.0 - 1e-6)

        # Case with < 2 samples in minority class -> must fail explicitly without faking SMOTE
        X_rare = rng.uniform(-1.0, 1.0, size=(12, 141)).astype(np.float32)
        y_rare = np.array([0] * 8 + [1] * 3 + [2] * 1, dtype=np.int64)
        X_fail, y_fail, audit_fail = apply_smote_with_audit(
            X_train_scaled=X_rare,
            y_train_encoded=y_rare,
            class_names=class_names,
            strategy="smote",
            seed=42,
        )
        self.assertIsNone(X_fail)
        self.assertIsNone(y_fail)
        self.assertEqual(audit_fail["balance_status"], "FAILED_SMOTE_MIN_SAMPLES")
        self.assertIn("Class_2", audit_fail["unbalanced_classes"])

    def test_05_tabnet_never_receives_test_data_as_eval_set(self) -> None:
        """5. TabNet Scratch uses only internal training-condition validation for early stopping."""
        X, y_str, conds, _ = _make_synthetic_loco_dataset(
            n_conditions=4, n_classes=3, samples_per_cond_class=6, seed=42
        )
        unique_classes = sorted(np.unique(y_str).tolist())
        y_enc = np.searchsorted(unique_classes, y_str).astype(np.int64)

        test_mask = conds == "Cond_00"
        train_mask = ~test_mask

        X_train_raw = X[train_mask]
        y_train_raw = y_enc[train_mask]
        conds_train_raw = conds[train_mask]
        # Give X_test distinct marker values >= 900.0 so any leakage into eval_set is unmistakable
        X_test_raw = np.full_like(X[test_mask], fill_value=999.0)

        X_train_scaled, X_test_scaled, _ = scale_loco_fold_strict(X_train_raw, X_test_raw)
        X_train_bal, y_train_bal, _ = apply_smote_with_audit(
            X_train_scaled, y_train_raw, unique_classes, strategy="smote", seed=42
        )
        assert X_train_bal is not None and y_train_bal is not None

        spy: List[Dict[str, Any]] = []
        preds, probs, meta = train_eval_tabnet_scratch(
            X_train_raw=X_train_raw,
            y_train_raw=y_train_raw,
            conds_train_raw=conds_train_raw,
            X_train_bal=X_train_bal,
            y_train_bal=y_train_bal,
            X_test_scaled=X_test_scaled,
            class_names=unique_classes,
            n_total_classes=len(unique_classes),
            balance_strategy="smote",
            seed=42,
            max_epochs=3,
            patience=2,
            batch_size=16,
            refit_full_train=True,
            eval_set_spy=spy,
        )

        self.assertEqual(len(spy), 2)
        self.assertEqual(spy[0]["stage"], "inner_validation")
        self.assertEqual(len(spy[0]["eval_set_X"]), 1)
        eval_x = spy[0]["eval_set_X"][0]
        # Verify eval_set comes from training conditions, never from the 999.0 test set
        self.assertLess(float(np.max(np.abs(eval_x))), 50.0)
        self.assertGreater(float(np.min(X_test_scaled)), 100.0)
        # Verify refit stage passes empty eval_set=[]
        self.assertEqual(spy[1]["stage"], "refit_full_train")
        self.assertEqual(spy[1]["eval_set_shapes"], [])
        self.assertEqual(preds.shape, (len(X_test_scaled),))
        self.assertEqual(probs.shape, (len(X_test_scaled), len(unique_classes)))
        self.assertEqual(meta["validation_split_mode"], "grouped_condition")

    def test_06_all_five_models_receive_identical_sample_identifiers(self) -> None:
        """6. All 5 models receive identical training and test sample identifiers per fold."""
        X, y_str, conds, sample_ids = _make_synthetic_loco_dataset(
            n_conditions=3, n_classes=3, samples_per_cond_class=6, seed=42
        )
        results = run_dataset_loco_evaluations(
            dataset_name="CWRU_12k",
            X_all=X,
            y_all_str=y_str,
            conds_all=conds,
            sample_ids_all=sample_ids,
            models=CANONICAL_MODELS,
            balance_strategies=["none", "smote"],
            seeds=[42],
            max_folds=1,
            epochs=2,
            batch_size=16,
            tabnet_patience=2,
        )
        df = pd.DataFrame(results)
        self.assertEqual(len(df), len(CANONICAL_MODELS) * 2)
        self.assertEqual(set(df["Model"].tolist()), set(CANONICAL_MODELS))
        # Every model on the fold must share the exact same Sample_Split_Signature
        signatures = df["Sample_Split_Signature"].unique().tolist()
        self.assertEqual(len(signatures), 1)
        self.assertTrue(all(df["Status"] == "OK"))

    def test_07_label_encoding_consistency_with_subset_classes(self) -> None:
        """7. Label encoding and probability columns remain consistent when classes are missing."""
        # Suppose total dataset has 4 classes (0, 1, 2, 3), but training fold only has [0, 2, 3]
        preds_local = np.array([0, 1, 2, 0], dtype=np.int64)
        probs_local = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
                [0.6, 0.2, 0.2],
            ],
            dtype=np.float64,
        )
        present_train = np.array([0, 2, 3], dtype=np.int64)
        preds_full, probs_full = project_predictions_to_full_vocabulary(
            preds_local=preds_local,
            probs_local=probs_local,
            present_train_classes=present_train,
            n_total_classes=4,
        )
        np.testing.assert_array_equal(preds_full, np.array([0, 2, 3, 0]))
        self.assertEqual(probs_full.shape, (4, 4))
        # Class 1 was absent in training -> column 1 must be 0.0
        np.testing.assert_allclose(probs_full[:, 1], np.zeros(4))
        np.testing.assert_allclose(probs_full.sum(axis=1), np.ones(4))

    def test_08_fold_counts_match_canonical_dataset_metadata(self) -> None:
        """8. Fold counts match CWRU_12k=4, CWRU_48k=12, HUST_Gearbox=30, UOEMD=16, PU=4 (Total=66)."""
        verified = verify_canonical_fold_metadata(TARGET_DATASETS)
        self.assertEqual(
            verified,
            {
                "CWRU_12k": 4,
                "CWRU_48k": 12,
                "HUST_Gearbox": 30,
                "UOEMD": 16,
                "PU": 4,
            },
        )
        self.assertEqual(sum(verified.values()), EXPECTED_TOTAL_FOLDS)
        self.assertEqual(EXPECTED_TOTAL_FOLDS, 66)

    def test_09_metrics_when_test_fold_lacks_classes(self) -> None:
        """9. Metrics handle missing test classes properly and never substitute 0.0 for invalid ROC-AUC."""
        # Case A: 2 of 3 classes present in test (like CWRU_48k Load_0HP_Sev_0.021)
        y_true_2 = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
        y_pred_2 = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)
        probs_3 = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.7, 0.1],
                [0.4, 0.5, 0.1],
                [0.1, 0.9, 0.0],
            ],
            dtype=np.float64,
        )
        m2 = compute_fold_metrics(y_true_2, y_pred_2, probs_3)
        self.assertTrue(m2["ROC_AUC_Valid"])
        self.assertFalse(math.isnan(m2["ROC_AUC"]))
        self.assertGreater(m2["ROC_AUC"], 0.8)
        self.assertEqual(m2["Test_Class_Count"], 2)

        # Case B: Only 1 class present in test -> ROC_AUC must be NaN and ROC_AUC_Valid=False (never 0.0)
        y_true_1 = np.array([0, 0, 0, 0], dtype=np.int64)
        y_pred_1 = np.array([0, 0, 1, 0], dtype=np.int64)
        m1 = compute_fold_metrics(y_true_1, y_pred_1, probs_3[:4])
        self.assertFalse(m1["ROC_AUC_Valid"])
        self.assertTrue(math.isnan(m1["ROC_AUC"]))
        self.assertNotEqual(m1["ROC_AUC"], 0.0)

    def test_10_failed_models_explicitly_identified_in_results_csv(self) -> None:
        """10. Failed models record explicit Status and Error_Message with NaN metrics (never 0.0)."""
        X, y_str, conds, sample_ids = _make_synthetic_loco_dataset(
            n_conditions=3, n_classes=3, samples_per_cond_class=5, seed=42
        )

        def _failing_runner(**kwargs: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            if kwargs["model_name"] == "SVM (RBF)":
                raise RuntimeError("Simulated solver failure")
            return evaluate_single_model_on_fold(**kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            results = run_dataset_loco_evaluations(
                dataset_name="CWRU_12k",
                X_all=X,
                y_all_str=y_str,
                conds_all=conds,
                sample_ids_all=sample_ids,
                models=["Random Forest", "SVM (RBF)"],
                balance_strategies=["none"],
                seeds=[42],
                max_folds=1,
                results_csv_path=csv_path,
                model_runner_override=_failing_runner,
            )
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 2)
            for col in REQUIRED_RESULTS_COLUMNS:
                self.assertIn(col, df.columns)

            svm_row = df[df["Model"] == "SVM (RBF)"].iloc[0]
            self.assertEqual(svm_row["Status"], "ERROR_MODEL_EXECUTION")
            self.assertIn("Simulated solver failure", str(svm_row["Error_Message"]))
            self.assertTrue(math.isnan(float(svm_row["Macro_F1"])))
            self.assertTrue(math.isnan(float(svm_row["Accuracy"])))
            self.assertTrue(math.isnan(float(svm_row["ROC_AUC"])))
            self.assertFalse(bool(svm_row["ROC_AUC_Valid"]))

            summaries = generate_summaries(df, Path(tmpdir))
            self.assertTrue(summaries["summary"].exists())
            self.assertTrue(summaries["summary_by_dataset_model"].exists())
            self.assertTrue(summaries["summary_across_seeds"].exists())

    def test_11_rejects_legacy_data_processed_directory(self) -> None:
        """Verify that `data/processed` is explicitly rejected in favor of `data/processed_tl/detrended_rms`."""
        with self.assertRaises(ValueError):
            validate_data_root_is_detrended_rms("/Users/vrocha/Documents/VibNet-1D/data/processed")
        with self.assertRaises(ValueError):
            validate_data_root_is_detrended_rms("/Users/vrocha/Documents/VibNet-1D/data/processed_tl/raw")
        valid_path = validate_data_root_is_detrended_rms(
            "/Users/vrocha/Documents/VibNet-1D/data/processed_tl/detrended_rms"
        )
        self.assertEqual(valid_path.name, "detrended_rms")

    def test_12_smoke_test_cwru12k_and_uoemd_end_to_end(self) -> None:
        """
        End-to-end synthetic smoke test of `run_benchmark` covering:
          - 1 fold from `CWRU_12k` (out of 4 canonical conditions, 4 classes)
          - 1 fold from `UOEMD` (out of 16 canonical conditions, 8 classes)
          - 1 seed (`42`)
          - Both balance strategies (`["none", "smote"]`)
          - All 5 models (`Random Forest`, `SVM (RBF)`, `XGBoost`, `MLP Scratch`, `TabNet Scratch`)
        """
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            detrended_rms_dir = tmp_root / "processed_tl" / "detrended_rms"
            output_root = tmp_root / "results" / "baseline_loco_v2"

            # Create canonical directory trees for CWRU_12k (4 conds, 4 classes) and UOEMD (16 conds, 8 classes)
            cwru_conds = ["Load_0HP", "Load_1HP", "Load_2HP", "Load_3HP"]
            cwru_classes = ["Class_0", "Class_1", "Class_2", "Class_3"]
            for c_idx, cond in enumerate(cwru_conds):
                for cls_idx, cls in enumerate(cwru_classes):
                    c_dir = detrended_rms_dir / "CWRU_12k" / cond / cls
                    c_dir.mkdir(parents=True, exist_ok=True)
                    # Make Class_0 smaller so SMOTE actively oversamples it
                    n_win = 3 if cls_idx == 0 else 5
                    for w_idx in range(n_win):
                        sig = (
                            np.sin(np.linspace(0, (cls_idx + 1) * 10 * np.pi, 256))
                            + 0.1 * c_idx
                            + rng.normal(0, 0.2, size=256)
                        ).astype(np.float32)
                        np.save(c_dir / f"win_{w_idx:03d}.npy", sig)

            uoemd_conds = [f"Cond_{i:02d}" for i in range(16)]
            uoemd_classes = [f"Class_{i}" for i in range(8)]
            for c_idx, cond in enumerate(uoemd_conds):
                for cls_idx, cls in enumerate(uoemd_classes):
                    c_dir = detrended_rms_dir / "UOEMD" / cond / cls
                    c_dir.mkdir(parents=True, exist_ok=True)
                    n_win = 2 if cls_idx == 0 else 3
                    for w_idx in range(n_win):
                        sig = (
                            np.cos(np.linspace(0, (cls_idx + 1) * 6 * np.pi, 256))
                            + 0.05 * c_idx
                            + rng.normal(0, 0.2, size=256)
                        ).astype(np.float32)
                        np.save(c_dir / f"win_{w_idx:03d}.npy", sig)

            def _synthetic_141_extractor(X_windows: np.ndarray, fs_hz: int) -> np.ndarray:
                # Deterministic 141-feature projection from window statistics + spectral FFT bins
                n_w = len(X_windows)
                out = np.zeros((n_w, FEATURE_DIMENSION), dtype=np.float32)
                fft_mag = np.abs(np.fft.rfft(X_windows, axis=1))
                n_bins = min(fft_mag.shape[1], FEATURE_DIMENSION - 5)
                out[:, :n_bins] = fft_mag[:, :n_bins]
                out[:, -5] = np.mean(X_windows, axis=1)
                out[:, -4] = np.std(X_windows, axis=1)
                out[:, -3] = np.max(np.abs(X_windows), axis=1)
                out[:, -2] = np.sqrt(np.mean(X_windows ** 2, axis=1))
                out[:, -1] = float(fs_hz) / 10000.0
                return out

            args = parse_args(
                [
                    "--targets",
                    "CWRU_12k",
                    "UOEMD",
                    "--models",
                    *CANONICAL_MODELS,
                    "--balance-strategies",
                    "none",
                    "smote",
                    "--seeds",
                    "42",
                    "--max-folds",
                    "1",
                    "--epochs",
                    "3",
                    "--batch-size",
                    "16",
                    "--tabnet-patience",
                    "2",
                    "--data-root",
                    str(detrended_rms_dir),
                    "--output-root",
                    str(output_root),
                    "--save-predictions",
                ]
            )
            run_dir = run_benchmark(
                args,
                feature_extractor=_synthetic_141_extractor,
                enforce_expected_folds=True,
            )

            for fname in (
                "config.json",
                "results.csv",
                "summary.csv",
                "summary_by_dataset_model.csv",
                "summary_across_seeds.csv",
                "run_metadata.json",
                "run.log",
                "predictions.csv",
            ):
                self.assertTrue((run_dir / fname).exists(), f"Missing output artifact: {fname}")

            df_res = pd.read_csv(run_dir / "results.csv")
            # 2 datasets * 1 fold * 1 seed * 2 balance strategies * 5 models = 20 rows
            self.assertEqual(len(df_res), 20)
            for col in REQUIRED_RESULTS_COLUMNS:
                self.assertIn(col, df_res.columns)
            self.assertTrue(all(df_res["Status"] == "OK"))
            self.assertTrue(all(df_res["ROC_AUC_Valid"] == True))  # noqa: E712
            self.assertFalse(df_res["Macro_F1"].isna().any())
            self.assertFalse(df_res["ROC_AUC"].isna().any())


if __name__ == "__main__":
    unittest.main()
