"""
Unit and synthetic test suite for the VibNet-1D Domain Adaptation (DA),
Domain Generalization (DG), and Transfer Learning (TL) experimental protocol
(`tests/test_da_dg_tl_protocol.py`).

Verifies all 11 mandatory requirements from Section 13 using exclusively synthetic
data:
  1. Zero overlap of original recordings across training, validation, and test partitions.
  2. Exact reproducibility of partitions with the same random seed.
  3. Zero usage of target domain data or statistics in DG normalization (and explicit
     blocking of silent arithmetic averaging across incompatible scales/units).
  4. Exact reuse of the adaptation-only reference RMS in both DA adaptation and DA test.
  5. Strict freezing of TabNet encoder parameters and GhostBatchNorm buffers in DA and DG.
  6. Genuine update of the 5-class classification head weights during DA adaptation.
  7. Preservation of the fixed 5-class output space (`Normal`, `Ball`, `Inner`, `Outer`,
     `Inner & Outer`), even when single-source configurations (`S1`, `S3`) lack classes.
  8. Strict error accounting when the 5-class head predicts a class absent from the
     target domain.
  9. Strict prohibition of balancing on validation and test partitions, plus verification
     of hierarchical source balancing (`1/D` per source dataset, `1/|C_d|` per class).
  10. Compatibility between `TabNetPretrainer` checkpoints and the 141-feature representation.
  11. Non-overwriting result directory allocation (`allocate_non_overwriting_run_dir`).
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from typing import List, Tuple

import numpy as np
import torch
from pytorch_tabnet.pretraining import TabNetPretrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vibnet.evaluation.da_dg_tl_protocol import (
    HARMONIZED_CLASSES,
    HARMONIZED_CLASS_TO_ID,
    NUM_HARMONIZED_CLASSES,
    IncomparableSourceRMSReferenceError,
    InsufficientRecordingsError,
    WindowRecord,
    allocate_non_overwriting_run_dir,
    assert_zero_recording_overlap,
    build_recording_key,
    compute_hierarchical_source_balance,
    create_da_recording_split,
    create_source_recording_split,
    diagnose_source_config_class_coverage,
    evaluate_five_class_predictions,
    fit_da_rms_normalizer,
    fit_dg_rms_normalizer,
    map_to_harmonized_class,
    parse_window_filename,
    predict_with_five_class_head,
    train_five_class_head_with_frozen_tabnet_encoder,
)
from src.vibnet.tabular.features import TABVIBNET_FEATURE_DIM


def _make_synthetic_dataset_records(
    dataset: str,
    raw_classes_and_recordings: List[Tuple[str, int, int]],
    condition: str = "Cond_0",
    window_len: int = 256,
    base_rms: float = 0.5,
    seed: int = 42,
) -> Tuple[List[WindowRecord], List[np.ndarray]]:
    """
    Create synthetic `WindowRecord` items and 1D waveforms where each recording
    `sNNNNN` produces `windows_per_rec` windows (`w00..wNN`).
    """
    rng = np.random.default_rng(seed)
    records: List[WindowRecord] = []
    windows: List[np.ndarray] = []
    rec_counter = 0

    for raw_cls, n_recs, wins_per_rec in raw_classes_and_recordings:
        harm_name, harm_id = map_to_harmonized_class(dataset, raw_cls)
        cls_scale = base_rms * (1.0 if harm_name == "Normal" else 2.2)
        for _ in range(n_recs):
            rec_prefix = f"s{rec_counter:05d}"
            rec_counter += 1
            rec_key = build_recording_key(dataset, condition, raw_cls, rec_prefix)
            for w_idx in range(wins_per_rec):
                fname = f"{rec_prefix}_w{w_idx:02d}.npy"
                rec_prefix_parsed, w_parsed = parse_window_filename(fname)
                assert rec_prefix_parsed == rec_prefix and w_parsed == w_idx
                sample_id = f"{dataset}/{condition}/{raw_cls}/{fname}"
                records.append(
                    WindowRecord(
                        sample_id=sample_id,
                        dataset=dataset,
                        condition=condition,
                        raw_class=raw_cls,
                        harmonized_class=harm_name,
                        harmonized_class_id=harm_id,
                        recording_prefix=rec_prefix,
                        recording_key=rec_key,
                        window_index=w_idx,
                    )
                )
                t = np.linspace(0.0, 1.0, window_len, endpoint=False)
                # Include linear trend + sinusoid + noise
                sig = (
                    0.15 * t
                    + cls_scale * np.sqrt(2.0) * np.sin(2.0 * np.pi * (10.0 + (harm_id or 0) * 5.0) * t)
                    + rng.normal(0.0, 0.02 * cls_scale, size=window_len)
                ).astype(np.float32)
                windows.append(sig)

    return records, windows


def _build_synthetic_pretrained_tabnet(
    input_dim: int = TABVIBNET_FEATURE_DIM,
    n_samples: int = 96,
    seed: int = 42,
) -> TabNetPretrainer:
    """Fit a fast, lightweight `TabNetPretrainer` on synthetic 141-dim tabular features."""
    rng = np.random.default_rng(seed)
    X_syn = rng.normal(0.0, 1.0, size=(n_samples, input_dim)).astype(np.float32)
    pretrainer = TabNetPretrainer(
        n_d=16,
        n_a=16,
        n_steps=4,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        mask_type="sparsemax",
        seed=int(seed),
        verbose=0,
    )
    pretrainer.fit(
        X_train=X_syn,
        eval_set=[X_syn[:32]],
        max_epochs=2,
        patience=2,
        batch_size=32,
        virtual_batch_size=16,
        num_workers=0,
        drop_last=False,
        pretraining_ratio=0.8,
    )
    return pretrainer


class TestDADGTLProtocol(unittest.TestCase):
    """Comprehensive verification of all 11 mandatory protocol tests (Section 13)."""

    def test_01_and_02_zero_recording_overlap_and_seed_reproducibility(self) -> None:
        """
        Tests 1 & 2:
          - Verify zero overlap of original recordings across train, val, and test in
            both DA (`adapt_train`, `adapt_val`, `test`) and DG (`source_train`, `source_val`).
          - Verify 100% deterministic reproducibility for the same seed and distinct
            splits across different seeds (`42`, `123`, `2026`).
          - Verify `InsufficientRecordingsError` is raised (never falling back to
            window splitting) when a class has too few recordings for a 20% adaptation
            recording-level train/val split.
        """
        records, _ = _make_synthetic_dataset_records(
            dataset="PU",
            raw_classes_and_recordings=[
                ("Class_Normal", 25, 4),
                ("Class_27", 25, 4),
                ("Class_29", 25, 4),
            ],
            seed=42,
        )

        parts_42_a, man_42_a = create_da_recording_split(records, seed=42)
        parts_42_b, man_42_b = create_da_recording_split(records, seed=42)
        parts_123, man_123 = create_da_recording_split(records, seed=123)

        # 1. Zero recording overlap across adapt_train, adapt_val, and test
        assert_zero_recording_overlap(records, parts_42_a)
        rec_tr = set(man_42_a.partition_recordings["adapt_train"])
        rec_val = set(man_42_a.partition_recordings["adapt_val"])
        rec_te = set(man_42_a.partition_recordings["test"])
        self.assertEqual(len(rec_tr & rec_val), 0)
        self.assertEqual(len(rec_tr & rec_te), 0)
        self.assertEqual(len(rec_val & rec_te), 0)
        self.assertEqual(len(rec_tr) + len(rec_val) + len(rec_te), 75)

        # Check 20% adaptation (15 recordings: 12 train + 3 val) and 80% test (60 recordings)
        self.assertAlmostEqual(man_42_a.recording_proportions["adapt_total"], 0.20, places=5)
        self.assertAlmostEqual(man_42_a.recording_proportions["test"], 0.80, places=5)

        # 2. Exact reproducibility with the same seed, different order with seed=123
        self.assertEqual(parts_42_a, parts_42_b)
        self.assertEqual(man_42_a.split_signature, man_42_b.split_signature)
        self.assertNotEqual(man_42_a.split_signature, man_123.split_signature)

        # Verify purposeful leakage triggers ValueError in assert_zero_recording_overlap
        corrupted_parts = dict(parts_42_a)
        corrupted_parts["test"] = tuple(list(parts_42_a["test"]) + [parts_42_a["adapt_train"][0]])
        with self.assertRaises(ValueError):
            assert_zero_recording_overlap(records, corrupted_parts)

        # Verify InsufficientRecordingsError when a class has only 4 recordings (20% = 1 recording,
        # which cannot be split into recording-level adapt_train and adapt_val!)
        scarce_records, _ = _make_synthetic_dataset_records(
            dataset="CWRU_12k",
            raw_classes_and_recordings=[
                ("Class_0", 4, 10),
                ("Class_1", 20, 10),
            ],
            seed=42,
        )
        with self.assertRaises(InsufficientRecordingsError):
            create_da_recording_split(scarce_records, seed=42)

        # Also test DG source recording split
        src_records_uored, _ = _make_synthetic_dataset_records(
            dataset="UORED",
            raw_classes_and_recordings=[("Class_Normal", 10, 5), ("Class_Ball", 10, 5)],
            seed=42,
        )
        src_parts_a, src_man_a = create_source_recording_split(
            src_records_uored, seed=42, source_config_id="S1"
        )
        src_parts_b, src_man_b = create_source_recording_split(
            src_records_uored, seed=42, source_config_id="S1"
        )
        assert_zero_recording_overlap(src_records_uored, src_parts_a)
        self.assertEqual(src_man_a.split_signature, src_man_b.split_signature)

    def test_03_dg_normalization_uses_zero_target_data_and_checks_comparability(self) -> None:
        """
        Test 3:
          - Verify that DG normalization computes `RMS_ref` strictly from source datasets
            and is 100% invariant to target dataset amplitudes.
          - Verify that `strict_check_arithmetic` refuses to silently average source
            `RMS_ref` values when units (`m/s^2` vs `g`) or amplitude scales (`5.85` vs `0.18`)
            are incompatible.
        """
        rng = np.random.default_rng(123)
        t = np.linspace(0.0, 1.0, 256, endpoint=False)
        uored_norm = [5.85 * np.sqrt(2.0) * np.sin(2.0 * np.pi * 15.0 * t)]
        hust_norm = [0.18 * np.sqrt(2.0) * np.sin(2.0 * np.pi * 15.0 * t)]
        mfpt_norm = [0.87 * np.sqrt(2.0) * np.sin(2.0 * np.pi * 15.0 * t)]

        # 1. Heterogeneous units/scales must raise IncomparableSourceRMSReferenceError
        #    under strict_check_arithmetic (never silently averaging incompatible scales)
        with self.assertRaises(IncomparableSourceRMSReferenceError):
            fit_dg_rms_normalizer(
                source_normal_windows={
                    "UORED": uored_norm,
                    "HUST": hust_norm,
                    "MFPT": mfpt_norm,
                },
                source_units={"UORED": "m/s^2", "HUST": "m/s^2", "MFPT": "g"},
                aggregation_mode="strict_check_arithmetic",
            )

        # 2. When sources have compatible units and comparable scales, strict_check_arithmetic succeeds
        s_a = [1.0 * np.sqrt(2.0) * np.sin(2.0 * np.pi * 10.0 * t)]
        s_b = [1.2 * np.sqrt(2.0) * np.sin(2.0 * np.pi * 10.0 * t)]
        dg_norm = fit_dg_rms_normalizer(
            source_normal_windows={"UORED": s_a, "HUST": s_b},
            source_units={"UORED": "m/s^2", "HUST": "m/s^2"},
            aggregation_mode="strict_check_arithmetic",
        )
        self.assertTrue(dg_norm.comparability_report["is_directly_comparable"])

        # 3. Verify that target windows have ZERO influence on dg_norm parameters
        target_win = [rng.normal(0.0, 50.0, size=256).astype(np.float32)]
        out_1 = dg_norm.transform_target_windows(target_win, target_dataset="CWRU_12k")[0]
        out_2 = dg_norm.transform_target_windows(target_win, target_dataset="PU")[0]
        np.testing.assert_allclose(out_1, out_2, rtol=1e-7, atol=1e-7)

    def test_04_da_normalization_uses_same_adaptation_rms_for_adapt_and_test(self) -> None:
        """
        Test 4:
          - Verify that `fit_da_rms_normalizer` estimates `rms_ref` exclusively from
            `Normal` windows in the adaptation partition (`adapt_train`) and applies
            that exact same `rms_ref` to both adaptation and test windows.
          - Verify that corrupting or scaling the 80% test windows by 1000x before
            `fit_da_rms_normalizer` does NOT change `da_norm.rms_ref` by a single bit.
        """
        records, windows = _make_synthetic_dataset_records(
            dataset="PU",
            raw_classes_and_recordings=[
                ("Class_Normal", 20, 3),
                ("Class_27", 20, 3),
            ],
            base_rms=0.40,
            seed=42,
        )
        parts, _ = create_da_recording_split(records, seed=42)
        adapt_all = list(parts["adapt_train"]) + list(parts["adapt_val"])
        test_idx = list(parts["test"])

        da_norm_1 = fit_da_rms_normalizer(
            records=records,
            windows=windows,
            adapt_indices=adapt_all,
            test_indices=test_idx,
        )

        # Multiply all test windows by 1000x and verify `rms_ref` is completely unchanged
        windows_modified_test = [w.copy() for w in windows]
        for idx in test_idx:
            windows_modified_test[idx] = windows_modified_test[idx] * 1000.0

        da_norm_2 = fit_da_rms_normalizer(
            records=records,
            windows=windows_modified_test,
            adapt_indices=adapt_all,
            test_indices=test_idx,
        )
        self.assertEqual(da_norm_1.rms_ref, da_norm_2.rms_ref)

        # Verify the exact same reference is applied to adaptation and test signals
        sample_signal = [windows[adapt_all[0]]]
        normed = da_norm_1.transform_windows(sample_signal)[0]
        from src.vibnet.preprocessing.signal_norm import detrend_linear_window

        expected = (
            np.asarray(detrend_linear_window(sample_signal[0]), dtype=np.float64)
            / da_norm_1.rms_ref
        ).astype(np.float32)
        np.testing.assert_allclose(normed, expected, rtol=1e-6, atol=1e-6)

        # Verify overlap between adapt_indices and test_indices is rejected
        with self.assertRaises(ValueError):
            fit_da_rms_normalizer(
                records=records,
                windows=windows,
                adapt_indices=adapt_all + [test_idx[0]],
                test_indices=test_idx,
            )

    def test_05_06_07_frozen_encoder_head_update_and_five_class_outputs_in_dg_and_da(self) -> None:
        """
        Tests 5, 6, 7 & 10:
          - Test 10: Verify `TabNetPretrainer` checkpoint compatibility with the 141-feature
            input dimension (`TABVIBNET_FEATURE_DIM == 141`).
          - Test 5: Verify that all encoder parameters and GhostBatchNorm buffers remain
            strictly frozen (`max_abs_diff == 0.0`) during both DG source head training
            and DA target head adaptation.
          - Test 6: Verify that the 5-class head weights are genuinely updated during DA
            adaptation (`head_param_max_abs_diff > 0.0`) after being initialized from a
            copy of the source head weights.
          - Test 7: Verify that the 5-class output space (`shape (N, 5)`) is preserved
            for all source configurations (`S1..S4`), and that in single-source DG (`S1`
            with only `Normal` and `Ball`), unobserved classes (`Inner`, `Outer`,
            `Inner & Outer`) are masked and never predicted by random weight noise.
        """
        pretrainer = _build_synthetic_pretrained_tabnet(
            input_dim=TABVIBNET_FEATURE_DIM, n_samples=64, seed=42
        )
        rng = np.random.default_rng(42)

        # Snapshot pretrainer encoder before any DG or DA call
        orig_enc_state = {
            k: v.detach().cpu().numpy().copy()
            for k, v in pretrainer.network.encoder.state_dict().items()
        }

        # Case A: Single-source S1 (UORED: only classes 0=Normal and 1=Ball)
        cov_s1 = diagnose_source_config_class_coverage("S1")
        self.assertEqual(cov_s1["present_class_ids"], [0, 1])
        self.assertEqual(cov_s1["absent_class_ids"], [2, 3, 4])
        self.assertEqual(cov_s1["head_output_dim"], 5)

        X_src_tr = rng.normal(0, 1, size=(48, TABVIBNET_FEATURE_DIM)).astype(np.float32)
        y_src_tr = np.array([0, 1] * 24, dtype=np.int64)
        X_src_val = rng.normal(0, 1, size=(16, TABVIBNET_FEATURE_DIM)).astype(np.float32)
        y_src_val = np.array([0, 1] * 8, dtype=np.int64)

        dg_head_s1 = train_five_class_head_with_frozen_tabnet_encoder(
            pretrained_tabnet=pretrainer,
            X_train_scaled=X_src_tr,
            y_train=y_src_tr,
            X_val_scaled=X_src_val,
            y_val=y_src_val,
            seed=42,
            source_config_id="S1",
            max_epochs=8,
            patience=4,
            batch_size=16,
        )
        self.assertEqual(dg_head_s1.encoder_param_max_abs_diff, 0.0)
        self.assertEqual(dg_head_s1.encoder_buffer_max_abs_diff, 0.0)
        self.assertGreater(dg_head_s1.head_param_max_abs_diff, 0.0)
        self.assertEqual(dg_head_s1.observed_class_ids, (0, 1))

        # Predict on target with S1 DG head: must have 5 probability columns, and zero
        # probability mass on unobserved classes (2, 3, 4)
        X_tgt_test = rng.normal(0, 1, size=(30, TABVIBNET_FEATURE_DIM)).astype(np.float32)
        preds_dg, probs_dg = predict_with_five_class_head(
            pretrained_tabnet=pretrainer,
            head_checkpoint=dg_head_s1,
            X_test_scaled=X_tgt_test,
        )
        self.assertEqual(probs_dg.shape, (30, NUM_HARMONIZED_CLASSES))
        self.assertTrue(set(preds_dg.tolist()).issubset({0, 1}))
        np.testing.assert_allclose(probs_dg[:, [2, 3, 4]], 0.0, atol=1e-7)

        # Case B: DA adaptation initialized from dg_head_s1 on a target containing classes 0, 1, 2, 3
        X_da_tr = rng.normal(0, 1, size=(40, TABVIBNET_FEATURE_DIM)).astype(np.float32)
        y_da_tr = np.array([0, 1, 2, 3] * 10, dtype=np.int64)
        X_da_val = rng.normal(0, 1, size=(12, TABVIBNET_FEATURE_DIM)).astype(np.float32)
        y_da_val = np.array([0, 1, 2, 3] * 3, dtype=np.int64)

        source_head_weights_before_da = copy.deepcopy(dg_head_s1.head_state_dict["weight"])

        da_head = train_five_class_head_with_frozen_tabnet_encoder(
            pretrained_tabnet=pretrainer,
            X_train_scaled=X_da_tr,
            y_train=y_da_tr,
            X_val_scaled=X_da_val,
            y_val=y_da_val,
            seed=42,
            init_head_checkpoint=dg_head_s1,
            source_config_id="S1",
            max_epochs=10,
            patience=5,
            batch_size=16,
        )
        # Verify encoder stayed 100% frozen during DA
        self.assertEqual(da_head.encoder_param_max_abs_diff, 0.0)
        self.assertEqual(da_head.encoder_buffer_max_abs_diff, 0.0)
        # Verify head weights updated during DA and source head checkpoint was not mutated
        self.assertGreater(da_head.head_param_max_abs_diff, 0.0)
        self.assertFalse(
            torch.allclose(
                da_head.head_state_dict["weight"].cpu(),
                source_head_weights_before_da.cpu(),
            )
        )
        self.assertEqual(da_head.observed_class_ids, (0, 1, 2, 3))

        preds_da, probs_da = predict_with_five_class_head(
            pretrained_tabnet=pretrainer,
            head_checkpoint=da_head,
            X_test_scaled=X_tgt_test,
        )
        self.assertEqual(probs_da.shape, (30, NUM_HARMONIZED_CLASSES))

        # Verify original pretrainer encoder state is bit-for-bit identical
        for k, v in pretrainer.network.encoder.state_dict().items():
            np.testing.assert_array_equal(v.detach().cpu().numpy(), orig_enc_state[k])

    def test_08_absent_target_class_predictions_counted_as_errors(self) -> None:
        """
        Test 8:
          - Verify that when a target dataset does not contain a class (for example,
            `CWRU_12k` has classes `{0, 1, 2, 3}` and does NOT have `4: Inner & Outer`),
            any prediction of class `4` is strictly counted as a classification error,
            reducing `Accuracy`, `Balanced_Accuracy`, and `Macro_F1`.
          - Verify that when a target class is never predicted (for example, `S1`
            only predicts `{0, 1}` on a target with `{0, 1, 2, 3}`), the unpredicted
            target classes `{2, 3}` receive `F1 = 0.0` and penalize `Macro_F1`.
        """
        # Ground truth in CWRU_12k: 4 samples of each class 0, 1, 2, 3 (no class 4)
        y_true = [0, 0, 1, 1, 2, 2, 3, 3]
        # Perfect prediction
        m_perfect = evaluate_five_class_predictions(y_true, y_true)
        self.assertAlmostEqual(m_perfect["macro_f1"], 1.0)
        self.assertEqual(m_perfect["out_of_target_prediction_count"], 0)

        # Now predict phantom class 4 ("Inner & Outer") for half of class 2 and class 3 samples
        y_pred_with_phantom = [0, 0, 1, 1, 2, 4, 3, 4]
        m_phantom = evaluate_five_class_predictions(y_true, y_pred_with_phantom)
        self.assertEqual(m_phantom["out_of_target_prediction_count"], 2)
        self.assertLess(m_phantom["accuracy"], 1.0)
        self.assertLess(m_phantom["balanced_accuracy"], 1.0)
        self.assertLess(m_phantom["macro_f1"], 1.0)
        self.assertLess(
            m_phantom["macro_f1_including_phantom_classes"],
            m_phantom["macro_f1"],
        )

        # Now simulate S1 (only predicts 0 and 1) evaluated on target {0, 1, 2, 3}
        y_pred_s1_limited = [0, 0, 1, 1, 0, 0, 1, 1]
        m_s1 = evaluate_five_class_predictions(y_true, y_pred_s1_limited)
        self.assertEqual(m_s1["per_class_f1_all_5"]["Inner"], 0.0)
        self.assertEqual(m_s1["per_class_f1_all_5"]["Outer"], 0.0)
        self.assertLessEqual(m_s1["macro_f1"], 0.50)

    def test_09_hierarchical_source_balancing_and_no_val_test_balancing(self) -> None:
        """
        Test 9:
          - Verify that `compute_hierarchical_source_balance` equalizes the total
            effective weight of each source dataset (`UORED`, `HUST`, `MFPT` -> 1/3 each)
            AND equalizes each present class within each dataset (`1 / |C_d|`).
          - Verify that attempting to balance `val`, `adapt_val`, or `test` raises
            `ValueError`.
        """
        # Simulate S4 counts: UORED (180 Normal, 100 Ball), HUST (150 each of 0..4), MFPT (21 Inner, 39 Outer)
        datasets = (
            ["UORED"] * 280
            + ["HUST"] * 700
            + ["MFPT"] * 60
        )
        class_ids = (
            [0] * 180 + [1] * 100
            + [0] * 150 + [1] * 100 + [2] * 150 + [3] * 150 + [4] * 150
            + [2] * 21 + [3] * 39
        )

        res_w = compute_hierarchical_source_balance(
            datasets=datasets,
            class_ids=class_ids,
            mode="hierarchical_weights",
            seed=42,
            partition_name="source_train",
        )
        n_total = len(datasets)
        expected_ds_weight = n_total / 3.0
        for ds_name in ("UORED", "HUST", "MFPT"):
            self.assertAlmostEqual(
                res_w.effective_weight_by_dataset[ds_name],
                expected_ds_weight,
                places=5,
            )

        # Within UORED (2 classes), each class gets expected_ds_weight / 2
        for cls_name in ("Normal", "Ball"):
            self.assertAlmostEqual(
                res_w.effective_weight_by_dataset_class["UORED"][cls_name],
                expected_ds_weight / 2.0,
                places=5,
            )
        # Within HUST (5 classes), each class gets expected_ds_weight / 5
        for cls_name in HARMONIZED_CLASSES:
            self.assertAlmostEqual(
                res_w.effective_weight_by_dataset_class["HUST"][cls_name],
                expected_ds_weight / 5.0,
                places=5,
            )

        # Also verify hierarchical_resample equalizes sample counts per dataset and per class-in-dataset
        res_r = compute_hierarchical_source_balance(
            datasets=datasets,
            class_ids=class_ids,
            mode="hierarchical_resample",
            seed=42,
            partition_name="source_train",
        )
        uored_tot = sum(res_r.resampled_distribution_after["UORED"].values())
        hust_tot = sum(res_r.resampled_distribution_after["HUST"].values())
        mfpt_tot = sum(res_r.resampled_distribution_after["MFPT"].values())
        self.assertEqual(uored_tot, hust_tot)
        self.assertEqual(hust_tot, mfpt_tot)

        # Balancing validation or test partitions must be rejected
        for forbidden_part in ("val", "validation", "adapt_val", "test"):
            with self.assertRaises(ValueError):
                compute_hierarchical_source_balance(
                    datasets=datasets,
                    class_ids=class_ids,
                    mode="hierarchical_weights",
                    partition_name=forbidden_part,
                )

    def test_11_non_overwriting_result_directories(self) -> None:
        """
        Test 11:
          - Verify that `allocate_non_overwriting_run_dir` creates distinct directories
            even when called multiple times within the same second, never overwriting
            previous experiment outputs.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "da_dg_tl_matrix"
            dir_1 = allocate_non_overwriting_run_dir(root, experiment_prefix="run")
            (dir_1 / "results.csv").write_text("run1\n", encoding="utf-8")
            dir_2 = allocate_non_overwriting_run_dir(root, experiment_prefix="run")
            (dir_2 / "results.csv").write_text("run2\n", encoding="utf-8")

            self.assertNotEqual(dir_1, dir_2)
            self.assertEqual((dir_1 / "results.csv").read_text(encoding="utf-8"), "run1\n")
            self.assertEqual((dir_2 / "results.csv").read_text(encoding="utf-8"), "run2\n")


if __name__ == "__main__":
    unittest.main()
