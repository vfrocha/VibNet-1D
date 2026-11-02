"""
Unit tests for the VibNet-1D modular architecture (`src.vibnet.*`, `configs/`,
and `experiments/tabvibnet/`).

Runs with the standard Python `unittest` runner (`py -m unittest discover -s tests -v`)
without requiring external ML packages (`numpy`, `torch`, `PyYAML`).
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.tabvibnet.run_contrastive_tl import (
    build_execution_plan as build_contrastive_plan,
)
from experiments.tabvibnet.run_cross_dataset_tl import (
    build_execution_plan as build_cross_plan,
)
from experiments.tabvibnet.run_variants_and_balance import (
    build_execution_plan as build_variants_plan,
)
from src.vibnet.data import (
    CANONICAL_DATASETS,
    audit_class_condition_coverage,
    build_reference_map,
    get_dataset_spec,
    get_sampling_rate_hz,
    list_sources,
    list_targets,
    load_yaml_config,
)
from src.vibnet.evaluation import (
    assert_no_condition_leakage,
    balance_target_training,
    build_source_combinations,
    collect_run_provenance,
    compute_fold_metrics,
    compute_selection_score,
    ensure_batchnorm_safe_batch_size,
    iter_loco_folds,
    trim_single_remainder_sample,
    write_run_metadata,
)
from src.vibnet.preprocessing import (
    compute_reference_rms,
    compute_window_rms,
    detrend_linear_window,
    fit_transform_target_fold,
    normalize_windows_by_reference,
    scale_source_domains,
)
from src.vibnet.signal import (
    TimeDomainDatasetSpec,
    TimeDomainLOCOAdapter,
    TimeDomainWindowBatch,
    verify_tabular_signal_loco_alignment,
)
from src.vibnet.tabular import (
    TABVIBNET_FEATURE_DIM,
    get_tabvibnet_feature_spec,
    normalize_model_return_tuple,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestVibNetModularArchitecture(unittest.TestCase):
    """Verify contracts across `src.vibnet.data`, `preprocessing`, `evaluation`, `tabular`, and `signal`."""

    def test_canonical_dataset_registry_and_sampling_rates(self) -> None:
        self.assertEqual(len(CANONICAL_DATASETS), 10)
        # Verify fixed sampling rates for Mechanical_Gear and Electric_Motor (ADR-004)
        self.assertEqual(get_sampling_rate_hz("Mechanical_Gear"), 5000)
        self.assertEqual(get_sampling_rate_hz("Electric_Motor"), 50000)
        self.assertEqual(get_sampling_rate_hz("CWRU_12k"), 12000)
        self.assertEqual(get_sampling_rate_hz("CWRU_48k"), 48000)
        self.assertEqual(get_sampling_rate_hz("PU"), 64000)
        self.assertEqual(get_sampling_rate_hz("HUST_Gearbox"), 25600)
        self.assertEqual(get_sampling_rate_hz("UOEMD"), 42000)

        # Unknown dataset must raise KeyError (no silent 12,000 Hz fallback)
        with self.assertRaises(KeyError):
            get_dataset_spec("NonExistentDataset")

        ref_map = build_reference_map()
        self.assertEqual(ref_map["CWRU_48k"], ("CWRU_12k", "Class_0"))
        self.assertEqual(ref_map["PU"], ("PU", "Healthy"))

        targets = list_targets(cross_dataset_benchmark_only=True)
        sources = list_sources(cross_dataset_benchmark_only=True)
        self.assertEqual(
            targets, ["CWRU_12k", "CWRU_48k", "HUST_Gearbox", "PU", "UOEMD"]
        )
        self.assertEqual(
            sources, ["UORED", "HUST", "Mechanical_Gear", "Electric_Motor"]
        )

    def test_yaml_configs_load_cleanly(self) -> None:
        datasets_cfg = load_yaml_config(PROJECT_ROOT / "configs" / "datasets.yaml")
        self.assertIn("datasets", datasets_cfg)
        self.assertEqual(
            datasets_cfg["datasets"]["Mechanical_Gear"]["fs_hz"], 5000
        )
        self.assertEqual(
            datasets_cfg["datasets"]["Electric_Motor"]["fs_hz"], 50000
        )

        cross_plan = build_cross_plan(
            PROJECT_ROOT / "configs" / "tabvibnet" / "tl_cross_dataset_mlp_tabnet.yaml"
        )
        self.assertEqual(cross_plan["num_source_combinations"], 15)
        self.assertEqual(cross_plan["total_loco_folds_per_combo"], 66)
        self.assertEqual(cross_plan["expected_total_rows"], 5940)

        contrastive_plan = build_contrastive_plan(
            PROJECT_ROOT / "configs" / "tabvibnet" / "tl_contrastive_mlp.yaml"
        )
        self.assertEqual(contrastive_plan["target_dataset"], "CWRU_48k")
        self.assertEqual(contrastive_plan["num_loco_folds"], 12)
        self.assertEqual(contrastive_plan["seeds"], [42, 123, 2026])

        variants_plan = build_variants_plan(
            PROJECT_ROOT / "configs" / "tabvibnet" / "tl_variants_and_balance.yaml",
            stage="variants",
        )
        self.assertEqual(variants_plan["target_dataset"], "CWRU_48k")
        self.assertEqual(variants_plan["num_loco_folds"], 12)

        td_cfg = load_yaml_config(
            PROJECT_ROOT / "configs" / "time_domain" / "template_1d_loco.yaml"
        )
        self.assertEqual(td_cfg["experiment_family"], "time_domain")
        self.assertEqual(td_cfg["data"]["window_duration_sec"], 1.0)

    def test_signal_rms_reference_and_leak_free_scaling(self) -> None:
        normal_windows = [[3.0, 4.0, 3.0, 4.0], [3.0, 4.0, 3.0, 4.0]]
        # RMS of [3, 4, 3, 4] is sqrt((9+16+9+16)/4) = sqrt(12.5)
        rms_ref = compute_reference_rms(normal_windows)
        self.assertAlmostEqual(rms_ref, (12.5) ** 0.5, places=6)

        norm_wins = normalize_windows_by_reference(
            normal_windows, rms_ref=rms_ref, detrend=False
        )
        self.assertAlmostEqual(compute_window_rms(norm_wins[0]), 1.0, places=6)

        # Linear ramp window should detrend to all zeros
        ramp = [2.0 * i + 5.0 for i in range(64)]
        detrended = detrend_linear_window(ramp)
        for val in detrended:
            self.assertAlmostEqual(val, 0.0, places=9)

        # Leak-free scaling: test statistics must NOT affect scaler mean/scale
        X_train = [[1.0, 10.0], [3.0, 20.0]]
        X_test = [[100.0, 1000.0]]
        X_tr_s, X_te_s, stats = fit_transform_target_fold(X_train, X_test)
        self.assertEqual(stats.mean, (2.0, 15.0))
        self.assertEqual(stats.scale, (1.0, 5.0))
        self.assertEqual(X_tr_s, [[-1.0, -1.0], [1.0, 1.0]])
        self.assertEqual(X_te_s, [[98.0, 197.0]])

        scaled_sources, _ = scale_source_domains({"S1": X_train})
        self.assertEqual(scaled_sources["S1"], [[-1.0, -1.0], [1.0, 1.0]])

    def test_loco_protocol_and_source_combinations(self) -> None:
        combos = build_source_combinations(
            ["UORED", "HUST", "Mechanical_Gear", "Electric_Motor"],
            modes=("single", "pairs", "triplets", "all"),
        )
        self.assertEqual(len(combos), 15)

        conditions = ["C1", "C1", "C2", "C2", "C3", "C3"]
        splits = list(iter_loco_folds(conditions))
        self.assertEqual(len(splits), 3)
        for sp in splits:
            assert_no_condition_leakage(conditions, sp.train_indices, sp.test_indices)
            self.assertNotIn(sp.test_condition, sp.train_conditions)

        # Verify leak detection raises ValueError
        with self.assertRaises(ValueError):
            assert_no_condition_leakage(conditions, [0, 2], [1, 3])

    def test_balancing_and_batchnorm_remainder_one_protection(self) -> None:
        # Reproduce exact PU fold C4_1500rpm_0.7Nm_400N count: 8,577 samples % 64 == 1
        safe_n, needed = ensure_batchnorm_safe_batch_size(8577, batch_size=64)
        self.assertTrue(needed)
        self.assertEqual(safe_n, 8576)
        self.assertEqual(safe_n % 64, 0)

        # Synthetic 65-sample dataset (65 % 64 == 1)
        X_synth = [[float(i), float(i * 2)] for i in range(65)]
        y_synth = [0] * 45 + [1] * 20
        X_trim, y_trim, dropped = trim_single_remainder_sample(
            X_synth, y_synth, batch_size=64, seed=42
        )
        self.assertTrue(dropped)
        self.assertEqual(len(X_trim), 64)
        self.assertEqual(len(y_trim), 64)
        # Majority class (0) loses 1 sample; minority class (1) keeps all 20 samples
        self.assertEqual(sum(1 for y in y_trim if y == 0), 44)
        self.assertEqual(sum(1 for y in y_trim if y == 1), 20)

        # Verify partial_smote produces non-uniform class weights unlike full smote/hybrid
        X_imb = [[float(i), 1.0] for i in range(20)] + [
            [float(i), -1.0] for i in range(6)
        ]
        y_imb = [0] * 20 + [1] * 6
        res_hybrid = balance_target_training(
            X_imb,
            y_imb,
            strategy="hybrid",
            seed=42,
            batch_size_for_batchnorm_safety=None,
        )
        self.assertAlmostEqual(res_hybrid.class_weights[0], 1.0, places=6)
        self.assertAlmostEqual(res_hybrid.class_weights[1], 1.0, places=6)

        res_partial = balance_target_training(
            X_imb,
            y_imb,
            strategy="partial_smote",
            seed=42,
            batch_size_for_batchnorm_safety=None,
        )
        self.assertNotAlmostEqual(
            res_partial.class_weights[0], res_partial.class_weights[1], places=3
        )

    def test_metrics_and_shifted_tuple_normalization(self) -> None:
        # When all classes are present in y_true, ROC_AUC is valid
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_proba = [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]
        m_full = compute_fold_metrics(
            y_true, y_pred, y_proba=y_proba, expected_classes=[0, 1]
        )
        self.assertEqual(m_full.macro_f1, 1.0)
        self.assertTrue(m_full.roc_auc_valid)
        self.assertEqual(m_full.roc_auc_ovr, 1.0)

        # When class 2 is missing from y_true (e.g. UOEMD condition 5..8),
        # roc_auc_ovr must be None (never 0.0!) and roc_auc_valid=False
        m_missing = compute_fold_metrics(
            y_true,
            y_pred,
            y_proba=[[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.7, 0.2]],
            expected_classes=[0, 1, 2],
        )
        self.assertFalse(m_missing.roc_auc_valid)
        self.assertIsNone(m_missing.roc_auc_ovr)

        # Selection score = average of MLP and TABNET Macro-F1
        sel = compute_selection_score({"MLP": 0.9202, "TABNET": 0.8909})
        self.assertAlmostEqual(sel, 0.90555, places=5)

        # 4-tuple return from PureMLPBaseline must not corrupt recall/roc_auc
        norm4 = normalize_model_return_tuple((0.85, 0.84, 0.86, 0.83))
        self.assertEqual(norm4.recall_macro, 0.83)
        self.assertIsNone(norm4.roc_auc_ovr)
        self.assertFalse(norm4.roc_auc_valid)

    def test_tabular_and_time_domain_loco_alignment(self) -> None:
        feat_spec = get_tabvibnet_feature_spec(include_aux_energy=False)
        self.assertEqual(feat_spec.total_dim, TABVIBNET_FEATURE_DIM)
        self.assertEqual(feat_spec.total_dim, 141)

        td_spec = TimeDomainDatasetSpec.from_dataset_name("CWRU_48k")
        self.assertEqual(td_spec.dataset_spec.sampling_rate_hz, 48000)
        self.assertEqual(td_spec.window_length, 2048)

        conditions = ("Cond_A", "Cond_A", "Cond_B", "Cond_B", "Cond_C", "Cond_C")
        labels = (0, 1, 0, 1, 0, 1)
        batch = TimeDomainWindowBatch(
            dataset_name="CWRU_48k",
            variant="detrended_rms",
            num_samples=6,
            channels=1,
            window_length=2048,
            labels=labels,
            conditions=conditions,
        )
        alignment = verify_tabular_signal_loco_alignment(conditions, labels, batch)
        self.assertTrue(alignment["aligned"])
        self.assertEqual(alignment["num_folds"], 3)

        adapter = TimeDomainLOCOAdapter(batch)
        self.assertEqual(len(list(adapter.iter_splits())), 3)

    def test_provenance_and_coverage_audit(self) -> None:
        cov = audit_class_condition_coverage(
            {
                "C1": ["Normal", "FaultA"],
                "C2": ["Normal", "FaultA"],
            }
        )
        self.assertTrue(cov["loco_eligible"])
        self.assertTrue(cov["full_class_coverage_every_fold"])

        with tempfile.TemporaryDirectory() as tmpdir:
            prov = collect_run_provenance(
                project_root=PROJECT_ROOT,
                script_paths=[Path(__file__)],
                seeds=[42, 123, 2026],
                extra_config={"test": True},
            )
            out_file = write_run_metadata(tmpdir, prov)
            self.assertTrue(out_file.is_file())
            self.assertEqual(prov["seeds"], [42, 123, 2026])
            self.assertIn("git_commit", prov)
            self.assertIn("script_hashes_sha256", prov)


if __name__ == "__main__":
    unittest.main()
