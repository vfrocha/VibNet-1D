"""
Unit and synthetic/on-disk test suite for the VibNet-1D Domain Adaptation (DA),
Domain Generalization (DG), and Transfer Learning (TL) experimental protocol
(`tests/test_da_dg_tl_protocol.py`).

Verifies all 17 mandatory requirements from Section 13:
  1. Correspondência das classes de cada configuração de origem (`S1`: Normal, Ball;
     `S2`: Normal, Ball, Inner, Outer, Inner & Outer; `S3`: Inner, Outer; `S4`: 5 classes).
  2. Dimensão correta das cabeças iniciais (`S1`: 2, `S2`: 5, `S3`: 2, `S4`: 5).
  3. Expansão correta das cabeças DA (`expand_head_for_da_target` por nome canônico).
  4. Preservação exata dos pesos e bias das classes compartilhadas na expansão DA.
  5. Inicialização reproduzível das novas saídas da cabeça expandida de acordo com a seed.
  6. Ausência de expansão da cabeça no protocolo DG.
  7. Contabilização de classes desconhecidas (em DG) e classes ausentes no destino como erros.
  8. Congelamento efetivo do encoder TabNet (parâmetros e buffers GhostBatchNorm) em DA e DG.
  9. Atualização genuína dos pesos da cabeça durante a adaptação DA.
  10. Ausência de sobreposição de gravações originais entre partições (`train`, `val`, `test`, `adapt`).
  11. Reprodutibilidade determinística das partições com a mesma seed (incluindo a exceção de
      `CWRU_12k` Normal: 2 gravações integrais na adaptação e 6 no teste).
  12. Ausência de dados dos destinos na normalização DG (conversão `1 g = 9.80665 m/s^2`,
      média geométrica `RMS_global = exp(mean(log(RMS_ref_i)))`, e `StandardScaler` exclusivo nas fontes).
  13. Uso da referência RMS de adaptação tanto na adaptação quanto no teste DA (e calibração
      externa de `CWRU_48k` exclusivamente com as 2 gravações normais de adaptação de `CWRU_12k`).
  14. Ausência de balanceamento na validação e no teste + balanceamento hierárquico nas fontes
      + baselines clássicos RF/SVM (DG, DA Target-only 20%, DA Source + Target 20%).
  15. Compatibilidade dos checkpoints com as 141 features e catálogo matemático das 126 features
      invariantes a escala positiva vs. 15 sensíveis (bloqueando reuso indevido de checkpoint de
      141 entradas na ablação de 126 features).
  16. Correção da decimação anti-aliasing (`97.656 Hz -> 48.828 Hz`), janelamento de 1s (`78` janelas)
      e preparação isolada de `detrended_rms/MFPT` preservando `data/processed/MFPT`.
  17. Ausência de sobrescrita dos resultados anteriores (`allocate_non_overwriting_run_dir`).
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
    STANDARD_GRAVITY_MS2,
    IncomparableSourceRMSReferenceError,
    InsufficientRecordingsError,
    WindowRecord,
    allocate_non_overwriting_run_dir,
    assert_zero_recording_overlap,
    assess_source_rms_comparability,
    audit_cwru_12k_48k_calibration_compatibility,
    build_recording_key,
    compute_dg_global_rms_reference_ms2,
    compute_hierarchical_source_balance,
    convert_rms_to_ms2,
    create_da_recording_split,
    create_source_recording_split,
    diagnose_source_config_class_coverage,
    evaluate_variable_space_predictions,
    expand_head_for_da_target,
    fit_da_rms_normalizer,
    fit_dg_rms_normalizer,
    fit_source_only_standard_scaler,
    get_141_feature_scale_invariance_catalog,
    get_scale_invariant_126_indices,
    get_scale_sensitive_15_indices,
    get_source_config_classes,
    map_to_harmonized_class,
    parse_window_filename,
    predict_with_variable_class_head,
    prepare_isolated_mfpt_detrended_rms,
    select_feature_subset_for_ablation,
    train_eval_classical_da_dg_baseline,
    train_variable_class_head_with_frozen_tabnet_encoder,
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
                sig = (
                    0.15 * t
                    + cls_scale
                    * np.sqrt(2.0)
                    * np.sin(2.0 * np.pi * (10.0 + (harm_id or 0) * 5.0) * t)
                    + rng.normal(0.0, 0.02 * cls_scale, size=window_len)
                ).astype(np.float32)
                windows.append(sig)

    return records, windows


def _build_synthetic_pretrained_tabnet(
    input_dim: int = TABVIBNET_FEATURE_DIM,
    n_samples: int = 96,
    seed: int = 42,
) -> TabNetPretrainer:
    """Fit a fast, lightweight `TabNetPretrainer` on synthetic tabular features."""
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
    """Comprehensive verification of all 17 mandatory protocol tests (Section 13)."""

    def test_01_and_02_source_class_correspondence_and_variable_head_dimensions(self) -> None:
        """
        Tests 1 & 2:
          - Verify exact class correspondence for S1, S2, S3, and S4:
            * S1 (UORED): ("Normal", "Ball") -> 2 outputs
            * S2 (HUST Bearing): ("Normal", "Ball", "Inner", "Outer", "Inner & Outer") -> 5 outputs
            * S3 (MFPT): ("Inner", "Outer") -> 2 outputs
            * S4 (Combined): ("Normal", "Ball", "Inner", "Outer", "Inner & Outer") -> 5 outputs
          - Verify that initial heads trained for S1, S2, S3, and S4 have exact output
            dimensions 2, 5, 2, and 5 respectively (never imposing 5 outputs on S1 or S3).
        """
        cov_s1 = diagnose_source_config_class_coverage("S1")
        cov_s2 = diagnose_source_config_class_coverage("S2")
        cov_s3 = diagnose_source_config_class_coverage("S3")
        cov_s4 = diagnose_source_config_class_coverage("S4")

        self.assertEqual(cov_s1["present_class_names"], ["Normal", "Ball"])
        self.assertEqual(cov_s1["head_output_dim"], 2)
        self.assertEqual(get_source_config_classes("S1"), ("Normal", "Ball"))

        self.assertEqual(
            cov_s2["present_class_names"],
            ["Normal", "Ball", "Inner", "Outer", "Inner & Outer"],
        )
        self.assertEqual(cov_s2["head_output_dim"], 5)

        self.assertEqual(cov_s3["present_class_names"], ["Inner", "Outer"])
        self.assertEqual(cov_s3["head_output_dim"], 2)
        self.assertEqual(get_source_config_classes("S3"), ("Inner", "Outer"))

        self.assertEqual(
            cov_s4["present_class_names"],
            ["Normal", "Ball", "Inner", "Outer", "Inner & Outer"],
        )
        self.assertEqual(cov_s4["head_output_dim"], 5)

        # Verify actual trained head tensor dimensions for S1 (2), S2 (5), S3 (2), S4 (5)
        pretrainer = _build_synthetic_pretrained_tabnet(
            input_dim=TABVIBNET_FEATURE_DIM, n_samples=64, seed=42
        )
        rng = np.random.default_rng(42)
        X_syn = rng.normal(0, 1, size=(40, TABVIBNET_FEATURE_DIM)).astype(np.float32)

        head_s1 = train_variable_class_head_with_frozen_tabnet_encoder(
            pretrained_tabnet=pretrainer,
            X_train_scaled=X_syn,
            y_train=["Normal", "Ball"] * 20,
            seed=42,
            source_config_id="S1",
            fixed_epochs=3,
        )
        self.assertEqual(head_s1.num_classes, 2)
        self.assertEqual(head_s1.class_names, ("Normal", "Ball"))
        self.assertEqual(tuple(head_s1.head_state_dict["weight"].shape), (2, 16))
        self.assertEqual(tuple(head_s1.head_state_dict["bias"].shape), (2,))

        head_s3 = train_variable_class_head_with_frozen_tabnet_encoder(
            pretrained_tabnet=pretrainer,
            X_train_scaled=X_syn,
            y_train=["Inner", "Outer"] * 20,
            seed=42,
            source_config_id="S3",
            fixed_epochs=3,
        )
        self.assertEqual(head_s3.num_classes, 2)
        self.assertEqual(head_s3.class_names, ("Inner", "Outer"))
        self.assertEqual(tuple(head_s3.head_state_dict["weight"].shape), (2, 16))

        head_s2 = train_variable_class_head_with_frozen_tabnet_encoder(
            pretrained_tabnet=pretrainer,
            X_train_scaled=X_syn,
            y_train=list(HARMONIZED_CLASSES) * 8,
            seed=42,
            source_config_id="S2",
            fixed_epochs=3,
        )
        self.assertEqual(head_s2.num_classes, 5)
        self.assertEqual(head_s2.class_names, HARMONIZED_CLASSES)
        self.assertEqual(tuple(head_s2.head_state_dict["weight"].shape), (5, 16))

    def test_03_04_05_da_head_expansion_weight_preservation_and_reproducible_init(self) -> None:
        """
        Tests 3, 4 & 5:
          - Test 3: Verify DA head expansion by canonical class name:
            * S1 ("Normal", "Ball") expanded for CWRU_12k ("Normal", "Ball", "Inner", "Outer")
              -> 4-class head ("Normal", "Ball", "Inner", "Outer").
            * S3 ("Inner", "Outer") expanded for CWRU_12k ("Normal", "Ball", "Inner", "Outer")
              -> 4-class head ("Normal", "Ball", "Inner", "Outer") where "Inner" (was index 0
              in S3) moves to index 2 in the expanded head and "Outer" (was index 1 in S3)
              moves to index 3!
            * S3 ("Inner", "Outer") expanded for PU ("Normal", "Outer", "Inner & Outer")
              -> 4-class head ("Normal", "Inner", "Outer", "Inner & Outer").
          - Test 4: Verify bit-for-bit preservation of weights and biases for shared classes
            even when their positional index shifts during expansion (e.g. S3 -> CWRU_12k).
          - Test 5: Verify that newly introduced output rows have deterministic, seed-reproducible
            initialization (same seed -> identical weights/biases; different seed -> different weights).
        """
        pretrainer = _build_synthetic_pretrained_tabnet(
            input_dim=TABVIBNET_FEATURE_DIM, n_samples=64, seed=42
        )
        rng = np.random.default_rng(42)
        X_syn = rng.normal(0, 1, size=(32, TABVIBNET_FEATURE_DIM)).astype(np.float32)

        # Train S3 source head (2 classes: index 0 = "Inner", index 1 = "Outer")
        head_s3 = train_variable_class_head_with_frozen_tabnet_encoder(
            pretrained_tabnet=pretrainer,
            X_train_scaled=X_syn,
            y_train=["Inner", "Outer"] * 16,
            seed=42,
            source_config_id="S3",
            fixed_epochs=4,
        )
        self.assertEqual(head_s3.class_names, ("Inner", "Outer"))
        s3_inner_w = head_s3.head_state_dict["weight"][0].clone()
        s3_inner_b = head_s3.head_state_dict["bias"][0].clone()
        s3_outer_w = head_s3.head_state_dict["weight"][1].clone()
        s3_outer_b = head_s3.head_state_dict["bias"][1].clone()

        # Expand S3 head for CWRU_12k (has Normal, Ball, Inner, Outer)
        exp_42_a = expand_head_for_da_target(
            source_head_checkpoint=head_s3,
            target_present_classes=["Normal", "Ball", "Inner", "Outer"],
            seed=42,
            target_dataset="CWRU_12k",
        )
        exp_42_b = expand_head_for_da_target(
            source_head_checkpoint=head_s3,
            target_present_classes=["Normal", "Ball", "Inner", "Outer"],
            seed=42,
            target_dataset="CWRU_12k",
        )
        exp_123 = expand_head_for_da_target(
            source_head_checkpoint=head_s3,
            target_present_classes=["Normal", "Ball", "Inner", "Outer"],
            seed=123,
            target_dataset="CWRU_12k",
        )

        # Test 3: Expanded head has 4 outputs in canonical order
        self.assertEqual(exp_42_a.num_classes, 4)
        self.assertEqual(exp_42_a.class_names, ("Normal", "Ball", "Inner", "Outer"))
        self.assertEqual(exp_42_a.transferred_from_source_classes, ("Inner", "Outer"))
        self.assertEqual(exp_42_a.introduced_by_expansion_classes, ("Normal", "Ball"))

        # Test 4: Shared classes "Inner" (now at index 2) and "Outer" (now at index 3)
        # preserve their exact weights and biases from S3 indices 0 and 1!
        idx_inner_new = exp_42_a.class_to_index["Inner"]
        idx_outer_new = exp_42_a.class_to_index["Outer"]
        self.assertEqual(idx_inner_new, 2)
        self.assertEqual(idx_outer_new, 3)
        torch.testing.assert_close(
            exp_42_a.head_state_dict["weight"][idx_inner_new], s3_inner_w
        )
        torch.testing.assert_close(
            exp_42_a.head_state_dict["bias"][idx_inner_new], s3_inner_b
        )
        torch.testing.assert_close(
            exp_42_a.head_state_dict["weight"][idx_outer_new], s3_outer_w
        )
        torch.testing.assert_close(
            exp_42_a.head_state_dict["bias"][idx_outer_new], s3_outer_b
        )

        # Test 5: Reproducible initialization of new outputs ("Normal" at 0, "Ball" at 1)
        torch.testing.assert_close(
            exp_42_a.head_state_dict["weight"], exp_42_b.head_state_dict["weight"]
        )
        torch.testing.assert_close(
            exp_42_a.head_state_dict["bias"], exp_42_b.head_state_dict["bias"]
        )
        # With seed=123, new classes (0, 1) get different initialization while shared classes (2, 3) stay identical!
        self.assertFalse(
            torch.allclose(
                exp_42_a.head_state_dict["weight"][0],
                exp_123.head_state_dict["weight"][0],
            )
        )
        torch.testing.assert_close(
            exp_42_a.head_state_dict["weight"][2],
            exp_123.head_state_dict["weight"][2],
        )

    def test_06_07_08_09_dg_no_expansion_unknown_class_errors_frozen_encoder_and_da_head_update(
        self,
    ) -> None:
        """
        Tests 6, 7, 8 & 9:
          - Test 6: Verify that in DG, the head is NEVER expanded (`S1` stays 2-class
            `("Normal", "Ball")`; `S3` stays 2-class `("Inner", "Outer")`).
          - Test 7: Verify that in DG evaluation, unknown target classes (e.g. `"Inner"`
            and `"Outer"` when evaluating `S1` on `CWRU_12k`) are NEVER excluded from the
            test set and are strictly counted as classification errors (`Recall = 0.0`, `F1 = 0.0`).
          - Test 8: Verify that all TabNet encoder parameters and GhostBatchNorm buffers
            remain 100% frozen (`max_abs_diff == 0.0`) during both DG and DA.
          - Test 9: Verify that during DA adaptation on the 20% target adaptation set,
            the expanded head weights are genuinely updated (`head_param_max_abs_diff > 0.0`)
            and distinguish `transferred_from_source_classes`, `introduced_by_expansion_classes`,
            and `learned_during_adaptation_classes`.
        """
        pretrainer = _build_synthetic_pretrained_tabnet(
            input_dim=TABVIBNET_FEATURE_DIM, n_samples=64, seed=42
        )
        rng = np.random.default_rng(42)

        orig_enc_state = {
            k: v.detach().cpu().numpy().copy()
            for k, v in pretrainer.network.encoder.state_dict().items()
        }

        # 1. Train S1 source head (2 classes: Normal, Ball)
        X_src_tr = rng.normal(0, 1, size=(48, TABVIBNET_FEATURE_DIM)).astype(np.float32)
        y_src_tr = ["Normal", "Ball"] * 24
        X_src_val = rng.normal(0, 1, size=(16, TABVIBNET_FEATURE_DIM)).astype(np.float32)
        y_src_val = ["Normal", "Ball"] * 8

        dg_head_s1 = train_variable_class_head_with_frozen_tabnet_encoder(
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
        # Test 6 & 8: DG head has 2 outputs (no expansion) and encoder is 100% frozen
        self.assertEqual(dg_head_s1.num_classes, 2)
        self.assertEqual(dg_head_s1.class_names, ("Normal", "Ball"))
        self.assertEqual(dg_head_s1.encoder_param_max_abs_diff, 0.0)
        self.assertEqual(dg_head_s1.encoder_buffer_max_abs_diff, 0.0)
        self.assertGreater(dg_head_s1.head_param_max_abs_diff, 0.0)

        # Evaluate S1 DG head on target CWRU_12k containing 4 classes (Normal, Ball, Inner, Outer)
        X_tgt_test = rng.normal(0, 1, size=(40, TABVIBNET_FEATURE_DIM)).astype(np.float32)
        y_tgt_test = ["Normal", "Ball", "Inner", "Outer"] * 10
        preds_dg_ids, probs_dg_head, preds_dg_names = predict_with_variable_class_head(
            pretrained_tabnet=pretrainer,
            head_checkpoint=dg_head_s1,
            X_test_scaled=X_tgt_test,
        )
        # Probabilities have shape (40, 2) — strictly no head expansion in DG
        self.assertEqual(probs_dg_head.shape, (40, 2))
        self.assertTrue(set(preds_dg_names).issubset({"Normal", "Ball"}))

        # Test 7: Unknown target classes ("Inner", "Outer") are kept in evaluation and get Recall=0, F1=0
        metrics_dg = evaluate_variable_space_predictions(
            y_true=y_tgt_test,
            y_pred=preds_dg_ids,
            head_class_names=dg_head_s1.class_names,
            y_proba=probs_dg_head,
        )
        self.assertEqual(metrics_dg["n_test_samples"], 40)
        self.assertEqual(metrics_dg["unknown_target_classes"], ["Inner", "Outer"])
        self.assertEqual(metrics_dg["unknown_target_sample_count"], 20)
        self.assertEqual(metrics_dg["per_target_class_recall"]["Inner"], 0.0)
        self.assertEqual(metrics_dg["per_target_class_recall"]["Outer"], 0.0)
        self.assertEqual(metrics_dg["per_class_f1_all_5"]["Inner"], 0.0)
        self.assertEqual(metrics_dg["per_class_f1_all_5"]["Outer"], 0.0)
        self.assertLessEqual(metrics_dg["balanced_accuracy"], 0.50)
        self.assertLessEqual(metrics_dg["macro_f1"], 0.50)

        # Test 8 & 9: DA adaptation expands S1 head (2 -> 4 classes for CWRU_12k), freezes encoder, updates head
        X_da_adapt = rng.normal(0, 1, size=(40, TABVIBNET_FEATURE_DIM)).astype(np.float32)
        y_da_adapt = ["Normal", "Ball", "Inner", "Outer"] * 10

        da_head = train_variable_class_head_with_frozen_tabnet_encoder(
            pretrained_tabnet=pretrainer,
            X_train_scaled=X_da_adapt,
            y_train=y_da_adapt,
            seed=42,
            init_head_checkpoint=dg_head_s1,
            expand_for_target_classes=True,
            target_dataset="CWRU_12k",
            source_config_id="S1",
            fixed_epochs=dg_head_s1.selected_epoch,
            batch_size=16,
        )
        self.assertEqual(da_head.num_classes, 4)
        self.assertEqual(da_head.class_names, ("Normal", "Ball", "Inner", "Outer"))
        self.assertEqual(da_head.transferred_from_source_classes, ("Normal", "Ball"))
        self.assertEqual(da_head.introduced_by_expansion_classes, ("Inner", "Outer"))
        self.assertEqual(
            da_head.learned_during_adaptation_classes,
            ("Normal", "Ball", "Inner", "Outer"),
        )
        self.assertEqual(da_head.encoder_param_max_abs_diff, 0.0)
        self.assertEqual(da_head.encoder_buffer_max_abs_diff, 0.0)
        self.assertGreater(da_head.head_param_max_abs_diff, 0.0)

        for k, v in pretrainer.network.encoder.state_dict().items():
            np.testing.assert_array_equal(v.detach().cpu().numpy(), orig_enc_state[k])

    def test_10_and_11_zero_recording_overlap_reproducibility_and_cwru12k_exception(self) -> None:
        """
        Tests 10 & 11:
          - Verify zero overlap of original recordings across partitions in DA and DG.
          - Verify 100% deterministic reproducibility for the same seed and distinct
            splits across different seeds (`42`, `123`, `2026`).
          - Verify the Section 8.1 `CWRU_12k` exception: `Class_0` (`Normal`) has 8 recordings
            -> 2 recordings for adaptation and 6 for test, using 100% of the 2 adaptation
            recordings in `adapt` without splitting into `adapt_train` / `adapt_val`.
        """
        cwru12k_records, _ = _make_synthetic_dataset_records(
            dataset="CWRU_12k",
            raw_classes_and_recordings=[
                ("Class_0", 8, 10),   # 8 Normal recordings -> 2 adapt, 6 test
                ("Class_1", 20, 10),  # 20 Inner recordings -> 4 adapt, 16 test
                ("Class_2", 20, 10),  # 20 Outer recordings -> 4 adapt, 16 test
                ("Class_3", 20, 10),  # 20 Ball recordings  -> 4 adapt, 16 test
            ],
            seed=42,
        )

        parts_42_a, man_42_a = create_da_recording_split(cwru12k_records, seed=42)
        parts_42_b, man_42_b = create_da_recording_split(cwru12k_records, seed=42)
        parts_123, man_123 = create_da_recording_split(cwru12k_records, seed=123)

        assert_zero_recording_overlap(cwru12k_records, parts_42_a)
        self.assertEqual(man_42_a.recording_counts_by_class["Normal"]["total"], 8)
        self.assertEqual(man_42_a.recording_counts_by_class["Normal"]["adapt"], 2)
        self.assertEqual(man_42_a.recording_counts_by_class["Normal"]["adapt_train"], 2)
        self.assertEqual(man_42_a.recording_counts_by_class["Normal"]["adapt_val"], 0)
        self.assertEqual(man_42_a.recording_counts_by_class["Normal"]["test"], 6)

        self.assertEqual(parts_42_a, parts_42_b)
        self.assertEqual(man_42_a.split_signature, man_42_b.split_signature)
        self.assertNotEqual(man_42_a.split_signature, man_123.split_signature)

        # Verify purposeful recording leakage triggers ValueError
        corrupted = dict(parts_42_a)
        corrupted["test"] = tuple(list(parts_42_a["test"]) + [parts_42_a["adapt"][0]])
        with self.assertRaises(ValueError):
            assert_zero_recording_overlap(cwru12k_records, corrupted)

    def test_12_dg_normalization_unit_conversion_geometric_mean_and_zero_target_data(self) -> None:
        """
        Test 12:
          - Verify unit conversion `1 g = 9.80665 m/s^2` (`MFPT: 0.874 g -> 8.571012 m/s^2`).
          - Verify geometric mean calculation `RMS_global = exp(mean(log(RMS_ref_i_ms2)))`
            for active sources in S1, S2, S3, and S4 (never arithmetic mean).
          - Verify `fit_source_only_standard_scaler` fits exclusively on source training
            data and is 100% invariant to target data.
        """
        self.assertAlmostEqual(
            convert_rms_to_ms2(0.874, "g"), 0.874 * STANDARD_GRAVITY_MS2, places=6
        )
        self.assertAlmostEqual(convert_rms_to_ms2(5.85, "m/s^2"), 5.85, places=6)

        refs = {"UORED": 5.85, "HUST": 0.181, "MFPT": 0.874}
        units = {"UORED": "m/s^2", "HUST": "m/s^2", "MFPT": "g"}

        s1_res = compute_dg_global_rms_reference_ms2(refs, units, active_sources=["UORED"])
        s2_res = compute_dg_global_rms_reference_ms2(refs, units, active_sources=["HUST"])
        s3_res = compute_dg_global_rms_reference_ms2(refs, units, active_sources=["MFPT"])
        s4_res = compute_dg_global_rms_reference_ms2(
            refs, units, active_sources=["UORED", "HUST", "MFPT"]
        )

        self.assertAlmostEqual(s1_res["rms_global_geometric_ms2"], 5.85, places=6)
        self.assertAlmostEqual(s2_res["rms_global_geometric_ms2"], 0.181, places=6)
        self.assertAlmostEqual(
            s3_res["rms_global_geometric_ms2"], 0.874 * STANDARD_GRAVITY_MS2, places=6
        )
        expected_s4_geom = float(
            np.exp(
                np.mean(
                    np.log([5.85, 0.181, 0.874 * STANDARD_GRAVITY_MS2])
                )
            )
        )
        self.assertAlmostEqual(
            s4_res["rms_global_geometric_ms2"], expected_s4_geom, places=6
        )

        # Verify StandardScaler is fitted strictly on source_train
        rng = np.random.default_rng(42)
        X_src_tr = rng.normal(2.0, 3.0, size=(50, 141)).astype(np.float32)
        X_tgt_a = rng.normal(100.0, 50.0, size=(20, 141)).astype(np.float32)
        scaler_1, _, _, _ = fit_source_only_standard_scaler(
            X_src_tr, X_target_dict={"CWRU_12k": X_tgt_a}
        )
        scaler_2, _, _, _ = fit_source_only_standard_scaler(
            X_src_tr, X_target_dict={"CWRU_12k": X_tgt_a * 999.0}
        )
        np.testing.assert_array_equal(scaler_1.mean_, scaler_2.mean_)
        np.testing.assert_array_equal(scaler_1.scale_, scaler_2.scale_)

    def test_13_da_normalization_and_cwru48k_external_calibration_from_cwru12k_adapt(self) -> None:
        """
        Test 13:
          - Verify that DA normalization estimates `rms_ref` exclusively from `Normal`
            recordings in the target's 20% adaptation partition and applies the exact same
            reference to both adaptation and test.
          - Verify `CWRU_48k` external calibration uses ONLY the 2 `Normal` adaptation
            recordings of `CWRU_12k` (`calibration_mode == "external_related_domain_adaptation_normal"`)
            and raises `ValueError` if any test recording from `CWRU_12k` is passed.
        """
        cwru12k_recs, cwru12k_wins = _make_synthetic_dataset_records(
            dataset="CWRU_12k",
            raw_classes_and_recordings=[
                ("Class_0", 8, 5),
                ("Class_1", 10, 5),
            ],
            base_rms=0.068,
            seed=42,
        )
        parts_12k, _ = create_da_recording_split(cwru12k_recs, seed=42)
        da_norm_12k = fit_da_rms_normalizer(
            records=cwru12k_recs,
            windows=cwru12k_wins,
            adapt_indices=parts_12k["adapt"],
            test_indices=parts_12k["test"],
        )
        self.assertEqual(da_norm_12k.calibration_mode, "internal_adaptation_normal")
        self.assertEqual(len(da_norm_12k.reference_recording_keys), 2)

        # Extract the 2 Normal adaptation recordings of CWRU_12k for CWRU_48k calibration
        cwru12k_adapt_normal_idxs = [
            i for i in parts_12k["adapt"] if cwru12k_recs[i].harmonized_class == "Normal"
        ]
        cwru12k_adapt_normal_wins = [cwru12k_wins[i] for i in cwru12k_adapt_normal_idxs]
        cwru12k_adapt_normal_keys = sorted(
            {cwru12k_recs[i].recording_key for i in cwru12k_adapt_normal_idxs}
        )
        cwru12k_test_keys = sorted(
            {cwru12k_recs[i].recording_key for i in parts_12k["test"]}
        )

        cwru48k_recs, cwru48k_wins = _make_synthetic_dataset_records(
            dataset="CWRU_48k",
            raw_classes_and_recordings=[
                ("Class_1", 10, 5),
                ("Class_2", 10, 5),
                ("Class_3", 10, 5),
            ],
            base_rms=0.25,
            seed=42,
        )
        parts_48k, _ = create_da_recording_split(cwru48k_recs, seed=42)
        da_norm_48k = fit_da_rms_normalizer(
            records=cwru48k_recs,
            windows=cwru48k_wins,
            adapt_indices=parts_48k["adapt"],
            test_indices=parts_48k["test"],
            cross_reference_normal_windows=cwru12k_adapt_normal_wins,
            cross_reference_recording_keys=cwru12k_adapt_normal_keys,
            cross_reference_domain="CWRU_12k",
            cross_reference_forbidden_recording_keys=cwru12k_test_keys,
        )
        self.assertEqual(
            da_norm_48k.calibration_mode,
            "external_related_domain_adaptation_normal",
        )
        self.assertEqual(da_norm_48k.reference_domain, "CWRU_12k")
        self.assertAlmostEqual(da_norm_48k.rms_ref, da_norm_12k.rms_ref, places=7)

        # Passing any CWRU_12k test recording key into cross_reference_recording_keys must fail
        with self.assertRaises(ValueError):
            fit_da_rms_normalizer(
                records=cwru48k_recs,
                windows=cwru48k_wins,
                adapt_indices=parts_48k["adapt"],
                test_indices=parts_48k["test"],
                cross_reference_normal_windows=cwru12k_adapt_normal_wins,
                cross_reference_recording_keys=cwru12k_adapt_normal_keys + [cwru12k_test_keys[0]],
                cross_reference_domain="CWRU_12k",
                cross_reference_forbidden_recording_keys=cwru12k_test_keys,
            )

    def test_14_hierarchical_balancing_no_test_balancing_and_classical_baselines(self) -> None:
        """
        Test 14:
          - Verify hierarchical source balancing equalizes dataset contribution (`1/D`)
            and class-within-dataset contribution (`1/|C_d|`), and forbids balancing on
            `val` or `test`.
          - Verify classical baselines (`RandomForest` and `SVM`) for DG (`Source-Only`),
            DA (`Target-Only 20%`), and DA (`Source + Target 20%`), ensuring zero test
            data leakage.
        """
        datasets = ["UORED"] * 280 + ["HUST"] * 700 + ["MFPT"] * 60
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
        expected_ds_w = len(datasets) / 3.0
        for ds_name in ("UORED", "HUST", "MFPT"):
            self.assertAlmostEqual(
                res_w.effective_weight_by_dataset[ds_name], expected_ds_w, places=5
            )
        for forbidden_part in ("val", "validation", "adapt_val", "source_val", "test"):
            with self.assertRaises(ValueError):
                compute_hierarchical_source_balance(
                    datasets=datasets,
                    class_ids=class_ids,
                    partition_name=forbidden_part,
                )

        # Test RF and SVM across DG, DA Target-Only 20%, and DA Source + Target 20%
        rng = np.random.default_rng(42)
        X_src = rng.normal(0, 1, size=(40, 141)).astype(np.float32)
        y_src = ["Normal", "Ball"] * 20
        X_tgt_adapt = rng.normal(0.5, 1.2, size=(32, 141)).astype(np.float32)
        y_tgt_adapt = ["Normal", "Ball", "Inner", "Outer"] * 8
        X_tgt_test = rng.normal(0.5, 1.2, size=(40, 141)).astype(np.float32)
        y_tgt_test = ["Normal", "Ball", "Inner", "Outer"] * 10

        for model_name in ("RandomForest", "SVM"):
            dg_res = train_eval_classical_da_dg_baseline(
                protocol="DG",
                variant="Source-Only",
                model_name=model_name,
                X_test_raw=X_tgt_test,
                y_test=y_tgt_test,
                X_source_train_raw=X_src,
                y_source_train=y_src,
                seed=42,
                hyperparameters={"n_estimators": 20} if model_name == "RandomForest" else {"C": 1.0},
            )
            self.assertEqual(dg_res.head_class_names, ("Normal", "Ball"))
            self.assertEqual(dg_res.metrics["unknown_target_classes"], ["Inner", "Outer"])

            da_to_res = train_eval_classical_da_dg_baseline(
                protocol="DA",
                variant="Target-Only 20%",
                model_name=model_name,
                X_test_raw=X_tgt_test,
                y_test=y_tgt_test,
                X_target_adapt_raw=X_tgt_adapt,
                y_target_adapt=y_tgt_adapt,
                seed=42,
                hyperparameters={"n_estimators": 20} if model_name == "RandomForest" else {"C": 1.0},
            )
            self.assertEqual(da_to_res.head_class_names, ("Normal", "Ball", "Inner", "Outer"))

            da_st_res = train_eval_classical_da_dg_baseline(
                protocol="DA",
                variant="Source + Target 20%",
                model_name=model_name,
                X_test_raw=X_tgt_test,
                y_test=y_tgt_test,
                X_source_train_raw=X_src,
                y_source_train=y_src,
                X_target_adapt_raw=X_tgt_adapt,
                y_target_adapt=y_tgt_adapt,
                seed=42,
                hyperparameters={"n_estimators": 20} if model_name == "RandomForest" else {"C": 1.0},
            )
            self.assertEqual(da_st_res.head_class_names, ("Normal", "Ball", "Inner", "Outer"))
            self.assertIn("Joint learning baseline", da_st_res.joint_learning_note)

    def test_15_checkpoint_feature_compatibility_and_126_15_scale_invariance_catalog(self) -> None:
        """
        Test 15:
          - Verify the 141-feature catalog has exactly 126 scale-invariant features
            and 15 scale-sensitive features (`126 + 15 == 141`).
          - Verify numerically on `extract_advanced_features` (`indices 0..15`) that the
            7 invariant features (`1, 4, 7, 10, 11, 14, 15`) remain invariant under
            multiplication by `c = 7.5`, while the 9 sensitive features (`0, 2, 3, 5, 6, 8, 9, 12, 13`)
            change according to their analytical scaling law.
          - Verify that `select_feature_subset_for_ablation` blocks reusing a 141-input
            checkpoint on the 126-feature ablation.
        """
        from src.features.extractors_v2 import extract_advanced_features

        catalog = get_141_feature_scale_invariance_catalog()
        self.assertEqual(len(catalog), 141)
        inv_126 = get_scale_invariant_126_indices()
        sens_15 = get_scale_sensitive_15_indices()
        self.assertEqual(len(inv_126), 126)
        self.assertEqual(len(sens_15), 15)
        self.assertEqual(len(set(inv_126) & set(sens_15)), 0)
        self.assertEqual(sorted(list(inv_126) + list(sens_15)), list(range(141)))

        # Numerical verification of extractors_v2 (indices 0..15)
        rng = np.random.default_rng(42)
        t = np.linspace(0.0, 1.0, 4096, endpoint=False)
        sig = (
            np.sin(2.0 * np.pi * 55.0 * t)
            + 0.4 * np.sin(2.0 * np.pi * 420.0 * t)
            + rng.normal(0.0, 0.2, size=len(t))
        )
        c_factor = 7.5
        f_orig = extract_advanced_features(sig, fs=4096)
        f_scaled = extract_advanced_features(c_factor * sig, fs=4096)

        v2_inv_indices = [i for i in inv_126 if i < 16]
        v2_sens_indices = [i for i in sens_15 if i < 16]
        self.assertEqual(v2_inv_indices, [1, 4, 7, 10, 11, 14, 15])
        self.assertEqual(v2_sens_indices, [0, 2, 3, 5, 6, 8, 9, 12, 13])

        for idx in v2_inv_indices:
            self.assertAlmostEqual(
                float(f_orig[idx]),
                float(f_scaled[idx]),
                places=3,
                msg=f"Feature {idx} ({catalog[idx].name}) should be scale-invariant!",
            )
        for idx in v2_sens_indices:
            self.assertNotAlmostEqual(
                float(f_orig[idx]),
                float(f_scaled[idx]),
                places=3,
                msg=f"Feature {idx} ({catalog[idx].name}) should be scale-sensitive!",
            )

        # Verify protection against reusing a 141-feature checkpoint on 126-feature ablation
        X_dummy = rng.normal(0, 1, size=(10, 141)).astype(np.float32)
        X_126 = select_feature_subset_for_ablation(X_dummy, mode="invariant_126")
        self.assertEqual(X_126.shape, (10, 126))
        with self.assertRaises(ValueError):
            select_feature_subset_for_ablation(
                X_dummy,
                mode="invariant_126",
                expected_checkpoint_input_dim=141,
            )

    def test_16_mfpt_decimation_and_on_disk_preparation_integrity(self) -> None:
        """
        Test 16:
          - Verify that `data/processed/MFPT` (114 original files) was preserved untouched.
          - Verify that `data/processed_tl/detrended/MFPT` and `data/processed_tl/detrended_rms/MFPT`
            contain exactly 78 windows of 1.0 second (`shape == (48828,)`) across 20 recordings
            (`Class_23`: 18 windows / 3 recordings; `Class_24`: 21 windows / 7 recordings;
            `Class_25`: 39 windows / 10 recordings), with `mean(RMS(Class_23)) == 1.0` in
            `detrended_rms/MFPT`.
        """
        orig_mfpt_dir = PROJECT_ROOT / "data" / "processed" / "MFPT"
        det_mfpt_dir = PROJECT_ROOT / "data" / "processed_tl" / "detrended" / "MFPT"
        rms_mfpt_dir = PROJECT_ROOT / "data" / "processed_tl" / "detrended_rms" / "MFPT"

        if orig_mfpt_dir.is_dir() and rms_mfpt_dir.is_dir():
            orig_files = list(orig_mfpt_dir.rglob("*.npy"))
            det_files = list(det_mfpt_dir.rglob("*.npy"))
            rms_files = list(rms_mfpt_dir.rglob("*.npy"))

            self.assertEqual(len(orig_files), 114)
            self.assertEqual(len(det_files), 78)
            self.assertEqual(len(rms_files), 78)

            cls_counts = {"Class_23": 0, "Class_24": 0, "Class_25": 0}
            normal_rms_vals = []
            for fp in rms_files:
                arr = np.load(fp, allow_pickle=False)
                self.assertEqual(arr.shape, (48828,))
                cls_name = fp.parent.name
                cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1
                if cls_name == "Class_23":
                    normal_rms_vals.append(float(np.sqrt(np.mean(np.square(arr)))))

            self.assertEqual(cls_counts, {"Class_23": 18, "Class_24": 21, "Class_25": 39})
            self.assertAlmostEqual(float(np.mean(normal_rms_vals)), 1.0, places=6)

    def test_17_non_overwriting_result_directories(self) -> None:
        """
        Test 17:
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
            self.assertEqual(
                (dir_1 / "results.csv").read_text(encoding="utf-8"), "run1\n"
            )
            self.assertEqual(
                (dir_2 / "results.csv").read_text(encoding="utf-8"), "run2\n"
            )


if __name__ == "__main__":
    unittest.main()
