"""
Testes automatizados para os Baselines de Machine Learning Clássico
complementares ao Benchmark Scratch vs. Transfer Learning.

Cobre:
1. Validação do manifesto real de referência (396 entradas em run_20261105_141743).
2. Recuperação determinística de folds a partir do fold_manifest.csv com verificação
   estrita de split_signature e preprocessing_signature.
3. Bloqueio explícito (ValueError) em caso de divergência de sample_ids
   ou assinaturas, sem geração de splits substitutos.
4. Treinamento, predição probabilística (incluindo projeção segura para todas as
   classes do dataset) e cálculo de métricas para RandomForest, SVM e XGBoost.
5. Execução piloto em pequena escala, retomada incremental (--resume-dir),
   validação de integridade pós-execução e consolidação estatística conjunta
   (Scratch vs. TL vs. Classical Baselines) preservando 100% os artefatos originais.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.analyze_scratch_tl_classical_comparison import (
    run_consolidated_comparison,
)
from src.experiments.run_baseline_loco_v2 import (
    FEATURE_DIMENSION,
    PREPROCESSING_VARIANT,
     TASK_NAME,
    apply_smote_with_audit,
    compute_sample_split_signature,
    load_and_extract_target_dataset,
    project_predictions_to_full_vocabulary,
    scale_loco_fold_strict,
)
from src.experiments.run_scratch_vs_tl_benchmark import (
    compute_preprocessing_signature,
)
from src.experiments.run_scratch_vs_tl_classical_baselines import (
    CLASSICAL_ARCHITECTURES,
    CLASSICAL_METHOD,
    load_and_validate_reference_manifest,
    recover_and_verify_fold_from_manifest,
    run_classical_baselines_benchmark,
    train_eval_classical_model,
    validate_post_execution_integrity,
)

REAL_MANIFEST_PATH = (
    REPO_ROOT
    / "results"
    / "scratch_vs_tl"
    / "run_20261105_141743"
    / "fold_manifest.csv"
)


def _fast_synthetic_feature_extractor(signals: np.ndarray, fs: int) -> np.ndarray:
    """Extrator determinístico rápido de 25 features para testes unitários sintéticos."""
    n = signals.shape[0]
    feats = np.zeros((n, FEATURE_DIMENSION), dtype=np.float64)
    feats[:, 0] = np.mean(signals, axis=1)
    feats[:, 1] = np.std(signals, axis=1)
    feats[:, 2] = np.sqrt(np.mean(signals**2, axis=1))
    feats[:, 3] = np.max(np.abs(signals), axis=1)
    feats[:, 4] = np.ptp(signals, axis=1)
    fft_mag = np.abs(np.fft.rfft(signals, axis=1))
    for i in range(20):
        col_slice = fft_mag[:, i * 4 : (i + 1) * 4]
        feats[:, 5 + i] = np.mean(col_slice, axis=1)
    return feats


def _create_synthetic_dataset_tree(root: Path) -> Path:
    """Cria datasets sintéticos em formato .npy compatível com detrended_rms."""
    rng = np.random.default_rng(2026)
    data_root = root / "detrended_rms"

    datasets_spec = {
        "CWRU_12k": {
            "conditions": ["Cond_0", "Cond_1", "Cond_2", "Cond_3"],
            "classes": ["Class_Normal", "Class_Inner", "Class_Outer"],
            "n_windows": 8,
            "freq_shift": 0.0,
        },
        "PU": {
            "conditions": ["Cond_A", "Cond_B", "Cond_C", "Cond_D"],
            "classes": ["Class_Normal", "Class_Fault1", "Class_Fault2"],
            "n_windows": 8,
            "freq_shift": 5.0,
        },
    }

    t = np.linspace(0.0, 1.0, 256, endpoint=False)
    for ds_name, spec in datasets_spec.items():
        for cond_idx, cond in enumerate(spec["conditions"]):
            for cls_idx, cls_name in enumerate(spec["classes"]):
                folder = data_root / ds_name / cond / cls_name
                folder.mkdir(parents=True, exist_ok=True)
                n_w = spec["n_windows"]
                signals = np.zeros((n_w, 256), dtype=np.float32)
                base_freq = 10.0 + spec["freq_shift"] + 12.0 * cls_idx + 1.5 * cond_idx
                for i in range(n_w):
                    amp = 1.0 + 0.4 * cls_idx
                    noise = rng.normal(0.0, 0.15 + 0.05 * cls_idx, size=t.shape)
                    signals[i] = (
                        amp * np.sin(2.0 * np.pi * base_freq * t) + noise
                    ).astype(np.float32)
                np.save(folder / "signals.npy", signals)

    return data_root


def _build_synthetic_reference_benchmark(
    data_root: Path,
    ref_dir: Path,
    targets: List[str],
    seeds: List[int],
    balance_strategies: List[str],
) -> Path:
    """
    Constrói um diretório de benchmark de referência sintético contendo
    fold_manifest.csv e results.csv com o mesmo esquema e assinaturas do benchmark real.
    """
    ref_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, object]] = []
    results_rows: List[Dict[str, object]] = []
    logger = logging.getLogger("synth_ref_builder")

    for ds_name in targets:
        X_all, y_all_str, conds_all, sample_ids_all, _ = load_and_extract_target_dataset(
            dataset_name=ds_name,
            data_root=data_root,
            feature_extractor=_fast_synthetic_feature_extractor,
            logger=logger,
            enforce_expected_folds=False,
        )
        le = LabelEncoder()
        y_all_enc = le.fit_transform(y_all_str).astype(np.int64)
        class_names = [str(c) for c in le.classes_]
        conditions = sorted(np.unique(conds_all).tolist())

        for fold_idx, test_cond in enumerate(conditions):
            test_mask = conds_all == test_cond
            train_mask = ~test_mask
            X_train_raw = X_all[train_mask]
            y_train_raw = y_all_enc[train_mask]
            X_test_raw = X_all[test_mask]
            y_test = y_all_enc[test_mask]
            train_sids = [str(s) for s in sample_ids_all[train_mask].tolist()]
            test_sids = [str(s) for s in sample_ids_all[test_mask].tolist()]

            split_sig = compute_sample_split_signature(train_sids, test_sids)
            X_train_scaled, _, scaler = scale_loco_fold_strict(X_train_raw, X_test_raw)

            for seed in seeds:
                for bal_strat in balance_strategies:
                    X_tr_bal, y_tr_bal, smote_audit = apply_smote_with_audit(
                        X_train_scaled=X_train_scaled,
                        y_train_encoded=y_train_raw,
                        class_names=class_names,
                        strategy=bal_strat,
                        seed=int(seed),
                    )
                    prep_sig = compute_preprocessing_signature(
                        preprocessing_variant=PREPROCESSING_VARIANT,
                        feature_dim=int(X_all.shape[1]),
                        scaler_mean=scaler.mean_,
                        scaler_scale=scaler.scale_,
                        balance_strategy=bal_strat,
                        smote_params=smote_audit["smote_params"],
                    )
                    manifest_rows.append(
                        {
                            "target_dataset": ds_name,
                            "fold_id": fold_idx,
                            "condition": str(test_cond),
                            "train_sample_ids": json.dumps(train_sids),
                            "test_sample_ids": json.dumps(test_sids),
                            "seed": int(seed),
                            "balance_strategy": bal_strat,
                            "split_signature": split_sig,
                            "preprocessing_signature": prep_sig,
                            "n_train_before_balance": int(smote_audit["n_train_before"]),
                            "n_train_after_balance": int(smote_audit["n_train_after"]),
                            "n_test_samples": int(len(y_test)),
                            "class_counts_before": json.dumps(
                                smote_audit["counts_before"], ensure_ascii=False
                            ),
                            "class_counts_after": json.dumps(
                                smote_audit["counts_after"], ensure_ascii=False
                            ),
                        }
                    )

                    for arch in ["MLP", "TabNet"]:
                        for method in ["Scratch", "FineTuning", "LinearProbing"]:
                            results_rows.append(
                                {
                                    "target_dataset": ds_name,
                                    "task": TASK_NAME,
                                    "fold_id": fold_idx,
                                    "condition": str(test_cond),
                                    "seed": int(seed),
                                    "architecture": arch,
                                    "method": method,
                                    "balance_strategy": bal_strat,
                                    "source_datasets": (
                                        "None" if method == "Scratch" else "SRC_SYNTH"
                                    ),
                                    "preprocessing": PREPROCESSING_VARIANT,
                                    "feature_dimension": int(X_all.shape[1]),
                                    "accuracy": 0.78,
                                    "balanced_accuracy": 0.78,
                                    "macro_f1": 0.77,
                                    "roc_auc": 0.89,
                                    "roc_auc_valid": True,
                                    "roc_auc_note": "OK",
                                    "test_sample_count": int(len(y_test)),
                                    "test_class_count": len(class_names),
                                    "train_sample_count_before": int(
                                        smote_audit["n_train_before"]
                                    ),
                                    "train_sample_count_after": int(
                                        smote_audit["n_train_after"]
                                    ),
                                    "train_class_count": len(class_names),
                                    "class_counts_before": json.dumps(
                                        smote_audit["counts_before"], ensure_ascii=False
                                    ),
                                    "class_counts_after": json.dumps(
                                        smote_audit["counts_after"], ensure_ascii=False
                                    ),
                                    "backbone_frozen": method == "LinearProbing",
                                    "backbone_param_max_abs_diff": 0.0,
                                    "backbone_buffer_max_abs_diff": 0.0,
                                    "selected_epochs": 5,
                                    "split_signature": split_sig,
                                    "preprocessing_signature": prep_sig,
                                    "status": "OK",
                                    "error_message": "",
                                    "runtime_seconds": 0.12,
                                }
                            )

    pd.DataFrame(manifest_rows).to_csv(ref_dir / "fold_manifest.csv", index=False)
    pd.DataFrame(results_rows).to_csv(ref_dir / "results.csv", index=False)
    return ref_dir


def test_real_reference_manifest_validation():
    """Verifica que o fold_manifest.csv real de run_20261105_141743 possui 396 entradas íntegras."""
    if not REAL_MANIFEST_PATH.exists():
        if pytest is not None:
            pytest.skip("fold_manifest.csv real não encontrado no workspace.")
        return

    manifest_df = load_and_validate_reference_manifest(
        manifest_path=REAL_MANIFEST_PATH,
        enforce_full_396=True,
    )
    assert len(manifest_df) == 396
    assert set(manifest_df["target_dataset"].unique()) == {
        "CWRU_12k",
        "CWRU_48k",
        "HUST_Gearbox",
        "UOEMD",
        "PU",
    }


def test_fold_recovery_and_signature_enforcement(tmp_path: Path):
    """
    Testa que recover_and_verify_fold_from_manifest:
    1. Recupera exatamente os mesmos sample_ids e reproduz split_signature e preprocessing_signature;
    2. Levanta ValueError se houver divergência em sample_ids ou assinaturas.
    """
    data_root = _create_synthetic_dataset_tree(tmp_path)
    ref_dir = _build_synthetic_reference_benchmark(
        data_root=data_root,
        ref_dir=tmp_path / "ref_run",
        targets=["CWRU_12k"],
        seeds=[42],
        balance_strategies=["none", "smote"],
    )
    manifest_df = pd.read_csv(ref_dir / "fold_manifest.csv")
    logger = logging.getLogger("test_recovery")
    X_all, y_all_str, conds_all, sample_ids_all, _ = load_and_extract_target_dataset(
        dataset_name="CWRU_12k",
        data_root=data_root,
        feature_extractor=_fast_synthetic_feature_extractor,
        logger=logger,
        enforce_expected_folds=False,
    )
    le = LabelEncoder()
    y_all_enc = le.fit_transform(y_all_str).astype(np.int64)
    class_names = [str(c) for c in le.classes_]
    sample_id_to_idx = {
        str(sid): int(idx) for idx, sid in enumerate(sample_ids_all.tolist())
    }

    for _, m_row in manifest_df.iterrows():
        recovered = recover_and_verify_fold_from_manifest(
            manifest_row=m_row,
            X_all=X_all,
            y_all_enc=y_all_enc,
            conds_all=conds_all,
            sample_id_to_idx=sample_id_to_idx,
            class_names=class_names,
        )
        assert recovered["split_signature"] == m_row["split_signature"]
        assert recovered["preprocessing_signature"] == m_row["preprocessing_signature"]
        assert len(recovered["y_test"]) == int(m_row["n_test_samples"])

    # Testa falha explícita quando split_signature diverge
    corrupted_row = manifest_df.iloc[0].to_dict()
    corrupted_row["split_signature"] = "0000000000000000"
    with pytest.raises(ValueError, match="split_signature mismatch"):
        recover_and_verify_fold_from_manifest(
            manifest_row=corrupted_row,
            X_all=X_all,
            y_all_enc=y_all_enc,
            conds_all=conds_all,
            sample_id_to_idx=sample_id_to_idx,
            class_names=class_names,
        )

    # Testa falha explícita quando preprocessing_signature diverge
    corrupted_prep = manifest_df.iloc[0].to_dict()
    corrupted_prep["preprocessing_signature"] = "ffffffffffffffff"
    with pytest.raises(ValueError, match="preprocessing_signature mismatch"):
        recover_and_verify_fold_from_manifest(
            manifest_row=corrupted_prep,
            X_all=X_all,
            y_all_enc=y_all_enc,
            conds_all=conds_all,
            sample_id_to_idx=sample_id_to_idx,
            class_names=class_names,
        )

    # Testa falha explícita quando sample_id não existe no dataset
    corrupted_sid = manifest_df.iloc[0].to_dict()
    corrupted_sid["test_sample_ids"] = json.dumps(["CWRU_12k::Cond_0::Class_Normal::999999"])
    with pytest.raises(ValueError, match="Cannot resolve sample IDs"):
        recover_and_verify_fold_from_manifest(
            manifest_row=corrupted_sid,
            X_all=X_all,
            y_all_enc=y_all_enc,
            conds_all=conds_all,
            sample_id_to_idx=sample_id_to_idx,
            class_names=class_names,
        )


def test_probability_projection_missing_class():
    """Garante que project_predictions_to_full_vocabulary alinha colunas quando falta uma classe no treino."""
    preds_local = np.array([0, 1], dtype=np.int64)
    probs_local = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
    present_classes = np.array([0, 2], dtype=np.int64)
    preds_global, probs_global = project_predictions_to_full_vocabulary(
        preds_local=preds_local,
        probs_local=probs_local,
        present_classes=present_classes,
        n_total_classes=3,
    )
    assert preds_global.tolist() == [0, 2]
    assert probs_global.shape == (2, 3)
    np.testing.assert_allclose(probs_global[:, 1], 0.0)
    np.testing.assert_allclose(probs_global.sum(axis=1), 1.0)


def test_end_to_end_pilot_resume_and_consolidated_comparison(tmp_path: Path):
    """
    Executa piloto completo com RandomForest, SVM e XGBoost em 2 datasets sintéticos,
    verifica retomada incremental (--resume-dir), integridade do run de referência
    e análise estatística consolidada.
    """
    data_root = _create_synthetic_dataset_tree(tmp_path)
    ref_dir = _build_synthetic_reference_benchmark(
        data_root=data_root,
        ref_dir=tmp_path / "ref_run",
        targets=["CWRU_12k", "PU"],
        seeds=[42, 123],
        balance_strategies=["none", "smote"],
    )

    ref_manifest_sha_before = hashlib.sha256(
        (ref_dir / "fold_manifest.csv").read_bytes()
    ).hexdigest()
    ref_results_sha_before = hashlib.sha256(
        (ref_dir / "results.csv").read_bytes()
    ).hexdigest()

    cfg_path = tmp_path / "classical_cfg.yaml"
    cfg_dict = {
        "experiment_id": "scratch_vs_tl_classical_baselines_test",
        "task": "diagnosis",
        "reference_manifest_path": str(ref_dir / "fold_manifest.csv"),
        "output_root": str(tmp_path / "classical_out"),
        "target_datasets": ["CWRU_12k", "PU"],
        "architectures": ["RandomForest", "SVM", "XGBoost"],
        "method": CLASSICAL_METHOD,
        "balance_strategies": ["none", "smote"],
        "seeds": [42, 123],
        "hyperparameter_tuning": {
            "enabled": True,
            "strategy": "grid_search_stratified_kfold",
            "cv_folds": 3,
            "scoring": "macro_f1",
            "inner_scaling_and_smote_strict": True,
            "param_grids": {
                "RandomForest": {
                    "n_estimators": [10, 20],
                    "max_depth": [None, 5],
                },
                "SVM": {
                    "kernel": ["rbf"],
                    "C": [0.5, 2.0],
                },
                "XGBoost": {
                    "n_estimators": [10, 15],
                    "max_depth": [2, 3],
                },
            },
        },
        "hyperparameters": {
            "RandomForest": {
                "n_estimators": 20,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": True,
                "class_weight": None,
                "n_jobs": 1,
            },
            "SVM": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "probability": True,
                "class_weight": None,
            },
            "XGBoost": {
                "objective": "multi:softprob",
                "n_estimators": 15,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "min_child_weight": 1,
                "eval_metric": "mlogloss",
                "tree_method": "hist",
                "n_jobs": 1,
            },
        },
    }
    cfg_path.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    # Executa benchmark clássico completo no dataset sintético:
    # 2 datasets x 4 folds x 2 seeds x 2 balance_strategies x 3 modelos = 96 avaliações
    out_dir = run_classical_baselines_benchmark(
        config_path=cfg_path,
        manifest_path=ref_dir / "fold_manifest.csv",
        data_root=data_root,
        output_root=tmp_path / "classical_out",
        feature_extractor=_fast_synthetic_feature_extractor,
        enforce_expected_folds=False,
        enforce_full_manifest=False,
        is_synthetic_smoke=True,
    )

    results_df = pd.read_csv(out_dir / "results.csv")
    assert len(results_df) == 96
    assert (results_df["status"] == "OK").all()
    assert set(results_df["architecture"].unique()) == set(CLASSICAL_ARCHITECTURES)
    assert set(results_df["method"].unique()) == {CLASSICAL_METHOD}
    assert (results_df["source_datasets"] == "None").all()
    assert "selected_hyperparameters" in results_df.columns
    assert "inner_cv_macro_f1" in results_df.columns
    assert results_df["inner_cv_macro_f1"].notna().all()
    for hp_str in results_df["selected_hyperparameters"]:
        parsed_hp = json.loads(hp_str)
        assert isinstance(parsed_hp, dict) and len(parsed_hp) > 0

    integrity = json.loads(
        (out_dir / "integrity_validation.json").read_text(encoding="utf-8")
    )
    assert integrity["overall_valid"] is True

    # Testa retomada (--resume-dir): remove últimas 6 avaliações de results.csv e retoma
    trimmed_df = results_df.iloc[:-6].copy()
    trimmed_df.to_csv(out_dir / "results.csv", index=False)
    resumed_dir = run_classical_baselines_benchmark(
        config_path=cfg_path,
        manifest_path=ref_dir / "fold_manifest.csv",
        data_root=data_root,
        output_root=tmp_path / "classical_out",
        resume_dir=out_dir,
        feature_extractor=_fast_synthetic_feature_extractor,
        enforce_expected_folds=False,
        enforce_full_manifest=False,
        is_synthetic_smoke=True,
    )
    assert resumed_dir == out_dir
    resumed_df = pd.read_csv(out_dir / "results.csv")
    assert len(resumed_df) == 96
    assert (resumed_df["status"] == "OK").all()

    # Executa consolidação estatística conjunta (Scratch vs. TL vs. Classical Baselines)
    comp_dir = run_consolidated_comparison(
        neural_run_dir=ref_dir,
        classical_run_dir=out_dir,
        output_dir=tmp_path / "consolidated_comparison",
    )
    assert (comp_dir / "consolidated_results.csv").exists()
    assert (comp_dir / "statistical_tests.csv").exists()
    assert (comp_dir / "rankings.csv").exists()
    assert (comp_dir / "report.md").exists()

    combined_df = pd.read_csv(comp_dir / "consolidated_results.csv")
    # 192 avaliações neurais + 96 avaliações clássicas = 288
    assert len(combined_df) == 288

    # Verifica que os arquivos originais do benchmark de referência não foram tocados
    ref_manifest_sha_after = hashlib.sha256(
        (ref_dir / "fold_manifest.csv").read_bytes()
    ).hexdigest()
    ref_results_sha_after = hashlib.sha256(
        (ref_dir / "results.csv").read_bytes()
    ).hexdigest()
    assert ref_manifest_sha_before == ref_manifest_sha_after
    assert ref_results_sha_before == ref_results_sha_after
