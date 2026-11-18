#!/usr/bin/env python3
"""
Single Domain Generalization (DG) Pilot Runner (`src/experiments/run_dg_pilot.py`).

Executes ONLY a single controlled DG pilot (by default: S4 [UORED + HUST + MFPT] ->
CWRU_12k, seed=42, 141 features) following the strict leak-free VibNet-1D protocol:
  1. Loads unnormalized physical waveforms (detrended/native unit):
       - UORED: `data/processed/UORED` (m/s^2, 42,000 Hz)
       - HUST:  `data/processed/HUST` (m/s^2, 51,200 Hz)
       - MFPT:  `data/processed_tl/detrended/MFPT` (g, 48,828 Hz, 78 harmonized 1s windows)
       - CWRU_12k: `data/processed/CWRU_12k` (g, 12,000 Hz)
  2. Partitions source recordings into `source_train` (80%), `source_val` (20%),
     `pretrain_only_train` (80% of additional unsupervised source classes: Class_Cage,
     Class_44, Class_46, MFPT Class_23), and `pretrain_only_val` (20%) BEFORE any
     normalization or pretraining.
  3. Fits `DGRMSNormalizer` EXCLUSIVELY on healthy (`Normal` / `Class_23`) windows
     from source training recordings (`pretrain_train`), converting all sources to m/s^2
     and computing `RMS_global = exp(mean(log(RMS_ref_i_ms2)))`. Applies the exact same
     `RMS_global` (with `g -> m/s^2` conversion where applicable and linear `detrend`)
     to both sources and target. Zero target statistics are ever inspected.
  4. Extracts the 141 TabVibNet fusion features (`extractors_v2` + `SignalAIWrapper`).
  5. Fits `StandardScaler` strictly on source training features (`pretrain_train`).
  6. Pretrains `TabNetPretrainer` exclusively on unlabeled source training features
     (`pretrain_train`, zero validation recordings, zero target data, zero labels).
  7. Trains the 5-class supervised source head (`S4`) with hierarchical source balancing
     on `source_train` and early stopping on `source_val`, keeping the TabNet encoder
     100% frozen (`encoder_param_max_abs_diff == 0.0`, `encoder_buffer_max_abs_diff == 0.0`).
  8. Evaluates on 100% of target `CWRU_12k` windows without head expansion or target
     adaptation, counting out-of-target/unknown class predictions strictly as errors.
  9. Saves all results, metrics, configs, checkpoints, and manifests in a new
     non-overwriting run directory under `results/da_dg_tl_matrix/`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR
from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features
from src.vibnet.data.registry import get_sampling_rate_hz, load_yaml_config
from src.vibnet.evaluation.da_dg_tl_protocol import (
    DATASET_PHYSICAL_UNITS,
    SOURCE_CONFIGS,
    WindowRecord,
    allocate_non_overwriting_run_dir,
    compute_hierarchical_source_balance,
    create_source_recording_split,
    evaluate_variable_space_predictions,
    fit_dg_rms_normalizer,
    fit_self_supervised_tabnet_pretrainer,
    fit_source_only_standard_scaler,
    load_dataset_window_records,
    predict_with_variable_class_head,
    prepare_self_supervised_pretraining_inputs,
    train_variable_class_head_with_frozen_tabnet_encoder,
)
from src.vibnet.evaluation.provenance import collect_run_provenance, write_run_metadata

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "da_dg_tl_matrix.yaml"


def resolve_unnormalized_dataset_root(dataset_name: str) -> Path:
    """
    Return the parent directory containing unnormalized physical waveforms for
    `dataset_name`:
      - `MFPT` uses `data/processed_tl/detrended` (the 78 harmonized 1-second 48.828 kHz windows).
      - `UORED`, `HUST`, `CWRU_12k`, `CWRU_48k` use `data/processed`.
    """
    if dataset_name == "MFPT":
        return PROJECT_ROOT / "data" / "processed_tl" / "detrended"
    return PROJECT_ROOT / "data" / "processed"


def run_single_dg_pilot(
    source_config_id: str = "S4",
    target_dataset: str = "CWRU_12k",
    seed: int = 42,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_root: Path = RESULTS_DIR / "da_dg_tl_matrix",
) -> Path:
    t0 = time.time()
    cfg = load_yaml_config(config_path)
    hp = dict(cfg.get("models", {}).get("tabnet_hyperparameters", {}))

    if source_config_id not in SOURCE_CONFIGS:
        raise ValueError(f"Unsupported source_config_id: {source_config_id}")
    active_sources = SOURCE_CONFIGS[source_config_id]

    target_unit = DATASET_PHYSICAL_UNITS.get(target_dataset, "unknown")
    if target_unit.lower() in ("v", "volt", "volts", "voltage"):
        raise ValueError(
            f"Target '{target_dataset}' has physical unit '{target_unit}' and is blocked "
            "for 141-feature DG pilot."
        )

    print("=" * 78)
    print(f"VIBNET-1D DG PILOT | {source_config_id} {list(active_sources)} -> {target_dataset} | seed={seed} | 141 features")
    print("=" * 78)

    # 1. Load source records and waveforms (native physical units, pre-RMS)
    source_records: List[WindowRecord] = []
    source_waveforms: List[np.ndarray] = []
    source_roots_used: Dict[str, str] = {}
    for s_name in active_sources:
        s_root = resolve_unnormalized_dataset_root(s_name)
        source_roots_used[s_name] = str(s_root / s_name)
        recs, waves = load_dataset_window_records(
            dataset_name=s_name,
            data_root=s_root,
            load_waveforms=True,
            deduplicate_identical_recordings_by_sha256=True,
        )
        assert waves is not None
        source_records.extend(recs)
        source_waveforms.extend(waves)
        print(f"[DATA] Loaded source {s_name}: {len(recs)} windows from {s_root / s_name}")

    # 2. Load target records and waveforms (native physical units, pre-RMS)
    t_root = resolve_unnormalized_dataset_root(target_dataset)
    target_records, target_waveforms = load_dataset_window_records(
        dataset_name=target_dataset,
        data_root=t_root,
        load_waveforms=True,
        deduplicate_identical_recordings_by_sha256=True,
    )
    assert target_waveforms is not None
    print(f"[DATA] Loaded target {target_dataset}: {len(target_records)} windows from {t_root / target_dataset}")

    # 3. Split source recordings BEFORE any normalization or pretraining
    src_partitions, src_manifest = create_source_recording_split(
        records=source_records,
        seed=int(seed),
        val_fraction=0.20,
        source_config_id=source_config_id,
        include_additional_unsupervised_for_pretraining=True,
    )
    pretrain_train_set = set(src_partitions["pretrain_train"])
    print(
        f"[SPLIT] source_train={len(src_partitions['source_train'])} wins | "
        f"source_val={len(src_partitions['source_val'])} wins | "
        f"pretrain_only_train={len(src_partitions['pretrain_only_train'])} wins | "
        f"pretrain_only_val={len(src_partitions['pretrain_only_val'])} wins | "
        f"pretrain_train_total={len(src_partitions['pretrain_train'])} wins"
    )

    # 4. Fit DGRMSNormalizer STRICTLY on healthy windows of source TRAINING recordings
    source_normal_train_windows: Dict[str, List[np.ndarray]] = {s: [] for s in active_sources}
    source_normal_train_rec_keys: Dict[str, List[str]] = {s: [] for s in active_sources}
    for idx in sorted(pretrain_train_set):
        rec = source_records[idx]
        if rec.harmonized_class == "Normal" or rec.raw_class in ("Class_Normal", "Class_23", "Class_0"):
            source_normal_train_windows[rec.dataset].append(source_waveforms[idx])
            source_normal_train_rec_keys[rec.dataset].append(rec.recording_key)

    source_units = {s: DATASET_PHYSICAL_UNITS[s] for s in active_sources}
    dg_norm = fit_dg_rms_normalizer(
        source_normal_windows=source_normal_train_windows,
        source_units=source_units,
        aggregation_mode="geometric_mean_ms2",
        apply_detrend=True,
    )
    print(
        f"[NORM] DG RMS_global (geometric mean, m/s^2) = {dg_norm.global_target_rms_ref:.6f} | "
        f"source refs (m/s^2) = { {k: round(v, 4) for k, v in dg_norm.source_rms_refs_ms2.items()} }"
    )

    # 5. Normalize sources & target using the EXACT same RMS_global (with g -> m/s^2 + detrend)
    #    and extract 141 features per dataset using canonical sampling rate
    X_source_141 = np.zeros((len(source_records), 141), dtype=np.float32)
    fs_used_by_dataset: Dict[str, int] = {}

    for s_name in active_sources:
        ds_indices = [i for i, r in enumerate(source_records) if r.dataset == s_name]
        ds_raw_waves = [source_waveforms[i] for i in ds_indices]
        ds_norm_waves = dg_norm.transform_source_windows(
            source_name=s_name,
            windows=ds_raw_waves,
            feature_mode="full_141",
        )
        fs_hz = get_sampling_rate_hz(s_name)
        fs_used_by_dataset[s_name] = int(fs_hz)
        print(f"[FEAT] Extracting 141 features for source {s_name} ({len(ds_norm_waves)} wins, fs={fs_hz} Hz)...")
        X_ds = extract_fusion_features(
            X_raw=np.asarray(ds_norm_waves, dtype=np.float32),
            fs=int(fs_hz),
            vibnet_extractor_func=extract_advanced_features,
            pipeline_name="all",
        )
        if X_ds.shape != (len(ds_indices), 141):
            raise RuntimeError(
                f"Expected 141 features for {s_name}, got shape {X_ds.shape}"
            )
        X_source_141[ds_indices] = X_ds

    target_norm_waves = dg_norm.transform_target_windows(
        windows=target_waveforms,
        target_dataset=target_dataset,
        feature_mode="full_141",
    )
    t_fs_hz = get_sampling_rate_hz(target_dataset)
    fs_used_by_dataset[target_dataset] = int(t_fs_hz)
    print(f"[FEAT] Extracting 141 features for target {target_dataset} ({len(target_norm_waves)} wins, fs={t_fs_hz} Hz)...")
    X_target_141 = extract_fusion_features(
        X_raw=np.asarray(target_norm_waves, dtype=np.float32),
        fs=int(t_fs_hz),
        vibnet_extractor_func=extract_advanced_features,
        pipeline_name="all",
    )
    if X_target_141.shape != (len(target_records), 141):
        raise RuntimeError(
            f"Expected 141 features for {target_dataset}, got shape {X_target_141.shape}"
        )

    # 6. Fit StandardScaler strictly on source training data (pretrain_train: zero val, zero target)
    pretrain_idxs = list(src_partitions["pretrain_train"])
    sup_train_idxs = list(src_partitions["source_train"])
    sup_val_idxs = list(src_partitions["source_val"])

    scaler, _, X_sup_val_scaled, scaled_targets = fit_source_only_standard_scaler(
        X_source_train=X_source_141[pretrain_idxs],
        X_source_val=X_source_141[sup_val_idxs],
        X_target_dict={target_dataset: X_target_141},
    )
    assert X_sup_val_scaled is not None
    X_sup_train_scaled = scaler.transform(X_source_141[sup_train_idxs]).astype(np.float32)
    X_target_scaled = scaled_targets[target_dataset]

    # 7. Self-supervised TabNetPretrainer on unlabeled source training data (141 features)
    pretrain_inputs = prepare_self_supervised_pretraining_inputs(
        records=source_records,
        partitions=src_partitions,
        X_all_141=X_source_141,
        source_config_id=source_config_id,
        feature_mode="full_141",
        scaler=scaler,
    )
    print(
        f"[PRETRAIN] Fitting TabNetPretrainer (141 features, max_epochs={hp.get('pretrain_epochs', 30)}, "
        f"n_windows={pretrain_inputs.X_pretrain_unlabeled.shape[0]}, labels=NONE)..."
    )
    pretrainer, pretrain_audit = fit_self_supervised_tabnet_pretrainer(
        pretrain_inputs=pretrain_inputs,
        seed=int(seed),
        max_epochs=int(hp.get("pretrain_epochs", 30)),
        batch_size=int(hp.get("batch_size", 64)),
        pretraining_ratio=0.5,
        tabnet_params=hp,
    )

    # 8. Train supervised source head on source_train (with hierarchical source balancing)
    #    and early stopping on source_val, with TabNet encoder 100% frozen
    y_sup_train = [str(source_records[i].harmonized_class) for i in sup_train_idxs]
    ds_sup_train = [source_records[i].dataset for i in sup_train_idxs]
    y_sup_val = [str(source_records[i].harmonized_class) for i in sup_val_idxs]

    bal_res = compute_hierarchical_source_balance(
        datasets=ds_sup_train,
        class_ids=y_sup_train,
        mode="hierarchical_weights",
        seed=int(seed),
        partition_name="source_train",
    )
    print(
        f"[HEAD] Training supervised source head ({source_config_id}) with frozen encoder "
        f"(max_epochs={hp.get('head_max_epochs', 40)}, patience={hp.get('head_patience', 10)})..."
    )
    head_ckpt = train_variable_class_head_with_frozen_tabnet_encoder(
        pretrained_tabnet=pretrainer,
        X_train_scaled=X_sup_train_scaled,
        y_train=y_sup_train,
        X_val_scaled=X_sup_val_scaled,
        y_val=y_sup_val,
        seed=int(seed),
        sample_weights=bal_res.sample_weights,
        source_config_id=source_config_id,
        max_epochs=int(hp.get("head_max_epochs", 40)),
        patience=int(hp.get("head_patience", 10)),
        batch_size=int(hp.get("batch_size", 64)),
        lr=float(hp.get("lr_head_source", 0.01)),
    )
    print(
        f"[HEAD] Selected epoch={head_ckpt.selected_epoch} | best_val_loss={head_ckpt.best_val_loss:.4f} | "
        f"encoder_param_diff={head_ckpt.encoder_param_max_abs_diff} | "
        f"encoder_buf_diff={head_ckpt.encoder_buffer_max_abs_diff} | "
        f"head_param_diff={head_ckpt.head_param_max_abs_diff:.6f}"
    )

    # 9. Evaluate on source_val (diagnostic) and on 100% of target CWRU_12k (DG test)
    val_pred_ids, val_probs, _ = predict_with_variable_class_head(
        pretrained_tabnet=pretrainer,
        head_checkpoint=head_ckpt,
        X_test_scaled=X_sup_val_scaled,
    )
    source_val_metrics = evaluate_variable_space_predictions(
        y_true=y_sup_val,
        y_pred=val_pred_ids,
        head_class_names=head_ckpt.class_names,
        y_proba=val_probs,
    )

    y_target_true = [str(r.harmonized_class) for r in target_records]
    tgt_pred_ids, tgt_probs, tgt_pred_names = predict_with_variable_class_head(
        pretrained_tabnet=pretrainer,
        head_checkpoint=head_ckpt,
        X_test_scaled=X_target_scaled,
    )
    target_dg_metrics = evaluate_variable_space_predictions(
        y_true=y_target_true,
        y_pred=tgt_pred_ids,
        head_class_names=head_ckpt.class_names,
        y_proba=tgt_probs,
    )

    elapsed_sec = time.time() - t0
    print("=" * 78)
    print(
        f"[RESULT] Source Val ({source_config_id}): Macro-F1={source_val_metrics['macro_f1']:.4f} | "
        f"BalAcc={source_val_metrics['balanced_accuracy']:.4f} | Acc={source_val_metrics['accuracy']:.4f}"
    )
    print(
        f"[RESULT] DG Target ({target_dataset}): Macro-F1={target_dg_metrics['macro_f1']:.4f} | "
        f"Macro-F1(w/phantom)={target_dg_metrics['macro_f1_including_phantom_classes']:.4f} | "
        f"BalAcc={target_dg_metrics['balanced_accuracy']:.4f} | "
        f"Acc={target_dg_metrics['accuracy']:.4f} | "
        f"ROC-AUC={target_dg_metrics['roc_auc_ovr']} | "
        f"Out-of-target preds={target_dg_metrics['out_of_target_prediction_count']}/{target_dg_metrics['n_test_samples']} "
        f"({100.0 * target_dg_metrics['out_of_target_prediction_rate']:.2f}%)"
    )
    print(f"[TIME] Elapsed: {elapsed_sec:.2f}s")
    print("=" * 78)

    # 10. Save all artifacts in a new non-overwriting run directory
    prefix = f"pilot_dg_{source_config_id}_to_{target_dataset}_seed{seed}"
    run_dir = allocate_non_overwriting_run_dir(output_root, experiment_prefix=prefix)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save checkpoints
    pretrainer.save_model(str(ckpt_dir / f"tabnet_pretrainer_{source_config_id}_seed{seed}"))
    torch.save(
        {
            "latent_dim": head_ckpt.latent_dim,
            "class_names": head_ckpt.class_names,
            "head_state_dict": head_ckpt.head_state_dict,
            "source_config_id": head_ckpt.source_config_id,
            "selected_epoch": head_ckpt.selected_epoch,
            "best_val_loss": head_ckpt.best_val_loss,
            "encoder_param_max_abs_diff": head_ckpt.encoder_param_max_abs_diff,
            "encoder_buffer_max_abs_diff": head_ckpt.encoder_buffer_max_abs_diff,
            "head_param_max_abs_diff": head_ckpt.head_param_max_abs_diff,
        },
        ckpt_dir / f"source_head_{source_config_id}_seed{seed}.pt",
    )
    np.savez(
        ckpt_dir / f"scaler_{source_config_id}_seed{seed}.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
        n_samples_seen=np.asarray([scaler.n_samples_seen_], dtype=np.int64),
    )

    # Save source split manifest (JSON & CSV)
    (run_dir / "source_split_manifest.json").write_text(
        json.dumps(src_manifest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (run_dir / "source_split_manifest.csv").open("w", newline="", encoding="utf-8") as fp:
        fieldnames = [
            "dataset",
            "recording_key",
            "recording_prefix",
            "raw_class",
            "harmonized_class",
            "condition",
            "partition",
            "seed",
            "window_count",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for entry in src_manifest.recording_entries:
            writer.writerow(entry.to_dict())

    # Save per-window target predictions
    with (run_dir / "target_predictions.csv").open("w", newline="", encoding="utf-8") as fp:
        prob_cols = [f"prob_{c.replace(' ', '_')}" for c in head_ckpt.class_names]
        fieldnames = [
            "sample_id",
            "recording_key",
            "condition",
            "raw_class",
            "true_class",
            "pred_class",
            "is_correct",
        ] + prob_cols
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for idx, rec in enumerate(target_records):
            row: Dict[str, Any] = {
                "sample_id": rec.sample_id,
                "recording_key": rec.recording_key,
                "condition": rec.condition,
                "raw_class": rec.raw_class,
                "true_class": y_target_true[idx],
                "pred_class": tgt_pred_names[idx],
                "is_correct": int(y_target_true[idx] == tgt_pred_names[idx]),
            }
            for col_i, col_name in enumerate(prob_cols):
                row[col_name] = float(tgt_probs[idx, col_i])
            writer.writerow(row)

    pilot_config = {
        "protocol": "DG",
        "source_config_id": source_config_id,
        "active_sources": list(active_sources),
        "target_dataset": target_dataset,
        "seed": int(seed),
        "feature_dimension": 141,
        "feature_mode": "full_141",
        "source_roots_used": source_roots_used,
        "target_root_used": str(t_root / target_dataset),
        "sampling_rates_hz": fs_used_by_dataset,
        "physical_units": {
            **source_units,
            target_dataset: target_unit,
        },
        "tabnet_hyperparameters": hp,
        "standard_scaler_scope": "source_pretrain_train_only",
        "source_balancing_mode": "hierarchical_weights",
        "encoder_frozen": True,
        "target_statistics_used": False,
    }
    (run_dir / "pilot_config.json").write_text(
        json.dumps(pilot_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pilot_results = {
        "pilot_summary": {
            "protocol": "DG",
            "source_config_id": source_config_id,
            "target_dataset": target_dataset,
            "seed": int(seed),
            "feature_dimension": 141,
            "elapsed_seconds": round(elapsed_sec, 3),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "dg_rms_normalization": {
            "aggregation_mode": dg_norm.aggregation_mode,
            "global_rms_ref_ms2": float(dg_norm.global_target_rms_ref),
            "source_rms_refs_native": dg_norm.source_rms_refs,
            "source_rms_refs_ms2": dg_norm.source_rms_refs_ms2,
            "source_units": dg_norm.source_units,
            "target_unit": target_unit,
            "source_normal_train_recording_counts": {
                s: len(set(keys)) for s, keys in source_normal_train_rec_keys.items()
            },
            "source_normal_train_window_counts": {
                s: len(wins) for s, wins in source_normal_train_windows.items()
            },
            "same_global_rms_ref_applied_to_sources_and_target": True,
            "target_statistics_used": False,
        },
        "self_supervised_pretraining_audit": pretrain_audit,
        "hierarchical_source_balancing_audit": {
            "mode": bal_res.mode,
            "distribution_before": bal_res.distribution_before,
            "effective_weight_by_dataset": bal_res.effective_weight_by_dataset,
            "effective_weight_by_dataset_class": bal_res.effective_weight_by_dataset_class,
        },
        "source_head_training_audit": {
            "head_class_names": list(head_ckpt.class_names),
            "head_output_dim": int(head_ckpt.num_classes),
            "selected_epoch": int(head_ckpt.selected_epoch),
            "best_val_loss": float(head_ckpt.best_val_loss),
            "encoder_param_max_abs_diff": float(head_ckpt.encoder_param_max_abs_diff),
            "encoder_buffer_max_abs_diff": float(head_ckpt.encoder_buffer_max_abs_diff),
            "head_param_max_abs_diff": float(head_ckpt.head_param_max_abs_diff),
        },
        "source_val_metrics": source_val_metrics,
        "target_dg_metrics": target_dg_metrics,
    }
    (run_dir / "pilot_results.json").write_text(
        json.dumps(pilot_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    provenance = collect_run_provenance(
        project_root=PROJECT_ROOT,
        script_paths=[
            __file__,
            "src/vibnet/evaluation/da_dg_tl_protocol.py",
            "configs/da_dg_tl_matrix.yaml",
        ],
        seeds=[int(seed)],
        extra_config=pilot_config,
    )
    write_run_metadata(run_dir, provenance)
    print(f"[SAVED] All pilot artifacts saved to: {run_dir}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Executa um único piloto controlado de Domain Generalization (DG) do VibNet-1D."
    )
    parser.add_argument("--source-config", default="S4", choices=["S1", "S2", "S3", "S4"])
    parser.add_argument("--target", default="CWRU_12k", choices=["CWRU_12k", "CWRU_48k"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=RESULTS_DIR / "da_dg_tl_matrix")
    args = parser.parse_args(argv)

    run_single_dg_pilot(
        source_config_id=args.source_config,
        target_dataset=args.target,
        seed=int(args.seed),
        config_path=args.config,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

