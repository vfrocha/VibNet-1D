"""
Real Execution Engine for the VibNet-1D DA, DG, and TL Experimental Matrix
(`src.vibnet.evaluation.da_dg_tl_runner`).

Reuses all methodological primitives from `da_dg_tl_protocol`, `protocols`, `metrics`,
and `balancing` to execute:
  - **DG**: `S1..S4` on physically compatible targets (`CWRU_12k`, `CWRU_48k`), blocking
    `PU` under 141 features (`V` vs `m/s^2`). Uses unified geometric `RMS_global` in
    `m/s^2` fitted strictly on source training normal recordings, frozen TabNet encoder,
    variable-space source head (no expansion), and `RandomForest`/`SVM` (`Source-Only`).
  - **DA**: `S1..S4` on `CWRU_12k`, `CWRU_48k`, and `PU` with 20% adaptation / 80% test
    by recording (respecting `CWRU_12k` Normal 2/6 exception, `CWRU_48k` external
    calibration from `CWRU_12k` adaptation normal recordings, and `PU` SHA-256 duplicate
    grouping). Adapts only the canonically expanded TabNet head with frozen encoder, plus
    `RandomForest`/`SVM` (`Target-Only 20%` and `Source + Target 20%`).
  - **TL**: `S1..S4` on `CWRU_12k`, `CWRU_48k`, `PU`, `HUST_Gearbox`, and `UOEMD` under
    strict Leave-One-Condition-Out (LOCO) with native target class vocabularies, fold-local
    training-only RMS normalization, TabNet (`Scratch`, `Fine-Tuning`, `Linear Probing`),
    and `RandomForest`/`SVM` (`Target LOCO Train`).
  - Safe atomic resume (`COMPLETED.json` sentinels) without ever overwriting existing runs.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features
from src.vibnet.data.registry import get_dataset_spec, get_sampling_rate_hz
from src.vibnet.evaluation.balancing import trim_single_remainder_sample
from src.vibnet.evaluation.da_dg_tl_protocol import (
    DATASET_PHYSICAL_UNITS,
    SOURCE_CONFIGS,
    WindowRecord,
    assert_zero_recording_overlap,
    build_recording_key,
    compute_hierarchical_source_balance,
    create_da_recording_split,
    create_source_recording_split,
    evaluate_variable_space_predictions,
    fit_da_rms_normalizer,
    fit_dg_rms_normalizer,
    fit_self_supervised_tabnet_pretrainer,
    fit_source_only_standard_scaler,
    load_dataset_window_records,
    parse_window_filename,
    predict_with_variable_class_head,
    prepare_self_supervised_pretraining_inputs,
    select_feature_subset_for_ablation,
    train_eval_classical_da_dg_baseline,
    train_variable_class_head_with_frozen_tabnet_encoder,
)
from src.vibnet.evaluation.metrics import compute_fold_metrics
from src.vibnet.evaluation.protocols import LOCOFoldSplit, iter_loco_folds
from src.vibnet.preprocessing.signal_norm import (
    EPSILON_RMS,
    compute_reference_rms,
    detrend_linear_window,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_unnormalized_dataset_root(dataset_name: str, project_root: Path = PROJECT_ROOT) -> Path:
    """
    Return the parent directory containing unnormalized physical waveforms:
      - `MFPT` uses `data/processed_tl/detrended` (78 harmonized 1-second 48.828 kHz windows).
      - All other datasets use `data/processed`.
    """
    if dataset_name == "MFPT":
        return project_root / "data" / "processed_tl" / "detrended"
    return project_root / "data" / "processed"


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically via temp file + `os.replace` so resume sentinels are never partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def is_job_completed(job_dir: Path) -> bool:
    """Return True iff the job's atomic `COMPLETED.json` sentinel exists and is valid JSON."""
    sentinel = job_dir / "COMPLETED.json"
    if not sentinel.is_file():
        return False
    try:
        data = json.loads(sentinel.read_text(encoding="utf-8"))
        return bool(data.get("status") in ("COMPLETED", "BLOCKED_INCOMPATIBLE_UNIT"))
    except Exception:
        return False


def _extract_141_features_checked(
    waveforms: Sequence[np.ndarray],
    dataset_name: str,
) -> np.ndarray:
    fs_hz = int(get_sampling_rate_hz(dataset_name))
    X = extract_fusion_features(
        X_raw=np.asarray(waveforms, dtype=np.float32),
        fs=fs_hz,
        vibnet_extractor_func=extract_advanced_features,
        pipeline_name="all",
    )
    if X.ndim != 2 or X.shape != (len(waveforms), 141):
        raise RuntimeError(
            f"Expected (N={len(waveforms)}, 141) features for '{dataset_name}', got {X.shape}"
        )
    return X


@dataclass(frozen=True)
class NativeTLRecord:
    """Window record preserving the native class vocabulary for Transfer Learning (TL) + LOCO."""

    sample_id: str
    dataset: str
    condition: str
    raw_class: str
    recording_prefix: str
    recording_key: str
    window_index: int


def load_native_tl_dataset_records(
    dataset_name: str,
    data_root: Path,
    load_waveforms: bool = True,
) -> Tuple[List[NativeTLRecord], Optional[List[np.ndarray]]]:
    """
    Scan `<data_root>/<dataset_name>/<condition>/<raw_class>/sNNNNN_wNN.npy` preserving
    each dataset's native class vocabulary (`HUST_Gearbox`, `UOEMD`, `PU`, `CWRU_12k`, `CWRU_48k`).
    """
    ds_dir = Path(data_root) / dataset_name
    if not ds_dir.is_dir():
        raise FileNotFoundError(f"TL dataset directory not found: {ds_dir}")

    records: List[NativeTLRecord] = []
    waveforms: Optional[List[np.ndarray]] = [] if load_waveforms else None

    for fpath in sorted(ds_dir.rglob("*.npy")):
        rel_parts = fpath.relative_to(ds_dir).parts
        if len(rel_parts) != 3:
            raise ValueError(f"Invalid directory nesting depth ({len(rel_parts)} != 3) for '{fpath}'.")
        cond_name, raw_cls, fname = rel_parts
        rec_prefix, w_idx = parse_window_filename(fname)
        rec_key = build_recording_key(dataset_name, cond_name, raw_cls, rec_prefix)
        sample_id = f"{dataset_name}/{cond_name}/{raw_cls}/{fname}"
        records.append(
            NativeTLRecord(
                sample_id=sample_id,
                dataset=dataset_name,
                condition=cond_name,
                raw_class=raw_cls,
                recording_prefix=rec_prefix,
                recording_key=rec_key,
                window_index=w_idx,
            )
        )
        if waveforms is not None:
            arr = np.load(fpath, allow_pickle=False)
            waveforms.append(np.asarray(arr, dtype=np.float64).reshape(-1))

    if not records:
        raise ValueError(f"No .npy files found under {ds_dir}")
    return records, waveforms


def _is_normal_raw_class(dataset_name: str, raw_class: str) -> bool:
    norm_candidates = {
        "Class_Normal",
        "Class_Healthy",
        "Class_0",
        "Class_23",
        "Class_26",
        "Class_40",
        "Class_47",
        "Class_H",
        "Class_N",
    }
    if raw_class in norm_candidates:
        return True
    try:
        spec = get_dataset_spec(dataset_name)
        if raw_class == spec.normal_class or raw_class == f"Class_{spec.normal_class}":
            return True
    except Exception:
        pass
    return False


def resolve_tl_condition_group(
    dataset_name: str,
    condition: str,
    grouping_mode: str = "native_condition",
) -> str:
    """
    Map a native operating condition directory name to its LOCO fold group label
    according to `grouping_mode`:
      - `"native_condition"`: returns `condition` unchanged.
      - `CWRU_48k` + `"by_severity"`: maps `Load_*HP_Sev_0.007` -> `"Sev_0.007"` (3 folds).
      - `HUST_Gearbox` + `"by_speed"`: maps `Cond_20_0..Cond_40_4` -> `"20Hz".."40Hz"`
        and `Cond_L*_VS_0_40_0` -> `"VS_0_40_0"` (6 folds).
      - `HUST_Gearbox` + `"by_load"`: maps `Cond_*_k` and `Cond_Lk_VS_0_40_0` -> `"Load_k"` (5 folds).
    """
    mode = str(grouping_mode or "native_condition").strip()
    if mode == "native_condition":
        return str(condition)

    cond_str = str(condition).strip()
    if dataset_name == "CWRU_48k" and mode == "by_severity":
        m = re.search(r"(Sev_\d+\.\d+)", cond_str)
        if not m:
            raise ValueError(
                f"Cannot extract severity group from CWRU_48k condition '{cond_str}'."
            )
        return m.group(1)

    if dataset_name == "HUST_Gearbox" and mode in ("by_speed", "by_load"):
        m_const = re.fullmatch(r"Cond_(20|25|30|35|40)_([0-4])", cond_str)
        if m_const:
            speed, load = m_const.groups()
            return f"{speed}Hz" if mode == "by_speed" else f"Load_{load}"
        m_var = re.fullmatch(r"Cond_L([0-4])_VS_0_40_0", cond_str)
        if m_var:
            load = m_var.group(1)
            return "VS_0_40_0" if mode == "by_speed" else f"Load_{load}"
        raise ValueError(
            f"Unexpected HUST_Gearbox condition '{cond_str}' for grouping_mode='{mode}'."
        )

    raise ValueError(
        f"Unsupported TL LOCO grouping_mode='{mode}' for dataset '{dataset_name}'."
    )


def build_tl_loco_folds(
    records: Sequence[NativeTLRecord],
    dataset_name: str,
    grouping_mode: str = "native_condition",
) -> List[LOCOFoldSplit]:
    """
    Build deterministic LOCO folds for TL using `grouping_mode` and explicitly assert
    zero leakage of raw operating conditions and recording keys between train and test.
    """
    group_labels = [
        resolve_tl_condition_group(dataset_name, r.condition, grouping_mode=grouping_mode)
        for r in records
    ]
    folds = list(iter_loco_folds(group_labels))
    for fold in folds:
        tr_raw_conds = {records[i].condition for i in fold.train_indices}
        te_raw_conds = {records[i].condition for i in fold.test_indices}
        if tr_raw_conds & te_raw_conds:
            raise ValueError(
                f"Raw condition leakage in TL fold {fold.fold_index} ({fold.test_condition}): "
                f"{sorted(tr_raw_conds & te_raw_conds)}"
            )
        tr_recs = {records[i].recording_key for i in fold.train_indices}
        te_recs = {records[i].recording_key for i in fold.test_indices}
        if tr_recs & te_recs:
            raise ValueError(
                f"Recording leakage in TL fold {fold.fold_index} ({fold.test_condition}): "
                f"{sorted(tr_recs & te_recs)[:5]}"
            )
    return folds


def _save_split_manifest_files(manifest: Any, out_dir: Path, prefix: str) -> None:
    atomic_write_json(out_dir / f"{prefix}.json", manifest.to_dict())
    csv_path = out_dir / f"{prefix}.csv"
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
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    with tmp_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for entry in manifest.recording_entries:
            writer.writerow(entry.to_dict())
    os.replace(tmp_csv, csv_path)


def _normalize_source_self_rms(
    records: Sequence[WindowRecord],
    waveforms: Sequence[np.ndarray],
    pretrain_train_indices: Sequence[int],
    active_sources: Sequence[str],
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """
    For DA and TL source pretraining: normalize each active source domain using ONLY
    its own healthy (`Normal` / `Class_23`) training recordings (`pretrain_train`),
    preserving linear `detrend` and zero validation leakage.
    """
    train_set = set(int(i) for i in pretrain_train_indices)
    rms_by_source: Dict[str, float] = {}
    for s_name in active_sources:
        norm_waves = []
        for idx in sorted(train_set):
            r = records[idx]
            if r.dataset == s_name and (
                r.harmonized_class == "Normal" or _is_normal_raw_class(s_name, r.raw_class)
            ):
                norm_waves.append(
                    np.asarray(detrend_linear_window(waveforms[idx]), dtype=np.float64)
                )
        if not norm_waves:
            raise ValueError(f"Source '{s_name}' has 0 healthy windows in training split.")
        rms_by_source[s_name] = float(compute_reference_rms(norm_waves))

    out: List[np.ndarray] = []
    for idx, r in enumerate(records):
        denom = max(rms_by_source[r.dataset], EPSILON_RMS)
        arr = np.asarray(detrend_linear_window(waveforms[idx]), dtype=np.float64)
        out.append((arr / denom).astype(np.float32))
    return out, rms_by_source


class MatrixExecutionContext:
    """
    In-memory cache for raw waveforms and extracted fold/source bundles during a single
    matrix process, preventing redundant feature extraction across seeds/targets while
    never sharing statistics across domains or partitions.
    """

    def __init__(self, run_dir: Path, config: Dict[str, Any], project_root: Path = PROJECT_ROOT):
        self.run_dir = run_dir
        self.config = config
        self.project_root = project_root
        self.hp = dict(config.get("models", {}).get("tabnet_hyperparameters", {}))
        self._raw_harmonized_cache: Dict[str, Tuple[List[WindowRecord], List[np.ndarray]]] = {}
        self._raw_native_tl_cache: Dict[str, Tuple[List[NativeTLRecord], List[np.ndarray]]] = {}
        self._tl_fold_feature_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        self._source_bundle_cache: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
        self._source_features_141_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    def get_harmonized_dataset(self, dataset_name: str) -> Tuple[List[WindowRecord], List[np.ndarray]]:
        if dataset_name not in self._raw_harmonized_cache:
            root = resolve_unnormalized_dataset_root(dataset_name, self.project_root)
            recs, waves = load_dataset_window_records(
                dataset_name=dataset_name,
                data_root=root,
                load_waveforms=True,
                deduplicate_identical_recordings_by_sha256=True,
            )
            assert waves is not None
            self._raw_harmonized_cache[dataset_name] = (recs, waves)
        return self._raw_harmonized_cache[dataset_name]

    def get_native_tl_dataset(self, dataset_name: str) -> Tuple[List[NativeTLRecord], List[np.ndarray]]:
        if dataset_name not in self._raw_native_tl_cache:
            root = resolve_unnormalized_dataset_root(dataset_name, self.project_root)
            recs, waves = load_native_tl_dataset_records(
                dataset_name=dataset_name,
                data_root=root,
                load_waveforms=True,
            )
            assert waves is not None
            self._raw_native_tl_cache[dataset_name] = (recs, waves)
        return self._raw_native_tl_cache[dataset_name]

    def get_or_build_source_bundle(
        self,
        protocol: str,
        source_config_id: str,
        seed: int,
        feature_mode: str = "full_141",
    ) -> Dict[str, Any]:
        """
        Build (or return cached) leak-free source split, RMS normalizer, features
        (`141` for `full_141` or `126` for `invariant_126`), StandardScaler, pretrained
        TabNetPretrainer, and (for DG/DA) frozen-encoder source head.
        - When `protocols.DG.dg_source_normalization_mode == "per_source_rms"` (default),
          `DG` (`full_141`), `DA`, and `TL` all share the exact same `SELF_RMS_TRAIN`
          source normalization (`RMS_normal = 1.0` per source) and pretrained encoder,
          while `DG` computes `dg_norm` strictly from source training normals to
          normalize the target domain (`zero_target_statistics_enforced=True`).
        - When `feature_mode == "invariant_126"`, trains a dedicated 126-input
          `TabNetPretrainer` and `StandardScaler` on the 126 scale-invariant features.
        """
        dg_source_norm_mode = str(
            self.config.get("protocols", {})
            .get("DG", {})
            .get("dg_source_normalization_mode", "per_source_rms")
        )
        if protocol == "DG" and dg_source_norm_mode == "global_rms" and feature_mode == "full_141":
            norm_family = "DG_GLOBAL_MS2"
        else:
            norm_family = "SELF_RMS_TRAIN"

        cache_key = (norm_family, feature_mode, source_config_id, int(seed))
        if cache_key in self._source_bundle_cache:
            bundle = self._source_bundle_cache[cache_key]
            if protocol in ("DG", "DA") and "source_head_ckpt" not in bundle:
                self._fit_source_head_into_bundle(bundle, source_config_id, int(seed))
            return bundle

        feat_cache_key = (norm_family, source_config_id, int(seed))
        if feat_cache_key in self._source_features_141_cache:
            feat_pack = self._source_features_141_cache[feat_cache_key]
            active_sources = feat_pack["active_sources"]
            source_records = feat_pack["source_records"]
            src_partitions = feat_pack["src_partitions"]
            src_manifest = feat_pack["src_manifest"]
            X_source_141 = feat_pack["X_source_141"]
            dg_norm = feat_pack["dg_norm"]
            source_self_rms = feat_pack["source_self_rms"]
        else:
            active_sources = SOURCE_CONFIGS[source_config_id]
            source_records = []
            source_waveforms: List[np.ndarray] = []
            for s_name in active_sources:
                recs, waves = self.get_harmonized_dataset(s_name)
                source_records.extend(recs)
                source_waveforms.extend(waves)

            src_partitions, src_manifest = create_source_recording_split(
                records=source_records,
                seed=int(seed),
                val_fraction=0.20,
                source_config_id=source_config_id,
                include_additional_unsupervised_for_pretraining=True,
            )
            pretrain_idxs = list(src_partitions["pretrain_train"])

            # Always fit leak-free DGRMSNormalizer from source training normals so any
            # DG job sharing SELF_RMS_TRAIN has the exact geometric RMS_global in m/s^2
            # ready for target normalization without touching target statistics.
            source_normal_train_windows: Dict[str, List[np.ndarray]] = {s: [] for s in active_sources}
            for idx in pretrain_idxs:
                rec = source_records[idx]
                if rec.harmonized_class == "Normal" or _is_normal_raw_class(rec.dataset, rec.raw_class):
                    source_normal_train_windows[rec.dataset].append(source_waveforms[idx])
            source_units = {s: DATASET_PHYSICAL_UNITS[s] for s in active_sources}
            dg_norm = fit_dg_rms_normalizer(
                source_normal_windows=source_normal_train_windows,
                source_units=source_units,
                aggregation_mode="geometric_mean_ms2",
                apply_detrend=True,
            )

            X_source_141 = np.zeros((len(source_records), 141), dtype=np.float32)
            source_self_rms: Dict[str, float] = {}

            if norm_family == "DG_GLOBAL_MS2":
                for s_name in active_sources:
                    ds_indices = [i for i, r in enumerate(source_records) if r.dataset == s_name]
                    ds_norm_waves = dg_norm.transform_source_windows(
                        source_name=s_name,
                        windows=[source_waveforms[i] for i in ds_indices],
                        feature_mode="full_141",
                        source_norm_mode="global_rms",
                    )
                    X_source_141[ds_indices] = _extract_141_features_checked(ds_norm_waves, s_name)
            else:
                norm_waves_all, source_self_rms = _normalize_source_self_rms(
                    records=source_records,
                    waveforms=source_waveforms,
                    pretrain_train_indices=pretrain_idxs,
                    active_sources=active_sources,
                )
                for s_name in active_sources:
                    ds_indices = [i for i, r in enumerate(source_records) if r.dataset == s_name]
                    X_source_141[ds_indices] = _extract_141_features_checked(
                        [norm_waves_all[i] for i in ds_indices], s_name
                    )

            self._source_features_141_cache[feat_cache_key] = {
                "active_sources": active_sources,
                "source_records": source_records,
                "src_partitions": src_partitions,
                "src_manifest": src_manifest,
                "X_source_141": X_source_141,
                "dg_norm": dg_norm,
                "source_self_rms": source_self_rms,
            }

        pretrain_idxs = list(src_partitions["pretrain_train"])
        sup_train_idxs = list(src_partitions["source_train"])
        sup_val_idxs = list(src_partitions["source_val"])

        X_source_selected = select_feature_subset_for_ablation(X_source_141, mode=feature_mode)
        input_dim = int(X_source_selected.shape[1])

        scaler, _, X_sup_val_scaled, _ = fit_source_only_standard_scaler(
            X_source_train=X_source_selected[pretrain_idxs],
            X_source_val=X_source_selected[sup_val_idxs],
        )
        assert X_sup_val_scaled is not None
        X_sup_train_scaled = scaler.transform(X_source_selected[sup_train_idxs]).astype(np.float32)

        pretrain_inputs = prepare_self_supervised_pretraining_inputs(
            records=source_records,
            partitions=src_partitions,
            X_all_141=X_source_141,
            source_config_id=source_config_id,
            feature_mode=feature_mode,
            scaler=scaler,
        )
        pretrainer, pretrain_audit = fit_self_supervised_tabnet_pretrainer(
            pretrain_inputs=pretrain_inputs,
            seed=int(seed),
            max_epochs=int(self.hp.get("pretrain_epochs", 30)),
            batch_size=int(self.hp.get("batch_size", 64)),
            pretraining_ratio=0.5,
            tabnet_params=self.hp,
        )

        # Save shared pretrainer checkpoint safely inside run_dir/shared_pretrainers/
        dir_tag = (
            f"{norm_family}_{source_config_id}_seed{seed}"
            if feature_mode == "full_141"
            else f"{norm_family}_{feature_mode}_{source_config_id}_seed{seed}"
        )
        shared_dir = self.run_dir / "shared_pretrainers" / dir_tag
        shared_dir.mkdir(parents=True, exist_ok=True)
        pt_path = shared_dir / f"tabnet_pretrainer_{source_config_id}_seed{seed}"
        if not Path(str(pt_path) + ".zip").is_file():
            pretrainer.save_model(str(pt_path))
        np.savez(
            shared_dir / f"scaler_{source_config_id}_seed{seed}.npz",
            mean=scaler.mean_,
            scale=scaler.scale_,
            var=scaler.var_,
            n_samples_seen=np.asarray([scaler.n_samples_seen_], dtype=np.int64),
        )

        bundle: Dict[str, Any] = {
            "norm_family": norm_family,
            "dg_source_norm_mode": dg_source_norm_mode,
            "feature_mode": feature_mode,
            "input_dim": input_dim,
            "source_config_id": source_config_id,
            "active_sources": active_sources,
            "seed": int(seed),
            "source_records": source_records,
            "src_partitions": src_partitions,
            "src_manifest": src_manifest,
            "dg_norm": dg_norm,
            "source_self_rms": source_self_rms,
            "X_source_141": X_source_141,
            "X_source_selected": X_source_selected,
            "scaler": scaler,
            "X_sup_train_scaled": X_sup_train_scaled,
            "X_sup_val_scaled": X_sup_val_scaled,
            "pretrainer": pretrainer,
            "pretrain_audit": pretrain_audit,
            "shared_pretrainer_zip": str(pt_path) + ".zip",
        }
        if protocol in ("DG", "DA"):
            self._fit_source_head_into_bundle(bundle, source_config_id, int(seed))
        self._source_bundle_cache[cache_key] = bundle
        return bundle

    def _fit_source_head_into_bundle(
        self,
        bundle: Dict[str, Any],
        source_config_id: str,
        seed: int,
    ) -> None:
        source_records = bundle["source_records"]
        sup_train_idxs = list(bundle["src_partitions"]["source_train"])
        sup_val_idxs = list(bundle["src_partitions"]["source_val"])
        y_sup_train = [str(source_records[i].harmonized_class) for i in sup_train_idxs]
        ds_sup_train = [source_records[i].dataset for i in sup_train_idxs]
        y_sup_val = [str(source_records[i].harmonized_class) for i in sup_val_idxs]

        balance_mode = str(
            self.config.get("source_balancing", {}).get(
                "default_mode", "hierarchical_class_equalized"
            )
        )
        bal_res = compute_hierarchical_source_balance(
            datasets=ds_sup_train,
            class_ids=y_sup_train,
            mode=balance_mode,
            seed=int(seed),
            partition_name="source_train",
        )
        head_ckpt = train_variable_class_head_with_frozen_tabnet_encoder(
            pretrained_tabnet=bundle["pretrainer"],
            X_train_scaled=bundle["X_sup_train_scaled"],
            y_train=y_sup_train,
            X_val_scaled=bundle["X_sup_val_scaled"],
            y_val=y_sup_val,
            seed=int(seed),
            sample_weights=bal_res.sample_weights,
            source_config_id=source_config_id,
            max_epochs=int(self.hp.get("head_max_epochs", 40)),
            patience=int(self.hp.get("head_patience", 10)),
            batch_size=int(self.hp.get("batch_size", 64)),
            lr=float(self.hp.get("lr_head_source", 0.01)),
        )
        val_pred_ids, val_probs, _ = predict_with_variable_class_head(
            pretrained_tabnet=bundle["pretrainer"],
            head_checkpoint=head_ckpt,
            X_test_scaled=bundle["X_sup_val_scaled"],
        )
        source_val_metrics = evaluate_variable_space_predictions(
            y_true=y_sup_val,
            y_pred=val_pred_ids,
            head_class_names=head_ckpt.class_names,
            y_proba=val_probs,
        )
        bundle["balance_mode"] = balance_mode
        bundle["y_sup_train"] = y_sup_train
        bundle["y_sup_val"] = y_sup_val
        bundle["bal_res"] = bal_res
        bundle["source_head_ckpt"] = head_ckpt
        bundle["source_val_metrics"] = source_val_metrics

    def get_or_extract_tl_fold_features(
        self,
        target_dataset: str,
        fold_index: int,
        fold_train_indices: Sequence[int],
        fold_test_indices: Sequence[int],
        grouping_mode: str = "native_condition",
    ) -> Dict[str, Any]:
        """
        Compute fold-local RMS_ref strictly from healthy windows in the LOCO training
        conditions (or CWRU_12k Class_0 for CWRU_48k), normalize train & test windows
        with that exact RMS_ref + detrend, and extract 141 features (cached per target,
        grouping_mode & fold).
        """
        key = (target_dataset, str(grouping_mode), int(fold_index))
        if key in self._tl_fold_feature_cache:
            return self._tl_fold_feature_cache[key]

        recs, waves = self.get_native_tl_dataset(target_dataset)
        norm_train_waves: List[np.ndarray] = []
        for idx in fold_train_indices:
            if _is_normal_raw_class(target_dataset, recs[idx].raw_class):
                norm_train_waves.append(
                    np.asarray(detrend_linear_window(waves[idx]), dtype=np.float64)
                )

        if norm_train_waves:
            rms_ref = float(compute_reference_rms(norm_train_waves))
            rms_mode = "target_loco_train_conditions_normal"
        elif target_dataset == "CWRU_48k":
            ref_recs, ref_waves = self.get_native_tl_dataset("CWRU_12k")
            ref_norm_waves = [
                np.asarray(detrend_linear_window(ref_waves[i]), dtype=np.float64)
                for i, r in enumerate(ref_recs)
                if _is_normal_raw_class("CWRU_12k", r.raw_class)
            ]
            rms_ref = float(compute_reference_rms(ref_norm_waves))
            rms_mode = "external_cwru12k_normal_reference"
        else:
            raise ValueError(
                f"TL target '{target_dataset}' fold {fold_index} has 0 healthy windows in training conditions."
            )

        denom = max(rms_ref, EPSILON_RMS)
        train_norm = [
            (np.asarray(detrend_linear_window(waves[i]), dtype=np.float64) / denom).astype(np.float32)
            for i in fold_train_indices
        ]
        test_norm = [
            (np.asarray(detrend_linear_window(waves[i]), dtype=np.float64) / denom).astype(np.float32)
            for i in fold_test_indices
        ]
        X_tr = _extract_141_features_checked(train_norm, target_dataset)
        X_te = _extract_141_features_checked(test_norm, target_dataset)

        payload = {
            "X_train_raw": X_tr,
            "X_test_raw": X_te,
            "rms_ref": rms_ref,
            "rms_mode": rms_mode,
        }
        self._tl_fold_feature_cache[key] = payload
        return payload


# =============================================================================
# 1. DG JOB EXECUTOR (TABNET + RF + SVM)
# =============================================================================


def execute_dg_job(
    ctx: MatrixExecutionContext,
    source_config_id: str,
    target_dataset: str,
    seed: int,
    job_dir: Path,
    include_classical: bool = True,
    feature_mode: str = "full_141",
) -> Dict[str, Any]:
    t0 = time.time()
    job_dir.mkdir(parents=True, exist_ok=True)
    target_unit = DATASET_PHYSICAL_UNITS.get(target_dataset, "unknown")
    feature_dim = 141 if feature_mode == "full_141" else 126

    # Enforce physical unit block for PU ONLY under 141-feature DG;
    # under invariant_126 (scale-invariant ablation), PU (Volts) is unblocked.
    if feature_mode == "full_141" and target_unit.lower() in ("v", "volt", "volts", "voltage"):
        blocked_payload = {
            "status": "BLOCKED_INCOMPATIBLE_UNIT",
            "protocol": "DG",
            "feature_mode": feature_mode,
            "source_config": source_config_id,
            "target": target_dataset,
            "seed": int(seed),
            "feature_dimension": feature_dim,
            "reason": (
                f"Target '{target_dataset}' has physical unit '{target_unit}' (Volts), "
                "which is incompatible with source RMS_global in m/s^2 without a verified "
                "sensor sensitivity calibration factor. Blocked under full_141; use "
                "invariant_126 scale-invariant ablation for DG on PU."
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(job_dir / "COMPLETED.json", blocked_payload)
        return blocked_payload

    bundle = ctx.get_or_build_source_bundle(
        "DG", source_config_id, int(seed), feature_mode=feature_mode
    )
    dg_norm = bundle["dg_norm"]
    head_ckpt = bundle["source_head_ckpt"]
    pretrainer = bundle["pretrainer"]
    scaler = bundle["scaler"]
    dg_source_norm_mode = bundle["dg_source_norm_mode"]

    tgt_recs, tgt_waves = ctx.get_harmonized_dataset(target_dataset)
    tgt_norm_waves = dg_norm.transform_target_windows(
        windows=tgt_waves,
        target_dataset=target_dataset,
        feature_mode=feature_mode,
    )
    X_tgt_141 = _extract_141_features_checked(tgt_norm_waves, target_dataset)
    X_tgt_selected = select_feature_subset_for_ablation(
        X_tgt_141,
        mode=feature_mode,
        expected_checkpoint_input_dim=feature_dim,
    )
    X_tgt_scaled = scaler.transform(X_tgt_selected).astype(np.float32)
    y_tgt_true = [str(r.harmonized_class) for r in tgt_recs]

    # TabNet DG evaluation (frozen encoder, unexpanded source head)
    pred_ids, probs, pred_names = predict_with_variable_class_head(
        pretrained_tabnet=pretrainer,
        head_checkpoint=head_ckpt,
        X_test_scaled=X_tgt_scaled,
    )
    tabnet_metrics = evaluate_variable_space_predictions(
        y_true=y_tgt_true,
        y_pred=pred_ids,
        head_class_names=head_ckpt.class_names,
        y_proba=probs,
    )

    models_results: Dict[str, Any] = {
        "TabNet_DG": {
            "metrics": tabnet_metrics,
            "head_classes": list(head_ckpt.class_names),
            "head_output_dim": int(head_ckpt.num_classes),
            "selected_epoch": int(head_ckpt.selected_epoch),
            "encoder_param_max_abs_diff": float(head_ckpt.encoder_param_max_abs_diff),
            "encoder_buffer_max_abs_diff": float(head_ckpt.encoder_buffer_max_abs_diff),
        }
    }

    # Classical baselines (RF and SVM Source-Only)
    if include_classical:
        sup_tr_idxs = list(bundle["src_partitions"]["source_train"])
        X_src_tr_raw = bundle["X_source_selected"][sup_tr_idxs]
        y_src_tr = bundle["y_sup_train"]
        w_src_tr = bundle["bal_res"].sample_weights
        for clf_name in ("RandomForest", "SVM"):
            res = train_eval_classical_da_dg_baseline(
                protocol="DG",
                variant="Source-Only",
                model_name=clf_name,
                X_test_raw=X_tgt_selected,
                y_test=y_tgt_true,
                seed=int(seed),
                X_source_train_raw=X_src_tr_raw,
                y_source_train=y_src_tr,
                source_sample_weights=w_src_tr,
            )
            models_results[f"{clf_name}_DG_SourceOnly"] = {
                "metrics": res.metrics,
                "head_classes": list(res.head_class_names),
                "hyperparameters": res.hyperparameters,
            }

    # Save checkpoints & manifests
    ckpt_dir = job_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latent_dim": head_ckpt.latent_dim,
            "class_names": head_ckpt.class_names,
            "head_state_dict": head_ckpt.head_state_dict,
            "source_config_id": source_config_id,
            "feature_mode": feature_mode,
            "feature_dimension": feature_dim,
            "selected_epoch": head_ckpt.selected_epoch,
            "encoder_param_max_abs_diff": head_ckpt.encoder_param_max_abs_diff,
            "encoder_buffer_max_abs_diff": head_ckpt.encoder_buffer_max_abs_diff,
        },
        ckpt_dir / f"source_head_{source_config_id}_{feature_mode}_seed{seed}.pt",
    )
    _save_split_manifest_files(bundle["src_manifest"], job_dir, "source_split_manifest")

    elapsed = round(time.time() - t0, 3)
    same_global_applied = bool(dg_source_norm_mode == "global_rms" and feature_mode == "full_141")
    result_payload = {
        "status": "COMPLETED",
        "protocol": "DG",
        "feature_mode": feature_mode,
        "source_config": source_config_id,
        "sources": list(SOURCE_CONFIGS[source_config_id]),
        "target": target_dataset,
        "seed": int(seed),
        "feature_dimension": feature_dim,
        "balance_mode": bundle.get("balance_mode", ""),
        "elapsed_seconds": elapsed,
        "shared_pretrainer_zip": bundle["shared_pretrainer_zip"],
        "dg_rms_normalization": {
            "dg_source_normalization_mode": dg_source_norm_mode,
            "global_rms_ref_ms2": float(dg_norm.global_target_rms_ref),
            "source_rms_refs_ms2": dg_norm.source_rms_refs_ms2,
            "source_self_rms_native": bundle.get("source_self_rms", {}),
            "target_unit": target_unit,
            "same_global_rms_ref_applied_to_sources_and_target": same_global_applied,
            "target_statistics_used": False,
        },
        "pretrain_audit": bundle["pretrain_audit"],
        "source_val_metrics": bundle["source_val_metrics"],
        "models": models_results,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(job_dir / "metrics.json", result_payload)
    atomic_write_json(
        job_dir / "COMPLETED.json",
        {
            "status": "COMPLETED",
            "protocol": "DG",
            "feature_mode": feature_mode,
            "feature_dimension": feature_dim,
            "source_config": source_config_id,
            "target": target_dataset,
            "seed": int(seed),
            "elapsed_seconds": elapsed,
            "tabnet_macro_f1": float(tabnet_metrics["macro_f1"]),
            "models_macro_f1": {
                m_name: float(m_data["metrics"]["macro_f1"])
                for m_name, m_data in models_results.items()
            },
            "timestamp_utc": result_payload["timestamp_utc"],
        },
    )
    return result_payload


# =============================================================================
# 2. DA JOB EXECUTOR (TABNET EXPANDED HEAD + RF/SVM TARGET-ONLY & SOURCE+TARGET)
# =============================================================================


def execute_da_job(
    ctx: MatrixExecutionContext,
    source_config_id: str,
    target_dataset: str,
    seed: int,
    job_dir: Path,
    include_classical: bool = True,
) -> Dict[str, Any]:
    t0 = time.time()
    job_dir.mkdir(parents=True, exist_ok=True)

    bundle = ctx.get_or_build_source_bundle("DA", source_config_id, int(seed))
    pretrainer = bundle["pretrainer"]
    source_head_ckpt = bundle["source_head_ckpt"]
    source_scaler = bundle["scaler"]

    tgt_recs, tgt_waves = ctx.get_harmonized_dataset(target_dataset)
    da_parts, da_manifest = create_da_recording_split(
        records=tgt_recs,
        seed=int(seed),
        adapt_fraction=0.20,
        adapt_val_fraction=0.0,
    )
    assert_zero_recording_overlap(tgt_recs, da_parts)

    adapt_idxs = list(da_parts["adapt"])
    test_idxs = list(da_parts["test"])

    # Leak-free DA RMS normalization (internal adaptation Normal, or CWRU_12k adapt Normal for CWRU_48k)
    if target_dataset == "CWRU_48k":
        c12_recs, c12_waves = ctx.get_harmonized_dataset("CWRU_12k")
        c12_parts, _ = create_da_recording_split(
            records=c12_recs,
            seed=int(seed),
            adapt_fraction=0.20,
            adapt_val_fraction=0.0,
        )
        c12_adapt_norm_idxs = [
            i for i in c12_parts["adapt"] if c12_recs[i].harmonized_class == "Normal"
        ]
        c12_test_keys = sorted({c12_recs[i].recording_key for i in c12_parts["test"]})
        da_norm = fit_da_rms_normalizer(
            records=tgt_recs,
            windows=tgt_waves,
            adapt_indices=adapt_idxs,
            test_indices=test_idxs,
            apply_detrend=True,
            cross_reference_normal_windows=[c12_waves[i] for i in c12_adapt_norm_idxs],
            cross_reference_recording_keys=sorted(
                {c12_recs[i].recording_key for i in c12_adapt_norm_idxs}
            ),
            cross_reference_domain="CWRU_12k",
            cross_reference_forbidden_recording_keys=c12_test_keys,
        )
    else:
        da_norm = fit_da_rms_normalizer(
            records=tgt_recs,
            windows=tgt_waves,
            adapt_indices=adapt_idxs,
            test_indices=test_idxs,
            apply_detrend=True,
        )

    tgt_norm_waves = da_norm.transform_windows(tgt_waves)
    X_tgt_141 = _extract_141_features_checked(tgt_norm_waves, target_dataset)

    X_adapt_raw = X_tgt_141[adapt_idxs]
    y_adapt = [str(tgt_recs[i].harmonized_class) for i in adapt_idxs]
    X_test_raw = X_tgt_141[test_idxs]
    y_test = [str(tgt_recs[i].harmonized_class) for i in test_idxs]

    # Transform target with the frozen encoder's source StandardScaler
    X_adapt_scaled = source_scaler.transform(X_adapt_raw).astype(np.float32)
    X_test_scaled = source_scaler.transform(X_test_raw).astype(np.float32)

    # Adapt only the canonically expanded head on 20% target adaptation partition (encoder 100% frozen)
    da_fixed_epochs = max(10, int(source_head_ckpt.selected_epoch))
    da_head_ckpt = train_variable_class_head_with_frozen_tabnet_encoder(
        pretrained_tabnet=pretrainer,
        X_train_scaled=X_adapt_scaled,
        y_train=y_adapt,
        seed=int(seed),
        init_head_checkpoint=source_head_ckpt,
        expand_for_target_classes=True,
        target_dataset=target_dataset,
        source_config_id=source_config_id,
        fixed_epochs=da_fixed_epochs,
        batch_size=int(ctx.hp.get("batch_size", 64)),
        lr=float(ctx.hp.get("lr_head_da", 0.005)),
    )

    pred_ids, probs, _ = predict_with_variable_class_head(
        pretrained_tabnet=pretrainer,
        head_checkpoint=da_head_ckpt,
        X_test_scaled=X_test_scaled,
    )
    tabnet_da_metrics = evaluate_variable_space_predictions(
        y_true=y_test,
        y_pred=pred_ids,
        head_class_names=da_head_ckpt.class_names,
        y_proba=probs,
    )

    models_results: Dict[str, Any] = {
        "TabNet_DA_ExpandedHead": {
            "metrics": tabnet_da_metrics,
            "head_classes": list(da_head_ckpt.class_names),
            "head_output_dim": int(da_head_ckpt.num_classes),
            "transferred_from_source_classes": list(da_head_ckpt.transferred_from_source_classes),
            "introduced_by_expansion_classes": list(da_head_ckpt.introduced_by_expansion_classes),
            "learned_during_adaptation_classes": list(da_head_ckpt.learned_during_adaptation_classes),
            "fixed_epochs_from_source": int(da_fixed_epochs),
            "encoder_param_max_abs_diff": float(da_head_ckpt.encoder_param_max_abs_diff),
            "encoder_buffer_max_abs_diff": float(da_head_ckpt.encoder_buffer_max_abs_diff),
            "head_param_max_abs_diff": float(da_head_ckpt.head_param_max_abs_diff),
        }
    }

    if include_classical:
        sup_tr_idxs = list(bundle["src_partitions"]["source_train"])
        X_src_tr_raw = bundle["X_source_141"][sup_tr_idxs]
        y_src_tr = bundle["y_sup_train"]
        w_src_tr = bundle["bal_res"].sample_weights
        for clf_name in ("RandomForest", "SVM"):
            res_to = train_eval_classical_da_dg_baseline(
                protocol="DA",
                variant="Target-Only 20%",
                model_name=clf_name,
                X_test_raw=X_test_raw,
                y_test=y_test,
                seed=int(seed),
                X_target_adapt_raw=X_adapt_raw,
                y_target_adapt=y_adapt,
            )
            models_results[f"{clf_name}_DA_TargetOnly20"] = {
                "metrics": res_to.metrics,
                "head_classes": list(res_to.head_class_names),
                "hyperparameters": res_to.hyperparameters,
            }
            res_st = train_eval_classical_da_dg_baseline(
                protocol="DA",
                variant="Source + Target 20%",
                model_name=clf_name,
                X_test_raw=X_test_raw,
                y_test=y_test,
                seed=int(seed),
                X_source_train_raw=X_src_tr_raw,
                y_source_train=y_src_tr,
                source_sample_weights=w_src_tr,
                X_target_adapt_raw=X_adapt_raw,
                y_target_adapt=y_adapt,
            )
            models_results[f"{clf_name}_DA_SourcePlusTarget20"] = {
                "metrics": res_st.metrics,
                "head_classes": list(res_st.head_class_names),
                "hyperparameters": res_st.hyperparameters,
                "joint_learning_note": res_st.joint_learning_note,
            }

    # Save checkpoints & manifests
    ckpt_dir = job_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latent_dim": da_head_ckpt.latent_dim,
            "class_names": da_head_ckpt.class_names,
            "head_state_dict": da_head_ckpt.head_state_dict,
            "transferred_from_source_classes": da_head_ckpt.transferred_from_source_classes,
            "introduced_by_expansion_classes": da_head_ckpt.introduced_by_expansion_classes,
            "learned_during_adaptation_classes": da_head_ckpt.learned_during_adaptation_classes,
            "encoder_param_max_abs_diff": da_head_ckpt.encoder_param_max_abs_diff,
            "encoder_buffer_max_abs_diff": da_head_ckpt.encoder_buffer_max_abs_diff,
        },
        ckpt_dir / f"da_expanded_head_{source_config_id}_to_{target_dataset}_seed{seed}.pt",
    )
    _save_split_manifest_files(bundle["src_manifest"], job_dir, "source_split_manifest")
    _save_split_manifest_files(da_manifest, job_dir, "target_da_split_manifest")

    elapsed = round(time.time() - t0, 3)
    result_payload = {
        "status": "COMPLETED",
        "protocol": "DA",
        "source_config": source_config_id,
        "sources": list(SOURCE_CONFIGS[source_config_id]),
        "target": target_dataset,
        "seed": int(seed),
        "feature_dimension": 141,
        "elapsed_seconds": elapsed,
        "shared_pretrainer_zip": bundle["shared_pretrainer_zip"],
        "da_rms_normalization": {
            "rms_ref": float(da_norm.rms_ref),
            "calibration_mode": da_norm.calibration_mode,
            "reference_domain": da_norm.reference_domain,
            "n_reference_windows": int(da_norm.n_reference_windows),
            "reference_recording_keys": list(da_norm.reference_recording_keys),
            "test_data_used_in_normalization": False,
        },
        "target_da_split_summary": {
            "split_signature": da_manifest.split_signature,
            "recording_counts_by_class": da_manifest.recording_counts_by_class,
            "window_counts_by_class": da_manifest.window_counts_by_class,
        },
        "models": models_results,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(job_dir / "metrics.json", result_payload)
    atomic_write_json(
        job_dir / "COMPLETED.json",
        {
            "status": "COMPLETED",
            "protocol": "DA",
            "source_config": source_config_id,
            "target": target_dataset,
            "seed": int(seed),
            "elapsed_seconds": elapsed,
            "tabnet_macro_f1": float(tabnet_da_metrics["macro_f1"]),
            "models_macro_f1": {
                m_name: float(m_data["metrics"]["macro_f1"])
                for m_name, m_data in models_results.items()
            },
            "timestamp_utc": result_payload["timestamp_utc"],
        },
    )
    return result_payload


# =============================================================================
# 3. TL JOB EXECUTOR (LOCO: SCRATCH, FINE-TUNING, LINEAR PROBING, RF, SVM)
# =============================================================================


def _fit_eval_tabnet_tl_classifier(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    expected_classes: List[int],
    seed: int,
    hp: Mapping[str, Any],
    pretrainer: Optional[Any] = None,
) -> Dict[str, Any]:
    from pytorch_tabnet.tab_model import TabNetClassifier

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    bs = min(int(hp.get("batch_size", 64)), max(2, len(X_tr)))
    if len(X_tr) > bs and (len(X_tr) % bs == 1):
        X_trim, y_trim, _ = trim_single_remainder_sample(
            X_tr.tolist(), y_tr.tolist(), batch_size=bs, seed=seed
        )
        X_fit = np.asarray(X_trim, dtype=np.float32)
        y_fit = np.asarray(y_trim, dtype=np.int64)
    else:
        X_fit = np.asarray(X_tr, dtype=np.float32)
        y_fit = np.asarray(y_tr, dtype=np.int64)

    clf = TabNetClassifier(
        n_d=int(hp.get("n_d", 16)),
        n_a=int(hp.get("n_a", 16)),
        n_steps=int(hp.get("n_steps", 4)),
        gamma=float(hp.get("gamma", 1.3)),
        n_independent=int(hp.get("n_independent", 2)),
        n_shared=int(hp.get("n_shared", 2)),
        mask_type=str(hp.get("mask_type", "sparsemax")),
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=float(hp.get("lr_tl_finetune", 0.01))),
        seed=int(seed),
        verbose=0,
    )
    eff_vbs = min(bs, max(2, len(X_fit) // 2))
    fit_kwargs: Dict[str, Any] = {
        "X_train": X_fit,
        "y_train": y_fit,
        "max_epochs": int(hp.get("head_max_epochs", 40)),
        "batch_size": bs,
        "virtual_batch_size": eff_vbs,
        "drop_last": False,
        "weights": 1,
    }
    if pretrainer is not None:
        fit_kwargs["from_unsupervised"] = pretrainer

    clf.fit(**fit_kwargs)
    preds = np.asarray(clf.predict(X_te), dtype=np.int64)
    probs = np.asarray(clf.predict_proba(X_te), dtype=np.float64)
    fm = compute_fold_metrics(
        y_true=y_te.tolist(),
        y_pred=preds.tolist(),
        y_proba=probs.tolist(),
        expected_classes=expected_classes,
    )
    return {
        "accuracy": fm.accuracy,
        "macro_f1": fm.macro_f1,
        "weighted_f1": fm.weighted_f1,
        "precision_macro": fm.precision_macro,
        "recall_macro": fm.recall_macro,
        "roc_auc_ovr": fm.roc_auc_ovr,
        "roc_auc_valid": fm.roc_auc_valid,
    }


def _fit_eval_tabnet_tl_linear_probing(
    pretrainer: Any,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    num_classes: int,
    expected_classes: List[int],
    seed: int,
    hp: Mapping[str, Any],
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = copy.deepcopy(pretrainer.network).to(device)
    embedder = backbone.embedder
    encoder = backbone.encoder
    for p in embedder.parameters():
        p.requires_grad = False
    for p in encoder.parameters():
        p.requires_grad = False
    embedder.eval()
    encoder.eval()

    enc_before = {k: v.detach().cpu().numpy().copy() for k, v in encoder.state_dict().items()}

    with torch.no_grad():
        x_tr_t = torch.tensor(np.asarray(X_tr, dtype=np.float32), device=device)
        steps_tr, _ = encoder(embedder(x_tr_t))
        z_tr = torch.sum(torch.stack(steps_tr, dim=0), dim=0).detach()

        x_te_t = torch.tensor(np.asarray(X_te, dtype=np.float32), device=device)
        steps_te, _ = encoder(embedder(x_te_t))
        z_te = torch.sum(torch.stack(steps_te, dim=0), dim=0).detach()

    latent_dim = int(z_tr.shape[1])
    head = nn.Linear(latent_dim, num_classes).to(device)

    # Balanced class weights strictly from y_tr (never y_te)
    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float64)
    cls_w = np.where(counts > 0, len(y_tr) / (num_classes * np.maximum(counts, 1.0)), 0.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cls_w, dtype=torch.float32, device=device)
    )
    optimizer = optim.AdamW(head.parameters(), lr=float(hp.get("lr_head_source", 0.01)), weight_decay=1e-4)

    ds = TensorDataset(z_tr.cpu(), torch.tensor(y_tr, dtype=torch.long))
    gen = torch.Generator().manual_seed(int(seed))
    bs = min(int(hp.get("batch_size", 64)), max(2, len(ds)))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, generator=gen)

    max_ep = int(hp.get("head_max_epochs", 40))
    head.train()
    for _ in range(max_ep):
        for bz, by in loader:
            bz, by = bz.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(head(bz), by)
            loss.backward()
            optimizer.step()

    enc_after = {k: v.detach().cpu().numpy().copy() for k, v in encoder.state_dict().items()}
    enc_max_diff = max(
        float(np.max(np.abs(enc_after[k] - enc_before[k]))) for k in enc_before
    )
    if enc_max_diff > 0.0:
        raise AssertionError(f"Linear Probing encoder freeze violated: diff={enc_max_diff}")

    head.eval()
    with torch.no_grad():
        logits = head(z_te)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
        preds = np.argmax(probs, axis=1).astype(np.int64)

    fm = compute_fold_metrics(
        y_true=y_te.tolist(),
        y_pred=preds.tolist(),
        y_proba=probs.tolist(),
        expected_classes=expected_classes,
    )
    return {
        "accuracy": fm.accuracy,
        "macro_f1": fm.macro_f1,
        "weighted_f1": fm.weighted_f1,
        "precision_macro": fm.precision_macro,
        "recall_macro": fm.recall_macro,
        "roc_auc_ovr": fm.roc_auc_ovr,
        "roc_auc_valid": fm.roc_auc_valid,
        "encoder_max_abs_diff": enc_max_diff,
    }


def _aggregate_loco_fold_metrics(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ("accuracy", "macro_f1", "weighted_f1", "precision_macro", "recall_macro")
    agg: Dict[str, Any] = {"n_folds": len(fold_results)}
    for k in keys:
        vals = [float(f[k]) for f in fold_results]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals, ddof=0))
    auc_vals = [float(f["roc_auc_ovr"]) for f in fold_results if f.get("roc_auc_valid") and f.get("roc_auc_ovr") is not None]
    agg["roc_auc_ovr_mean"] = float(np.mean(auc_vals)) if auc_vals else None
    agg["roc_auc_ovr_std"] = float(np.std(auc_vals, ddof=0)) if auc_vals else None
    agg["roc_auc_valid_folds"] = len(auc_vals)
    return agg


def execute_tl_job(
    ctx: MatrixExecutionContext,
    source_config_id: str,
    target_dataset: str,
    seed: int,
    job_dir: Path,
    include_classical: bool = True,
    max_loco_folds: Optional[int] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    job_dir.mkdir(parents=True, exist_ok=True)

    bundle = ctx.get_or_build_source_bundle("TL", source_config_id, int(seed))
    pretrainer = bundle["pretrainer"]

    tgt_recs, _ = ctx.get_native_tl_dataset(target_dataset)
    native_classes = sorted({r.raw_class for r in tgt_recs})
    cls_to_id = {c: idx for idx, c in enumerate(native_classes)}
    expected_ids = list(range(len(native_classes)))
    y_all = np.asarray([cls_to_id[r.raw_class] for r in tgt_recs], dtype=np.int64)

    grouping_mode = str(
        ctx.config.get("protocols", {})
        .get("TL", {})
        .get("target_loco_grouping", {})
        .get(target_dataset, "native_condition")
    )
    all_folds = build_tl_loco_folds(tgt_recs, target_dataset, grouping_mode=grouping_mode)
    selected_folds = all_folds[: int(max_loco_folds)] if max_loco_folds is not None else all_folds

    per_model_folds: Dict[str, List[Dict[str, Any]]] = {
        "TabNet_TL_Scratch": [],
        "TabNet_TL_FineTuning": [],
        "TabNet_TL_LinearProbing": [],
    }
    if include_classical:
        per_model_folds["RandomForest_TL_LOCO"] = []
        per_model_folds["SVM_TL_LOCO"] = []

    fold_audit_rows: List[Dict[str, Any]] = []

    for fold in selected_folds:
        tr_idx = list(fold.train_indices)
        te_idx = list(fold.test_indices)
        y_tr = y_all[tr_idx]
        y_te = y_all[te_idx]

        fold_feats = ctx.get_or_extract_tl_fold_features(
            target_dataset=target_dataset,
            fold_index=fold.fold_index,
            fold_train_indices=tr_idx,
            fold_test_indices=te_idx,
            grouping_mode=grouping_mode,
        )
        # Fit StandardScaler strictly on LOCO training conditions
        fold_scaler = StandardScaler()
        X_tr_scaled = fold_scaler.fit_transform(fold_feats["X_train_raw"]).astype(np.float32)
        X_te_scaled = fold_scaler.transform(fold_feats["X_test_raw"]).astype(np.float32)

        fold_audit_rows.append(
            {
                "fold_index": fold.fold_index,
                "loco_grouping_mode": grouping_mode,
                "test_condition": fold.test_condition,
                "n_train_conditions": len(fold.train_conditions),
                "n_train_windows": len(tr_idx),
                "n_test_windows": len(te_idx),
                "rms_ref": fold_feats["rms_ref"],
                "rms_mode": fold_feats["rms_mode"],
            }
        )

        # 1. Scratch
        m_scratch = _fit_eval_tabnet_tl_classifier(
            X_tr_scaled, y_tr, X_te_scaled, y_te, expected_ids, int(seed), ctx.hp, pretrainer=None
        )
        m_scratch["test_condition"] = fold.test_condition
        per_model_folds["TabNet_TL_Scratch"].append(m_scratch)

        # 2. Fine-Tuning (initialized from source pretrainer)
        m_ft = _fit_eval_tabnet_tl_classifier(
            X_tr_scaled, y_tr, X_te_scaled, y_te, expected_ids, int(seed), ctx.hp, pretrainer=pretrainer
        )
        m_ft["test_condition"] = fold.test_condition
        per_model_folds["TabNet_TL_FineTuning"].append(m_ft)

        # 3. Linear Probing (frozen encoder)
        m_lp = _fit_eval_tabnet_tl_linear_probing(
            pretrainer, X_tr_scaled, y_tr, X_te_scaled, y_te, len(native_classes), expected_ids, int(seed), ctx.hp
        )
        m_lp["test_condition"] = fold.test_condition
        per_model_folds["TabNet_TL_LinearProbing"].append(m_lp)

        # 4. Classical baselines (RF and SVM on LOCO train)
        if include_classical:
            rf = RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=int(seed),
                n_jobs=-1,
            )
            rf.fit(X_tr_scaled, y_tr)
            rf_preds = rf.predict(X_te_scaled)
            rf_probs = rf.predict_proba(X_te_scaled)
            rf_fm = compute_fold_metrics(y_te.tolist(), rf_preds.tolist(), rf_probs.tolist(), expected_ids)
            per_model_folds["RandomForest_TL_LOCO"].append(
                {
                    "test_condition": fold.test_condition,
                    "accuracy": rf_fm.accuracy,
                    "macro_f1": rf_fm.macro_f1,
                    "weighted_f1": rf_fm.weighted_f1,
                    "precision_macro": rf_fm.precision_macro,
                    "recall_macro": rf_fm.recall_macro,
                    "roc_auc_ovr": rf_fm.roc_auc_ovr,
                    "roc_auc_valid": rf_fm.roc_auc_valid,
                }
            )

            svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, class_weight="balanced", random_state=int(seed))
            svm.fit(X_tr_scaled, y_tr)
            svm_preds = svm.predict(X_te_scaled)
            svm_probs = svm.predict_proba(X_te_scaled)
            svm_fm = compute_fold_metrics(y_te.tolist(), svm_preds.tolist(), svm_probs.tolist(), expected_ids)
            per_model_folds["SVM_TL_LOCO"].append(
                {
                    "test_condition": fold.test_condition,
                    "accuracy": svm_fm.accuracy,
                    "macro_f1": svm_fm.macro_f1,
                    "weighted_f1": svm_fm.weighted_f1,
                    "precision_macro": svm_fm.precision_macro,
                    "recall_macro": svm_fm.recall_macro,
                    "roc_auc_ovr": svm_fm.roc_auc_ovr,
                    "roc_auc_valid": svm_fm.roc_auc_valid,
                }
            )

    models_summary = {
        m_name: {
            "summary": _aggregate_loco_fold_metrics(folds_list),
            "per_fold": folds_list,
        }
        for m_name, folds_list in per_model_folds.items()
    }
    _save_split_manifest_files(bundle["src_manifest"], job_dir, "source_split_manifest")
    atomic_write_json(job_dir / "loco_folds_manifest.json", fold_audit_rows)

    elapsed = round(time.time() - t0, 3)
    result_payload = {
        "status": "COMPLETED",
        "protocol": "TL",
        "source_config": source_config_id,
        "sources": list(SOURCE_CONFIGS[source_config_id]),
        "target": target_dataset,
        "target_native_classes": native_classes,
        "loco_grouping_mode": grouping_mode,
        "seed": int(seed),
        "feature_dimension": 141,
        "n_loco_folds_executed": len(selected_folds),
        "n_loco_folds_total_available": len(all_folds),
        "elapsed_seconds": elapsed,
        "shared_pretrainer_zip": bundle["shared_pretrainer_zip"],
        "models": models_summary,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(job_dir / "metrics.json", result_payload)
    atomic_write_json(
        job_dir / "COMPLETED.json",
        {
            "status": "COMPLETED",
            "protocol": "TL",
            "source_config": source_config_id,
            "target": target_dataset,
            "loco_grouping_mode": grouping_mode,
            "seed": int(seed),
            "n_loco_folds_executed": len(selected_folds),
            "elapsed_seconds": elapsed,
            "models_macro_f1": {
                m_name: float(m_data["summary"]["macro_f1_mean"])
                for m_name, m_data in models_summary.items()
            },
            "timestamp_utc": result_payload["timestamp_utc"],
        },
    )
    return result_payload

