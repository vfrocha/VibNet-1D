"""
Harmonized 5-Class Vocabulary, Recording-Level Partitioning, Leak-Free RMS/Feature
Scaling, Hierarchical Source Balancing, and TabNet/Classical DA, DG & TL Protocol
Engine (`src.vibnet.evaluation.da_dg_tl_protocol`).

Implements the methodological guarantees for the VibNet-1D Domain Adaptation (DA),
Domain Generalization (DG), and Transfer Learning (TL) experimental matrix:
  1. Source configurations:
     - `S1`: `["UORED"]`
     - `S2`: `["HUST"]` (HUST Bearing, distinct from `HUST_Gearbox`)
     - `S3`: `["MFPT"]`
     - `S4`: `["UORED", "HUST", "MFPT"]`
  2. Fixed 5-class harmonized bearing ontology (`HARMONIZED_CLASSES`):
     - `0`: `"Normal"`
     - `1`: `"Ball"`
     - `2`: `"Inner"`
     - `3`: `"Outer"`
     - `4`: `"Inner & Outer"` (kept as an independent compound class)
     - Classification head output space is strictly fixed to `5` classes (`0..4`),
       with structural logit masking (`-1e9` bias on unobserved source classes)
       when a single-source configuration (`S1` or `S3`) lacks training examples
       for a subset of classes.
     - Predictions of classes absent in the target domain are counted as errors
       without removing any class from the 5-output head.
  3. Recording-level grouping (`recording_key = f"{dataset}::{condition}::{class_dir}::{sNNNNN}"`):
     - Zero recording overlap across `train`, `val`, and `test` partitions.
     - Deterministic, reproducible partition manifests indexed by `seed`.
     - For DA: 20% of recordings per class reserved for adaptation (split 80/20
       by recording into `adapt_train` and `adapt_val` for early stopping) and
       80% of recordings reserved exclusively for `test`. Halts with an explicit
       `InsufficientRecordingsError` if a class has fewer than the required
       minimum recordings (never falls back silently to window-level splitting).
  4. Leak-free reference RMS (`detrended_rms`) and feature scaling:
     - DA: `RMS_ref` computed exclusively from `Normal` signals in `adapt_train`
       (or adaptation partition) and applied to both adaptation and test windows;
       test windows are never used to estimate statistics.
     - DG: `RMS_ref` and `StandardScaler` estimated exclusively on source domains;
       zero target signals or labels are used before final evaluation.
  5. Hierarchical source balancing (`balance_hierarchical_sources`):
     - Equalizes the contribution of each source dataset and each present class
       within each source dataset via sample weights or stratified resampling,
       without applying cross-domain SMOTE and without modifying validation or test sets.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.vibnet.evaluation.balancing import trim_single_remainder_sample
from src.vibnet.preprocessing.signal_norm import (
    EPSILON_RMS,
    compute_reference_rms,
    detrend_linear_window,
)

# =============================================================================
# 1. CANONICAL SOURCE CONFIGURATIONS, TARGETS & 5-CLASS HARMONIZED VOCABULARY
# =============================================================================

SOURCE_CONFIGS: Dict[str, Tuple[str, ...]] = {
    "S1": ("UORED",),
    "S2": ("HUST",),
    "S3": ("MFPT",),
    "S4": ("UORED", "HUST", "MFPT"),
}

DA_DG_TARGETS: Tuple[str, ...] = ("CWRU_12k", "CWRU_48k", "PU")
TL_TARGETS: Tuple[str, ...] = ("CWRU_12k", "CWRU_48k", "PU", "HUST_Gearbox", "UOEMD")
CANONICAL_SEEDS: Tuple[int, ...] = (42, 123, 2026)

HARMONIZED_CLASSES: Tuple[str, ...] = (
    "Normal",
    "Ball",
    "Inner",
    "Outer",
    "Inner & Outer",
)
HARMONIZED_CLASS_TO_ID: Dict[str, int] = {
    name: idx for idx, name in enumerate(HARMONIZED_CLASSES)
}
ID_TO_HARMONIZED_CLASS: Dict[int, str] = {
    idx: name for idx, name in enumerate(HARMONIZED_CLASSES)
}
NUM_HARMONIZED_CLASSES: int = len(HARMONIZED_CLASSES)  # 5

# Verified mapping from on-disk folder names (`Class_*`) to the 5 harmonized classes.
# Classes outside the 5-class bearing ontology (such as `Class_Cage` in UORED and
# `Class_44`/`Class_46` in HUST) map to `None` (excluded from 5-class supervised head,
# optionally usable in unsupervised TabNetPretrainer).
DATASET_TO_HARMONIZED_CLASS_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "UORED": {
        "Class_Normal": "Normal",
        "Class_Healthy": "Normal",
        "Class_H": "Normal",
        "Class_47": "Normal",
        "Class_Ball": "Ball",
        "Class_B": "Ball",
        "Class_50": "Ball",
        "Class_Cage": None,
        "Class_C": None,
        "Class_51": None,
    },
    "HUST": {
        "Class_Normal": "Normal",
        "Class_N": "Normal",
        "Class_40": "Normal",
        "Class_43": "Ball",
        "Class_B": "Ball",
        "Class_41": "Inner",
        "Class_I": "Inner",
        "Class_42": "Outer",
        "Class_O": "Outer",
        "Class_45": "Inner & Outer",
        "Class_IO": "Inner & Outer",
        "Class_44": None,  # Compound Inner + Ball (outside 5-class vocabulary)
        "Class_IB": None,
        "Class_46": None,  # Compound Outer + Ball (outside 5-class vocabulary)
        "Class_OB": None,
    },
    "MFPT": {
        "Class_23": None,  # Per protocol Section 3: MFPT supervised annotations are Inner & Outer
        "Class_Normal": None,
        "Class_24": "Inner",
        "Class_Inner": "Inner",
        "Class_25": "Outer",
        "Class_Outer": "Outer",
    },
    "CWRU_12k": {
        "Class_0": "Normal",
        "Class_Normal": "Normal",
        "Class_1": "Inner",
        "Class_Inner": "Inner",
        "Class_2": "Outer",  # Verified in vibdata/raw/labels.csv: 2 = Outer Race
        "Class_Outer": "Outer",
        "Class_3": "Ball",   # Verified in vibdata/raw/labels.csv: 3 = Ball
        "Class_Ball": "Ball",
    },
    "CWRU_48k": {
        "Class_1": "Inner",
        "Class_Inner": "Inner",
        "Class_2": "Outer",
        "Class_Outer": "Outer",
        "Class_3": "Ball",
        "Class_Ball": "Ball",
    },
    "PU": {
        "Class_Normal": "Normal",
        "Class_26": "Normal",
        "Class_27": "Outer",
        "Class_Outer": "Outer",
        "Class_28": "Inner",
        "Class_Inner": "Inner",
        "Class_29": "Inner & Outer",
        "Class_Inner & Outer": "Inner & Outer",
    },
}

WINDOW_FILENAME_RE = re.compile(r"^(s\d+)_w(\d+)\.npy$", re.IGNORECASE)


class InsufficientRecordingsError(RuntimeError):
    """
    Raised when a dataset class does not have enough distinct original recordings
    to perform a recording-grouped split (e.g., DA 20% adaptation -> 80/20 train/val
    and 80% test). Never falls back silently to window-level splitting.
    """


class IncomparableSourceRMSReferenceError(ValueError):
    """
    Raised when attempting an unapproved direct arithmetic average of source RMS
    references across datasets with incompatible physical units or amplitude scales.
    """


# =============================================================================
# 2. RECORDING PROVENANCE & HARMONIZED CLASS MAPPING
# =============================================================================


@dataclass(frozen=True)
class WindowRecord:
    """Traces a single 1-second window back to its exact original recording."""

    sample_id: str
    dataset: str
    condition: str
    raw_class: str
    harmonized_class: Optional[str]
    harmonized_class_id: Optional[int]
    recording_prefix: str
    recording_key: str
    window_index: int


def parse_window_filename(filename: str) -> Tuple[str, int]:
    """
    Extract `(recording_prefix, window_index)` from a window filename `sNNNNN_wNN.npy`.
    Raises `ValueError` if the filename does not match the canonical pattern.
    """
    match = WINDOW_FILENAME_RE.fullmatch(filename.strip())
    if not match:
        raise ValueError(
            f"Window filename '{filename}' does not encode a valid recording ID "
            "(expected 'sNNNNN_wNN.npy')."
        )
    return match.group(1).lower(), int(match.group(2))


def build_recording_key(
    dataset: str,
    condition: str,
    raw_class: str,
    recording_prefix: str,
) -> str:
    """
    Build a globally unique recording identifier that guarantees all windows
    derived from the same physical recording stay grouped together.
    """
    return f"{dataset}::{condition}::{raw_class}::{recording_prefix.lower()}"


def map_to_harmonized_class(
    dataset: str,
    raw_class: str,
    allow_mfpt_normal: bool = False,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Map a dataset-native folder label (`raw_class`) to `(harmonized_name, harmonized_id)`
    in the fixed 5-class vocabulary (`0: Normal`, `1: Ball`, `2: Inner`, `3: Outer`,
    `4: Inner & Outer`).

    By default (`allow_mfpt_normal=False`), strictly enforces Section 3 of the protocol:
      - `UORED`: `Normal`, `Ball`
      - `HUST`: `Normal`, `Ball`, `Inner`, `Outer`, `Inner & Outer`
      - `MFPT`: `Inner`, `Outer` (while `MFPT` `Class_23` remains available for
        calculating healthy `RMS_ref`).
    """
    if dataset == "HUST_Bearing":
        dataset = "HUST"
    if dataset not in DATASET_TO_HARMONIZED_CLASS_MAP:
        raise KeyError(
            f"Dataset '{dataset}' has no mapping to the 5-class bearing ontology."
        )
    norm_raw = raw_class if raw_class.startswith("Class_") else f"Class_{raw_class}"
    ds_map = DATASET_TO_HARMONIZED_CLASS_MAP[dataset]
    if norm_raw not in ds_map and raw_class not in ds_map:
        raise ValueError(
            f"Unknown raw class '{raw_class}' for dataset '{dataset}'. "
            f"Known classes: {sorted(ds_map.keys())}"
        )
    harm_name = ds_map.get(norm_raw, ds_map.get(raw_class))
    if dataset == "MFPT" and allow_mfpt_normal and norm_raw in ("Class_23", "Class_Normal"):
        harm_name = "Normal"
    if harm_name is None:
        return None, None
    return harm_name, HARMONIZED_CLASS_TO_ID[harm_name]


def diagnose_source_config_class_coverage(
    source_config_id: str,
    allow_mfpt_normal: bool = False,
) -> Dict[str, Any]:
    """
    Analyze which of the 5 harmonized classes (`Normal`, `Ball`, `Inner`, `Outer`,
    `Inner & Outer`) are present vs. absent in source configuration `S1..S4`.
    """
    if source_config_id not in SOURCE_CONFIGS:
        raise KeyError(
            f"Unknown source configuration '{source_config_id}'. "
            f"Valid configurations: {sorted(SOURCE_CONFIGS.keys())}"
        )
    sources = SOURCE_CONFIGS[source_config_id]
    present_names: set[str] = set()
    by_source: Dict[str, List[str]] = {}
    for s_name in sources:
        s_classes = sorted(
            {
                harm
                for raw_c in DATASET_TO_HARMONIZED_CLASS_MAP[s_name]
                for harm, _ in [
                    map_to_harmonized_class(
                        s_name, raw_c, allow_mfpt_normal=allow_mfpt_normal
                    )
                ]
                if harm is not None
            }
        )
        by_source[s_name] = s_classes
        present_names.update(s_classes)

    present_ids = sorted(HARMONIZED_CLASS_TO_ID[c] for c in present_names)
    absent_ids = [
        idx for idx in range(NUM_HARMONIZED_CLASSES) if idx not in present_ids
    ]
    absent_names = [ID_TO_HARMONIZED_CLASS[idx] for idx in absent_ids]

    return {
        "source_config_id": source_config_id,
        "sources": list(sources),
        "classes_by_source": by_source,
        "present_class_ids": present_ids,
        "present_class_names": [ID_TO_HARMONIZED_CLASS[i] for i in present_ids],
        "absent_class_ids": absent_ids,
        "absent_class_names": absent_names,
        "full_5_class_coverage": len(present_ids) == NUM_HARMONIZED_CLASSES,
        "head_output_dim": NUM_HARMONIZED_CLASSES,
    }


# =============================================================================
# 3. RECORDING-GROUPED SPLITTING & LEAKAGE VERIFICATION (DA & DG)
# =============================================================================


@dataclass(frozen=True)
class RecordingSplitManifest:
    """
    Serializable, deterministic manifest of recording-grouped partitions
    (`adapt_train`, `adapt_val`, `test` for DA; `source_train`, `source_val` for DG).
    """

    protocol: str
    dataset_or_config: str
    seed: int
    partition_recordings: Dict[str, Tuple[str, ...]]
    partition_sample_ids: Dict[str, Tuple[str, ...]]
    recording_counts_by_class: Dict[str, Dict[str, int]]
    window_counts_by_class: Dict[str, Dict[str, int]]
    recording_proportions: Dict[str, float]
    window_proportions: Dict[str, float]
    split_signature: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "dataset_or_config": self.dataset_or_config,
            "seed": self.seed,
            "partition_recordings": {
                k: list(v) for k, v in self.partition_recordings.items()
            },
            "partition_sample_ids": {
                k: list(v) for k, v in self.partition_sample_ids.items()
            },
            "recording_counts_by_class": self.recording_counts_by_class,
            "window_counts_by_class": self.window_counts_by_class,
            "recording_proportions": self.recording_proportions,
            "window_proportions": self.window_proportions,
            "split_signature": self.split_signature,
        }


def assert_zero_recording_overlap(
    records: Sequence[WindowRecord],
    partition_indices: Mapping[str, Sequence[int]],
) -> None:
    """
    Verify that no `recording_key` (or `sample_id`) appears in more than one
    partition (`train`, `val`, `test`, `adapt_train`, `adapt_val`).
    Raises `ValueError` immediately if any overlap is found.
    """
    part_names = sorted(partition_indices.keys())
    rec_sets: Dict[str, set[str]] = {}
    sample_sets: Dict[str, set[str]] = {}

    for p_name in part_names:
        idxs = partition_indices[p_name]
        rec_sets[p_name] = {records[i].recording_key for i in idxs}
        sample_sets[p_name] = {records[i].sample_id for i in idxs}

    for i in range(len(part_names)):
        for j in range(i + 1, len(part_names)):
            p_a, p_b = part_names[i], part_names[j]
            overlap_rec = rec_sets[p_a] & rec_sets[p_b]
            if overlap_rec:
                raise ValueError(
                    f"Recording leakage detected between partitions '{p_a}' and '{p_b}': "
                    f"{sorted(overlap_rec)[:5]}"
                )
            overlap_smp = sample_sets[p_a] & sample_sets[p_b]
            if overlap_smp:
                raise ValueError(
                    f"Sample ID leakage detected between partitions '{p_a}' and '{p_b}': "
                    f"{sorted(overlap_smp)[:5]}"
                )


def create_da_recording_split(
    records: Sequence[WindowRecord],
    seed: int,
    adapt_fraction: float = 0.20,
    adapt_val_fraction: float = 0.20,
    min_adapt_train_recordings_per_class: int = 1,
    min_adapt_val_recordings_per_class: int = 1,
    min_test_recordings_per_class: int = 1,
) -> Tuple[Dict[str, Tuple[int, ...]], RecordingSplitManifest]:
    """
    Partition target dataset windows by original recording (`recording_key`) for
    Domain Adaptation (DA) per Section 5.1:
      - For each present harmonized class, reserve ~20% of recordings for adaptation
        and ~80% of recordings exclusively for test.
      - Within the 20% adaptation recordings of each class, split into `adapt_train`
        (80%) and `adapt_val` (20%) also strictly by recording.
      - Requires at least `min_adapt_train_recordings_per_class +
        min_adapt_val_recordings_per_class` adaptation recordings and
        `min_test_recordings_per_class` test recordings per class.
      - If any class has insufficient recordings (for example, when 20% of 8 recordings
        rounds to only 2 recordings, or < 2 adaptation recordings prevent a recording-level
        train/val split), raises `InsufficientRecordingsError` when strict minimums are
        not met — NEVER falls back silently to window-level splitting.
    """
    if not (0.0 < adapt_fraction < 1.0) or not (0.0 < adapt_val_fraction < 1.0):
        raise ValueError("adapt_fraction and adapt_val_fraction must be in (0, 1).")

    supervised_indices = [
        idx for idx, r in enumerate(records) if r.harmonized_class_id is not None
    ]
    if not supervised_indices:
        raise ValueError("No records with valid harmonized classes found for DA split.")

    datasets_in_records = sorted({records[i].dataset for i in supervised_indices})
    ds_label = "|".join(datasets_in_records)

    # Group window indices by harmonized_class -> recording_key
    class_to_rec_windows: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for idx in supervised_indices:
        rec = records[idx]
        assert rec.harmonized_class is not None
        class_to_rec_windows[rec.harmonized_class][rec.recording_key].append(idx)

    min_required_total = (
        min_adapt_train_recordings_per_class
        + min_adapt_val_recordings_per_class
        + min_test_recordings_per_class
    )

    adapt_train_indices: List[int] = []
    adapt_val_indices: List[int] = []
    test_indices: List[int] = []

    rec_counts_by_cls: Dict[str, Dict[str, int]] = {}
    win_counts_by_cls: Dict[str, Dict[str, int]] = {}

    for cls_name in sorted(class_to_rec_windows.keys()):
        rec_map = class_to_rec_windows[cls_name]
        unique_recs = sorted(rec_map.keys())
        n_recs = len(unique_recs)

        if n_recs < min_required_total:
            raise InsufficientRecordingsError(
                f"Dataset '{ds_label}' class '{cls_name}' has only {n_recs} unique "
                f"recordings, but at least {min_required_total} recordings are required "
                f"for a recording-grouped DA split (adapt_train>={min_adapt_train_recordings_per_class}, "
                f"adapt_val>={min_adapt_val_recordings_per_class}, test>={min_test_recordings_per_class}). "
                "Window-level splitting is strictly prohibited."
            )

        n_adapt = int(round(n_recs * adapt_fraction))
        min_adapt = (
            min_adapt_train_recordings_per_class + min_adapt_val_recordings_per_class
        )
        if n_adapt < min_adapt:
            raise InsufficientRecordingsError(
                f"Dataset '{ds_label}' class '{cls_name}' has {n_recs} recordings; "
                f"20% adaptation yields {n_adapt} recordings (< {min_adapt} required to "
                f"split into recording-level adapt_train and adapt_val without window leakage)."
            )
        if (n_recs - n_adapt) < min_test_recordings_per_class:
            raise InsufficientRecordingsError(
                f"Dataset '{ds_label}' class '{cls_name}' leaves fewer than "
                f"{min_test_recordings_per_class} test recordings."
            )

        # Deterministic shuffle per (dataset, class, seed)
        cls_seed_token = f"DA::{ds_label}::{cls_name}::{int(seed)}"
        cls_seed = int(
            hashlib.sha256(cls_seed_token.encode("utf-8")).hexdigest()[:8], 16
        )
        rng = random.Random(cls_seed)
        shuffled_recs = list(unique_recs)
        rng.shuffle(shuffled_recs)

        adapt_recs = shuffled_recs[:n_adapt]
        test_recs = shuffled_recs[n_adapt:]

        n_adapt_val = max(
            min_adapt_val_recordings_per_class,
            int(round(len(adapt_recs) * adapt_val_fraction)),
        )
        if len(adapt_recs) - n_adapt_val < min_adapt_train_recordings_per_class:
            n_adapt_val = len(adapt_recs) - min_adapt_train_recordings_per_class

        if (
            n_adapt_val < min_adapt_val_recordings_per_class
            or (len(adapt_recs) - n_adapt_val) < min_adapt_train_recordings_per_class
        ):
            raise InsufficientRecordingsError(
                f"Cannot split {len(adapt_recs)} adaptation recordings for class '{cls_name}' "
                f"into adapt_train (>={min_adapt_train_recordings_per_class}) and "
                f"adapt_val (>={min_adapt_val_recordings_per_class})."
            )

        adapt_val_recs = sorted(adapt_recs[:n_adapt_val])
        adapt_train_recs = sorted(adapt_recs[n_adapt_val:])
        test_recs_sorted = sorted(test_recs)

        cls_tr_win = [i for r_key in adapt_train_recs for i in rec_map[r_key]]
        cls_val_win = [i for r_key in adapt_val_recs for i in rec_map[r_key]]
        cls_te_win = [i for r_key in test_recs_sorted for i in rec_map[r_key]]

        adapt_train_indices.extend(cls_tr_win)
        adapt_val_indices.extend(cls_val_win)
        test_indices.extend(cls_te_win)

        rec_counts_by_cls[cls_name] = {
            "total": n_recs,
            "adapt_train": len(adapt_train_recs),
            "adapt_val": len(adapt_val_recs),
            "adapt_total": len(adapt_train_recs) + len(adapt_val_recs),
            "test": len(test_recs_sorted),
        }
        win_counts_by_cls[cls_name] = {
            "total": len(cls_tr_win) + len(cls_val_win) + len(cls_te_win),
            "adapt_train": len(cls_tr_win),
            "adapt_val": len(cls_val_win),
            "adapt_total": len(cls_tr_win) + len(cls_val_win),
            "test": len(cls_te_win),
        }

    partitions: Dict[str, Tuple[int, ...]] = {
        "adapt_train": tuple(sorted(adapt_train_indices)),
        "adapt_val": tuple(sorted(adapt_val_indices)),
        "test": tuple(sorted(test_indices)),
    }
    assert_zero_recording_overlap(records, partitions)

    part_recs = {
        k: tuple(sorted({records[i].recording_key for i in idxs}))
        for k, idxs in partitions.items()
    }
    part_samples = {
        k: tuple(records[i].sample_id for i in idxs)
        for k, idxs in partitions.items()
    }

    total_recs = sum(len(v) for v in part_recs.values())
    total_wins = sum(len(v) for v in part_samples.values())

    rec_props = {
        k: (len(v) / float(total_recs) if total_recs > 0 else 0.0)
        for k, v in part_recs.items()
    }
    rec_props["adapt_total"] = rec_props["adapt_train"] + rec_props["adapt_val"]

    win_props = {
        k: (len(v) / float(total_wins) if total_wins > 0 else 0.0)
        for k, v in part_samples.items()
    }
    win_props["adapt_total"] = win_props["adapt_train"] + win_props["adapt_val"]

    sig_payload = json.dumps(
        {
            "protocol": "DA",
            "dataset": ds_label,
            "seed": int(seed),
            "recordings": {k: list(v) for k, v in part_recs.items()},
        },
        sort_keys=True,
    )
    split_sig = hashlib.sha256(sig_payload.encode("utf-8")).hexdigest()[:16]

    manifest = RecordingSplitManifest(
        protocol="DA",
        dataset_or_config=ds_label,
        seed=int(seed),
        partition_recordings=part_recs,
        partition_sample_ids=part_samples,
        recording_counts_by_class=rec_counts_by_cls,
        window_counts_by_class=win_counts_by_cls,
        recording_proportions=rec_props,
        window_proportions=win_props,
        split_signature=split_sig,
    )
    return partitions, manifest


def create_source_recording_split(
    records: Sequence[WindowRecord],
    seed: int,
    val_fraction: float = 0.20,
    source_config_id: str = "S4",
) -> Tuple[Dict[str, Tuple[int, ...]], RecordingSplitManifest]:
    """
    Partition source domain records (`UORED`, `HUST`, `MFPT`) into `source_train` (80%)
    and `source_val` (20%) strictly by original recording (`recording_key`), stratified
    by `(dataset, harmonized_class)`.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1).")

    supervised_indices = [
        idx for idx, r in enumerate(records) if r.harmonized_class_id is not None
    ]
    if not supervised_indices:
        raise ValueError("No supervised source records available for source split.")

    group_to_rec_windows: Dict[Tuple[str, str], Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for idx in supervised_indices:
        rec = records[idx]
        assert rec.harmonized_class is not None
        group_to_rec_windows[(rec.dataset, rec.harmonized_class)][
            rec.recording_key
        ].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    rec_counts_by_cls: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "source_train": 0, "source_val": 0}
    )
    win_counts_by_cls: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "source_train": 0, "source_val": 0}
    )

    for (ds_name, cls_name), rec_map in sorted(group_to_rec_windows.items()):
        unique_recs = sorted(rec_map.keys())
        n_recs = len(unique_recs)
        if n_recs < 2:
            raise InsufficientRecordingsError(
                f"Source dataset '{ds_name}' class '{cls_name}' has only {n_recs} "
                "recording(s); at least 2 are required for a recording-level train/val split."
            )
        n_val = max(1, int(round(n_recs * val_fraction)))
        if n_recs - n_val < 1:
            n_val = n_recs - 1

        seed_token = f"SRC::{source_config_id}::{ds_name}::{cls_name}::{int(seed)}"
        grp_seed = int(
            hashlib.sha256(seed_token.encode("utf-8")).hexdigest()[:8], 16
        )
        rng = random.Random(grp_seed)
        shuffled = list(unique_recs)
        rng.shuffle(shuffled)

        val_recs = sorted(shuffled[:n_val])
        tr_recs = sorted(shuffled[n_val:])

        tr_wins = [i for r_key in tr_recs for i in rec_map[r_key]]
        val_wins = [i for r_key in val_recs for i in rec_map[r_key]]

        train_indices.extend(tr_wins)
        val_indices.extend(val_wins)

        rec_counts_by_cls[cls_name]["total"] += n_recs
        rec_counts_by_cls[cls_name]["source_train"] += len(tr_recs)
        rec_counts_by_cls[cls_name]["source_val"] += len(val_recs)

        win_counts_by_cls[cls_name]["total"] += len(tr_wins) + len(val_wins)
        win_counts_by_cls[cls_name]["source_train"] += len(tr_wins)
        win_counts_by_cls[cls_name]["source_val"] += len(val_wins)

    partitions: Dict[str, Tuple[int, ...]] = {
        "source_train": tuple(sorted(train_indices)),
        "source_val": tuple(sorted(val_indices)),
    }
    assert_zero_recording_overlap(records, partitions)

    part_recs = {
        k: tuple(sorted({records[i].recording_key for i in idxs}))
        for k, idxs in partitions.items()
    }
    part_samples = {
        k: tuple(records[i].sample_id for i in idxs)
        for k, idxs in partitions.items()
    }
    total_recs = sum(len(v) for v in part_recs.values())
    total_wins = sum(len(v) for v in part_samples.values())

    rec_props = {
        k: (len(v) / float(total_recs) if total_recs > 0 else 0.0)
        for k, v in part_recs.items()
    }
    win_props = {
        k: (len(v) / float(total_wins) if total_wins > 0 else 0.0)
        for k, v in part_samples.items()
    }

    sig_payload = json.dumps(
        {
            "protocol": "DG_SOURCE",
            "config": source_config_id,
            "seed": int(seed),
            "recordings": {k: list(v) for k, v in part_recs.items()},
        },
        sort_keys=True,
    )
    split_sig = hashlib.sha256(sig_payload.encode("utf-8")).hexdigest()[:16]

    manifest = RecordingSplitManifest(
        protocol="DG_SOURCE",
        dataset_or_config=source_config_id,
        seed=int(seed),
        partition_recordings=part_recs,
        partition_sample_ids=part_samples,
        recording_counts_by_class=dict(rec_counts_by_cls),
        window_counts_by_class=dict(win_counts_by_cls),
        recording_proportions=rec_props,
        window_proportions=win_props,
        split_signature=split_sig,
    )
    return partitions, manifest


# =============================================================================
# 4. LEAK-FREE RMS NORMALIZATION FOR DA AND DG
# =============================================================================


@dataclass(frozen=True)
class DARMSNormalizer:
    """
    Leak-free Domain Adaptation RMS reference normalizer (Section 4.1):
      - Estimates `rms_ref` exclusively from `Normal` windows in the target's
        adaptation training split (`adapt_train` / `adapt`).
      - Applies the exact same `rms_ref` to both adaptation and test windows.
      - Never touches test windows during `fit`.
    """

    target_dataset: str
    rms_ref: float
    n_reference_windows: int
    reference_recording_keys: Tuple[str, ...]
    apply_detrend: bool = True

    def transform_windows(self, windows: Sequence[np.ndarray]) -> List[np.ndarray]:
        denom = max(float(self.rms_ref), EPSILON_RMS)
        out: List[np.ndarray] = []
        for w in windows:
            arr = np.asarray(w, dtype=np.float64).reshape(-1)
            if self.apply_detrend:
                arr = np.asarray(detrend_linear_window(arr), dtype=np.float64)
            out.append((arr / denom).astype(np.float32))
        return out


def fit_da_rms_normalizer(
    records: Sequence[WindowRecord],
    windows: Sequence[np.ndarray],
    adapt_indices: Sequence[int],
    test_indices: Sequence[int],
    apply_detrend: bool = True,
    cross_reference_normal_windows: Optional[Sequence[np.ndarray]] = None,
    cross_reference_recording_keys: Optional[Sequence[str]] = None,
) -> DARMSNormalizer:
    """
    Fit `DARMSNormalizer` exclusively on `Normal` windows inside `adapt_indices`.
    Verifies zero index or recording overlap with `test_indices`.
    """
    adapt_set = set(int(i) for i in adapt_indices)
    test_set = set(int(i) for i in test_indices)
    if adapt_set & test_set:
        raise ValueError(
            "DA RMS normalizer received overlapping adaptation and test indices!"
        )
    adapt_recs = {records[i].recording_key for i in adapt_set}
    test_recs = {records[i].recording_key for i in test_set}
    if adapt_recs & test_recs:
        raise ValueError(
            "DA RMS normalizer detected recording leakage between adaptation and test!"
        )

    normal_indices = [
        i
        for i in sorted(adapt_set)
        if records[i].harmonized_class == "Normal"
        or records[i].raw_class in ("Class_0", "Class_Normal", "Class_23")
    ]

    ds_name = records[0].dataset if records else "UNKNOWN"
    if normal_indices:
        ref_windows: List[np.ndarray] = []
        for idx in normal_indices:
            arr = np.asarray(windows[idx], dtype=np.float64).reshape(-1)
            if apply_detrend:
                arr = np.asarray(detrend_linear_window(arr), dtype=np.float64)
            ref_windows.append(arr)
        rms_val = compute_reference_rms(ref_windows)
        ref_recs = tuple(sorted({records[i].recording_key for i in normal_indices}))
        return DARMSNormalizer(
            target_dataset=ds_name,
            rms_ref=float(rms_val),
            n_reference_windows=len(ref_windows),
            reference_recording_keys=ref_recs,
            apply_detrend=apply_detrend,
        )

    if cross_reference_normal_windows is not None and len(cross_reference_normal_windows) > 0:
        ref_windows = []
        for w in cross_reference_normal_windows:
            arr = np.asarray(w, dtype=np.float64).reshape(-1)
            if apply_detrend:
                arr = np.asarray(detrend_linear_window(arr), dtype=np.float64)
            ref_windows.append(arr)
        rms_val = compute_reference_rms(ref_windows)
        ref_recs = tuple(sorted(set(cross_reference_recording_keys or ())))
        return DARMSNormalizer(
            target_dataset=ds_name,
            rms_ref=float(rms_val),
            n_reference_windows=len(ref_windows),
            reference_recording_keys=ref_recs,
            apply_detrend=apply_detrend,
        )

    raise ValueError(
        f"Target dataset '{ds_name}' has 0 Normal windows in the adaptation partition "
        "and no explicit cross_reference_normal_windows were provided."
    )


@dataclass(frozen=True)
class DGRMSNormalizer:
    """
    Strict Domain Generalization RMS normalizer (Section 4.2):
      - Computed exclusively from source datasets (`source_rms_refs`).
      - Zero target data or statistics are ever inspected.
    """

    source_rms_refs: Dict[str, float]
    source_units: Dict[str, str]
    aggregation_mode: str
    global_target_rms_ref: float
    comparability_report: Dict[str, Any]
    apply_detrend: bool = True

    def transform_source_windows(
        self,
        source_name: str,
        windows: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        if source_name not in self.source_rms_refs:
            raise KeyError(f"Unknown source dataset '{source_name}'.")
        denom = max(float(self.source_rms_refs[source_name]), EPSILON_RMS)
        out: List[np.ndarray] = []
        for w in windows:
            arr = np.asarray(w, dtype=np.float64).reshape(-1)
            if self.apply_detrend:
                arr = np.asarray(detrend_linear_window(arr), dtype=np.float64)
            out.append((arr / denom).astype(np.float32))
        return out

    def transform_target_windows(
        self,
        windows: Sequence[np.ndarray],
        target_dataset: Optional[str] = None,
    ) -> List[np.ndarray]:
        # Target dataset is never used to estimate statistics; transform uses only
        # `global_target_rms_ref` estimated strictly on the sources.
        _ = target_dataset
        denom = max(float(self.global_target_rms_ref), EPSILON_RMS)
        out: List[np.ndarray] = []
        for w in windows:
            arr = np.asarray(w, dtype=np.float64).reshape(-1)
            if self.apply_detrend:
                arr = np.asarray(detrend_linear_window(arr), dtype=np.float64)
            out.append((arr / denom).astype(np.float32))
        return out


def assess_source_rms_comparability(
    source_rms_refs: Mapping[str, float],
    source_units: Mapping[str, str],
    max_allowed_ratio: float = 5.0,
) -> Dict[str, Any]:
    """
    Check whether source datasets have comparable physical units and healthy RMS scales
    (Section 4.2: "Verificar unidades, escalas, taxas de amostragem, amplitudes e
    estatísticas dos sinais antes de aprovar essa regra. Caso as referências não sejam
    comparáveis, não implemente silenciosamente a média.").
    """
    if not source_rms_refs:
        raise ValueError("source_rms_refs cannot be empty.")

    vals = [float(v) for v in source_rms_refs.values()]
    if any(v <= 0.0 or not math.isfinite(v) for v in vals):
        raise ValueError(f"Invalid source RMS references: {dict(source_rms_refs)}")

    units_set = {str(source_units.get(k, "unknown")) for k in source_rms_refs}
    min_rms = min(vals)
    max_rms = max(vals)
    scale_ratio = max_rms / max(min_rms, EPSILON_RMS)

    units_compatible = len(units_set) == 1 and "unknown" not in units_set
    scales_comparable = scale_ratio <= float(max_allowed_ratio)
    is_directly_comparable = bool(units_compatible and scales_comparable)

    arith_mean = sum(vals) / float(len(vals))
    geom_mean = math.exp(sum(math.log(v) for v in vals) / float(len(vals)))

    reasons: List[str] = []
    if not units_compatible:
        reasons.append(
            f"Heterogeneous physical units across sources: {dict(source_units)}"
        )
    if not scales_comparable:
        reasons.append(
            f"Source healthy RMS_ref spans a {scale_ratio:.2f}x factor "
            f"(min={min_rms:.6f}, max={max_rms:.6f} > max_allowed_ratio={max_allowed_ratio:.2f}x)"
        )

    return {
        "is_directly_comparable": is_directly_comparable,
        "units_compatible": units_compatible,
        "scales_comparable": scales_comparable,
        "scale_ratio_max_over_min": float(scale_ratio),
        "arithmetic_mean_rms_ref": float(arith_mean),
        "geometric_mean_rms_ref": float(geom_mean),
        "source_rms_refs": {k: float(v) for k, v in source_rms_refs.items()},
        "source_units": {k: str(source_units.get(k, "unknown")) for k in source_rms_refs},
        "incomparability_reasons": reasons,
    }


def fit_dg_rms_normalizer(
    source_normal_windows: Mapping[str, Sequence[np.ndarray]],
    source_units: Mapping[str, str],
    aggregation_mode: str = "strict_check_arithmetic",
    max_allowed_ratio: float = 5.0,
    apply_detrend: bool = True,
) -> DGRMSNormalizer:
    """
    Fit `DGRMSNormalizer` using ONLY source-domain healthy windows (zero target data).

    Supported `aggregation_mode` values:
      - `"strict_check_arithmetic"`: computes the arithmetic mean of source `RMS_ref`
        ONLY if `assess_source_rms_comparability` passes; otherwise raises
        `IncomparableSourceRMSReferenceError` (preventing silent averaging of
        incompatible scales/units per Section 4.2).
      - `"explicit_geometric_mean"`: uses the geometric mean across source `RMS_ref`
        values after explicit researcher approval, recording the comparability report.
      - `"explicit_arithmetic_mean"`: uses the arithmetic mean across source `RMS_ref`
        values after explicit researcher approval, recording the comparability report.
    """
    if not source_normal_windows:
        raise ValueError("source_normal_windows cannot be empty for DG normalization.")

    source_rms_refs: Dict[str, float] = {}
    for s_name, win_list in source_normal_windows.items():
        if not win_list:
            raise ValueError(f"Source '{s_name}' provided 0 Normal windows for RMS_ref.")
        det_list: List[np.ndarray] = []
        for w in win_list:
            arr = np.asarray(w, dtype=np.float64).reshape(-1)
            if apply_detrend:
                arr = np.asarray(detrend_linear_window(arr), dtype=np.float64)
            det_list.append(arr)
        source_rms_refs[s_name] = float(compute_reference_rms(det_list))

    report = assess_source_rms_comparability(
        source_rms_refs=source_rms_refs,
        source_units=source_units,
        max_allowed_ratio=max_allowed_ratio,
    )

    if aggregation_mode == "strict_check_arithmetic":
        if not report["is_directly_comparable"]:
            raise IncomparableSourceRMSReferenceError(
                "Cannot silently average source RMS_ref values in DG: "
                + "; ".join(report["incomparability_reasons"])
                + ". Choose an explicitly approved mode ('explicit_geometric_mean' or "
                "'explicit_arithmetic_mean' or scale-invariant 126-feature projection)."
            )
        global_ref = report["arithmetic_mean_rms_ref"]
    elif aggregation_mode == "explicit_geometric_mean":
        global_ref = report["geometric_mean_rms_ref"]
    elif aggregation_mode == "explicit_arithmetic_mean":
        global_ref = report["arithmetic_mean_rms_ref"]
    else:
        raise ValueError(
            f"Unknown DG aggregation_mode '{aggregation_mode}'."
        )

    return DGRMSNormalizer(
        source_rms_refs=source_rms_refs,
        source_units={k: str(source_units.get(k, "unknown")) for k in source_rms_refs},
        aggregation_mode=aggregation_mode,
        global_target_rms_ref=float(global_ref),
        comparability_report=report,
        apply_detrend=apply_detrend,
    )


# =============================================================================
# 5. HIERARCHICAL SOURCE BALANCING (NO CROSS-DOMAIN SMOTE, NO VAL/TEST BALANCING)
# =============================================================================


@dataclass(frozen=True)
class HierarchicalBalanceResult:
    """
    Contains hierarchical source-domain and class-within-domain balancing weights
    and/or resampled indices, along with full distribution audit metadata (Section 6).
    """

    sample_weights: np.ndarray
    resampled_indices: np.ndarray
    distribution_before: Dict[str, Dict[str, int]]
    effective_weight_by_dataset: Dict[str, float]
    effective_weight_by_dataset_class: Dict[str, Dict[str, float]]
    resampled_distribution_after: Dict[str, Dict[str, int]]
    mode: str


def compute_hierarchical_source_balance(
    datasets: Sequence[str],
    class_ids: Sequence[int],
    mode: str = "hierarchical_weights",
    seed: int = 42,
    partition_name: str = "train",
) -> HierarchicalBalanceResult:
    """
    Implement the hierarchical balancing protocol (Section 6):
      1. Equalize the total contribution of each source dataset `d in {1..D}` (`1 / D`).
      2. Within each source dataset `d`, equalize the contribution of each present
         class `c in C_d` (`1 / |C_d|`).
      3. For sample `i` belonging to `(d, c)` with count `N_{d,c}`, its normalized
         hierarchical weight is:
           `w_i = N_total / (D * |C_d| * N_{d,c})`
         so that `sum_i w_i == N_total`, every dataset `d` has total weight `N_total / D`,
         and every present class `c` inside dataset `d` has total weight
         `N_total / (D * |C_d|)`.
      4. Also supports `"hierarchical_resample"` (stratified within-domain resampling
         with replacement to an equal target quota per `(d, c)`, never interpolating
         across domains via SMOTE).
      5. Rejects any call where `partition_name` is `"val"`, `"validation"`, or `"test"`.
    """
    if partition_name.strip().lower() in ("val", "validation", "test", "adapt_val"):
        raise ValueError(
            f"Balancing is strictly prohibited on partition '{partition_name}' (Section 6.4)."
        )
    if len(datasets) != len(class_ids) or len(datasets) == 0:
        raise ValueError("datasets and class_ids must have the same non-zero length.")
    if mode not in ("hierarchical_weights", "hierarchical_resample", "none"):
        raise ValueError(
            f"Unsupported hierarchical balance mode '{mode}'. "
            "Expected 'hierarchical_weights', 'hierarchical_resample', or 'none'."
        )

    n_total = len(datasets)
    counts_by_ds_cls: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    indices_by_ds_cls: Dict[Tuple[str, int], List[int]] = defaultdict(list)

    for idx, (ds, cid) in enumerate(zip(datasets, class_ids)):
        ds_str = str(ds)
        c_int = int(cid)
        counts_by_ds_cls[ds_str][c_int] += 1
        indices_by_ds_cls[(ds_str, c_int)].append(idx)

    unique_datasets = sorted(counts_by_ds_cls.keys())
    n_datasets = len(unique_datasets)

    dist_before: Dict[str, Dict[str, int]] = {
        ds: {
            ID_TO_HARMONIZED_CLASS.get(c, str(c)): cnt
            for c, cnt in sorted(counts_by_ds_cls[ds].items())
        }
        for ds in unique_datasets
    }

    if mode == "none":
        weights = np.ones(n_total, dtype=np.float64)
        resampled_idx = np.arange(n_total, dtype=np.int64)
    elif mode == "hierarchical_weights":
        weights = np.zeros(n_total, dtype=np.float64)
        for idx, (ds, cid) in enumerate(zip(datasets, class_ids)):
            ds_str = str(ds)
            c_int = int(cid)
            n_cls_in_ds = len(counts_by_ds_cls[ds_str])
            n_dc = counts_by_ds_cls[ds_str][c_int]
            weights[idx] = float(n_total) / float(n_datasets * n_cls_in_ds * n_dc)
        resampled_idx = np.arange(n_total, dtype=np.int64)
    else:  # hierarchical_resample
        # Choose a common per-dataset quota divisible by each dataset's number of present classes
        max_ds_count = max(
            sum(counts_by_ds_cls[ds].values()) for ds in unique_datasets
        )
        lcm_classes = 1
        for ds in unique_datasets:
            k_c = len(counts_by_ds_cls[ds])
            lcm_classes = (lcm_classes * k_c) // math.gcd(lcm_classes, k_c)
        per_ds_quota = max(
            lcm_classes,
            int(math.ceil(max_ds_count / float(lcm_classes))) * lcm_classes,
        )

        rng = random.Random(int(seed))
        sampled: List[int] = []
        for ds in unique_datasets:
            present_c = sorted(counts_by_ds_cls[ds].keys())
            per_cell_quota = per_ds_quota // len(present_c)
            for c_int in present_c:
                cell_pool = sorted(indices_by_ds_cls[(ds, c_int)])
                # Take full copies first, then sample remainder deterministically
                full_reps, rem = divmod(per_cell_quota, len(cell_pool))
                cell_chosen = cell_pool * full_reps + rng.sample(cell_pool, rem)
                sampled.extend(cell_chosen)

        resampled_idx = np.asarray(sampled, dtype=np.int64)
        weights = np.ones(len(resampled_idx), dtype=np.float64)

    eff_ds_weight: Dict[str, float] = {ds: 0.0 for ds in unique_datasets}
    eff_ds_cls_weight: Dict[str, Dict[str, float]] = {
        ds: {
            ID_TO_HARMONIZED_CLASS.get(c, str(c)): 0.0
            for c in sorted(counts_by_ds_cls[ds].keys())
        }
        for ds in unique_datasets
    }
    dist_after: Dict[str, Dict[str, int]] = {
        ds: {
            ID_TO_HARMONIZED_CLASS.get(c, str(c)): 0
            for c in sorted(counts_by_ds_cls[ds].keys())
        }
        for ds in unique_datasets
    }

    for pos, orig_idx in enumerate(resampled_idx):
        ds_str = str(datasets[orig_idx])
        c_int = int(class_ids[orig_idx])
        c_name = ID_TO_HARMONIZED_CLASS.get(c_int, str(c_int))
        w_val = float(weights[pos]) if len(weights) == len(resampled_idx) else float(weights[orig_idx])
        eff_ds_weight[ds_str] += w_val
        eff_ds_cls_weight[ds_str][c_name] += w_val
        dist_after[ds_str][c_name] += 1

    return HierarchicalBalanceResult(
        sample_weights=weights,
        resampled_indices=resampled_idx,
        distribution_before=dist_before,
        effective_weight_by_dataset=eff_ds_weight,
        effective_weight_by_dataset_class=eff_ds_cls_weight,
        resampled_distribution_after=dist_after,
        mode=mode,
    )


# =============================================================================
# 6. FROZEN-ENCODER 5-CLASS HEAD FOR TABNET DA & DG + METRIC ACCOUNTING
# =============================================================================


def extract_tabnet_latent_features(
    embedder: Any,
    encoder: Any,
    X_scaled: np.ndarray,
    device: Any,
) -> np.ndarray:
    """
    Pass `X_scaled` through a frozen `TabNet` `embedder` + `encoder` in `.eval()`
    mode and return the aggregated latent representation `sum(steps_out)` (`(N, n_d)`).
    """
    import torch

    embedder.eval()
    encoder.eval()
    with torch.no_grad():
        x_t = torch.tensor(np.asarray(X_scaled, dtype=np.float32), dtype=torch.float32, device=device)
        embedded = embedder(x_t)
        steps_out, _ = encoder(embedded)
        latent = torch.sum(torch.stack(steps_out, dim=0), dim=0)
        return latent.detach().cpu().numpy().astype(np.float32)


def _snapshot_torch_module(module: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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


def _max_abs_diff(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> float:
    max_d = 0.0
    for k, v_a in a.items():
        if k in b:
            d = float(np.max(np.abs(b[k] - v_a)))
            if d > max_d:
                max_d = d
    return max_d


@dataclass
class FiveClassHeadCheckpoint:
    """
    Stores a trained 5-class classification head (`nn.Linear(latent_dim, 5)`)
    along with the boolean mask of classes observed during training (`observed_class_ids`),
    guaranteeing the 5-output contract (`Normal`, `Ball`, `Inner`, `Outer`, `Inner & Outer`)
    and preventing random predictions on unobserved source classes in single-source DG.
    """

    latent_dim: int
    head_state_dict: Dict[str, Any]
    observed_class_ids: Tuple[int, ...]
    source_config_id: str
    best_val_loss: float
    selected_epoch: int
    encoder_param_max_abs_diff: float
    encoder_buffer_max_abs_diff: float
    head_param_max_abs_diff: float


def train_five_class_head_with_frozen_tabnet_encoder(
    pretrained_tabnet: Any,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    sample_weights: Optional[np.ndarray] = None,
    init_head_checkpoint: Optional[FiveClassHeadCheckpoint] = None,
    source_config_id: str = "S4",
    max_epochs: int = 40,
    patience: int = 10,
    batch_size: int = 64,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    mask_unobserved_classes: bool = True,
) -> FiveClassHeadCheckpoint:
    """
    Train or adapt a fixed 5-output classification head (`nn.Linear(n_d, 5)`) on top
    of a completely frozen `TabNet` encoder (`embedder` + `encoder`):
      - Used for **DG** (when `init_head_checkpoint=None`, training on source_train
        and early-stopping on `source_val`) and for **DA** (when `init_head_checkpoint`
        is the source-trained head, copying and initializing from the source head
        weights and adapting on target `adapt_train` with early stopping on `adapt_val`).
      - Freezes all parameters and GhostBatchNorm buffers of `embedder` and `encoder`
        (`requires_grad=False`, `.eval()`) and asserts `encoder_param_max_abs_diff == 0.0`
        and `encoder_buffer_max_abs_diff == 0.0`.
      - Asserts `head_param_max_abs_diff > 0.0` (head weights are genuinely updated).
      - Preserves all 5 outputs (`output_dim == 5`). For classes absent from Both
        `y_train` and `init_head_checkpoint.observed_class_ids`, sets a structural
        logit mask (`-1e9`) during evaluation so unlearned classes are never predicted
        by random weight noise.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    emb_p_before, emb_b_before = _snapshot_torch_module(embedder)
    enc_p_before, enc_b_before = _snapshot_torch_module(encoder)

    # Probe latent dimension
    with torch.no_grad():
        probe_x = torch.tensor(X_train_scaled[:2], dtype=torch.float32, device=device)
        probe_steps, _ = encoder(embedder(probe_x))
        latent_dim = int(probe_steps[0].shape[1])

    head = nn.Linear(latent_dim, NUM_HARMONIZED_CLASSES).to(device)
    if init_head_checkpoint is not None:
        # Copy of source head initialized with source head weights (Section 7.3 & 7.4)
        head.load_state_dict(copy.deepcopy(init_head_checkpoint.head_state_dict))
        prior_observed = set(init_head_checkpoint.observed_class_ids)
    else:
        prior_observed = set()

    for p in head.parameters():
        p.requires_grad = True
    head_p_before, _ = _snapshot_torch_module(head)

    train_observed = {int(c) for c in np.unique(y_train)}
    if any(c < 0 or c >= NUM_HARMONIZED_CLASSES for c in train_observed):
        raise ValueError(
            f"y_train contains class IDs outside 0..{NUM_HARMONIZED_CLASSES - 1}: {sorted(train_observed)}"
        )
    combined_observed = tuple(sorted(prior_observed | train_observed))

    if sample_weights is not None:
        w_arr = np.asarray(sample_weights, dtype=np.float32)
        if len(w_arr) != len(X_train_scaled):
            raise ValueError("sample_weights length must match X_train_scaled.")
    else:
        w_arr = np.ones(len(X_train_scaled), dtype=np.float32)

    ds_train = TensorDataset(
        torch.tensor(np.asarray(X_train_scaled, dtype=np.float32), dtype=torch.float32),
        torch.tensor(np.asarray(y_train, dtype=np.int64), dtype=torch.long),
        torch.tensor(w_arr, dtype=torch.float32),
    )
    eff_b = min(int(batch_size), max(2, len(ds_train)))
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    loader = DataLoader(
        ds_train,
        batch_size=eff_b,
        shuffle=True,
        drop_last=False,
        generator=gen,
    )

    optimizer = optim.AdamW(
        head.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    criterion_none = nn.CrossEntropyLoss(reduction="none")

    X_val_t = torch.tensor(np.asarray(X_val_scaled, dtype=np.float32), dtype=torch.float32, device=device)
    y_val_t = torch.tensor(np.asarray(y_val, dtype=np.int64), dtype=torch.long, device=device)

    best_val_loss = float("inf")
    best_epoch = 1
    best_head_state = copy.deepcopy(head.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, int(max_epochs) + 1):
        embedder.eval()
        encoder.eval()
        head.train()
        for bx, by, bw in loader:
            bx = bx.to(device)
            by = by.to(device)
            bw = bw.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                emb_x = embedder(bx)
                steps_out, _ = encoder(emb_x)
                latent = torch.sum(torch.stack(steps_out, dim=0), dim=0)
            logits = head(latent)
            if mask_unobserved_classes and len(combined_observed) < NUM_HARMONIZED_CLASSES:
                mask = torch.zeros_like(logits)
                for c_idx in range(NUM_HARMONIZED_CLASSES):
                    if c_idx not in combined_observed:
                        mask[:, c_idx] = -1e9
                logits = logits + mask
            per_sample_loss = criterion_none(logits, by)
            loss = torch.sum(per_sample_loss * bw) / torch.clamp(torch.sum(bw), min=1e-8)
            loss.backward()
            optimizer.step()

        # Validation step for early stopping (strictly on source_val or adapt_val)
        embedder.eval()
        encoder.eval()
        head.eval()
        with torch.no_grad():
            if len(X_val_t) > 0:
                val_steps, _ = encoder(embedder(X_val_t))
                val_latent = torch.sum(torch.stack(val_steps, dim=0), dim=0)
                val_logits = head(val_latent)
                if mask_unobserved_classes and len(combined_observed) < NUM_HARMONIZED_CLASSES:
                    mask = torch.zeros_like(val_logits)
                    for c_idx in range(NUM_HARMONIZED_CLASSES):
                        if c_idx not in combined_observed:
                            mask[:, c_idx] = -1e9
                    val_logits = val_logits + mask
                val_loss = float(torch.mean(criterion_none(val_logits, y_val_t)).item())
            else:
                val_loss = float(loss.item())

        if val_loss < best_val_loss - 1e-6 or epoch == 1:
            best_val_loss = val_loss
            best_epoch = epoch
            best_head_state = copy.deepcopy(head.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= int(patience):
                break

    head.load_state_dict(best_head_state)

    emb_p_after, emb_b_after = _snapshot_torch_module(embedder)
    enc_p_after, enc_b_after = _snapshot_torch_module(encoder)
    head_p_after, _ = _snapshot_torch_module(head)

    enc_param_diff = max(
        _max_abs_diff(emb_p_before, emb_p_after),
        _max_abs_diff(enc_p_before, enc_p_after),
    )
    enc_buf_diff = max(
        _max_abs_diff(emb_b_before, emb_b_after),
        _max_abs_diff(enc_b_before, enc_b_after),
    )
    head_param_diff = _max_abs_diff(head_p_before, head_p_after)

    if enc_param_diff > 0.0 or enc_buf_diff > 0.0:
        raise AssertionError(
            f"TabNet encoder freeze invariant violated! "
            f"param_diff={enc_param_diff}, buffer_diff={enc_buf_diff}"
        )
    if head_param_diff <= 0.0:
        raise AssertionError(
            "5-class classification head parameters were not updated during training!"
        )

    return FiveClassHeadCheckpoint(
        latent_dim=latent_dim,
        head_state_dict=copy.deepcopy(head.state_dict()),
        observed_class_ids=combined_observed,
        source_config_id=source_config_id,
        best_val_loss=float(best_val_loss),
        selected_epoch=int(best_epoch),
        encoder_param_max_abs_diff=float(enc_param_diff),
        encoder_buffer_max_abs_diff=float(enc_buf_diff),
        head_param_max_abs_diff=float(head_param_diff),
    )


def predict_with_five_class_head(
    pretrained_tabnet: Any,
    head_checkpoint: FiveClassHeadCheckpoint,
    X_test_scaled: np.ndarray,
    mask_unobserved_classes: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate `X_test_scaled` using the frozen TabNet encoder and the 5-output
    classification head (`head_checkpoint`).

    Always returns:
      - `y_pred`: integer class predictions in `{0, 1, 2, 3, 4}` (`shape (N,)`)
      - `y_proba`: probability matrix across all 5 harmonized classes (`shape (N, 5)`)
    """
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_net = pretrained_tabnet.network.to(device)
    embedder = backbone_net.embedder
    encoder = backbone_net.encoder
    embedder.eval()
    encoder.eval()

    head = nn.Linear(head_checkpoint.latent_dim, NUM_HARMONIZED_CLASSES).to(device)
    head.load_state_dict(copy.deepcopy(head_checkpoint.head_state_dict))
    head.eval()

    if head.out_features != NUM_HARMONIZED_CLASSES:
        raise AssertionError(
            f"Classification head must have {NUM_HARMONIZED_CLASSES} outputs, "
            f"got {head.out_features}."
        )

    with torch.no_grad():
        x_t = torch.tensor(np.asarray(X_test_scaled, dtype=np.float32), dtype=torch.float32, device=device)
        steps_out, _ = encoder(embedder(x_t))
        latent = torch.sum(torch.stack(steps_out, dim=0), dim=0)
        logits = head(latent)
        if mask_unobserved_classes and len(head_checkpoint.observed_class_ids) < NUM_HARMONIZED_CLASSES:
            for c_idx in range(NUM_HARMONIZED_CLASSES):
                if c_idx not in head_checkpoint.observed_class_ids:
                    logits[:, c_idx] = -1e9
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
        preds = np.argmax(probs, axis=1).astype(np.int64)

    if probs.shape != (len(X_test_scaled), NUM_HARMONIZED_CLASSES):
        raise AssertionError(
            f"Expected probability output of shape ({len(X_test_scaled)}, {NUM_HARMONIZED_CLASSES}), "
            f"got {probs.shape}."
        )
    return preds, probs


def evaluate_five_class_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Evaluate 5-class predictions (`0..4`) against ground-truth target labels `y_true`
    under the strict rules of Section 3 & Section 11:
      - The output space of the head is fixed to 5 classes (`0..4`).
      - Any prediction of a class that does not exist in the target domain (`yp not in Y_target`)
        is strictly counted as a classification error:
          * It decreases `Accuracy` (`yp != yt`).
          * It is a False Negative for the sample's true class `yt`, decreasing both
            `Recall_{yt}` (`Balanced_Accuracy`) and `F1_{yt}` (`Macro_F1_Target_Present`).
          * In `Macro_F1_All_Active` (`labels = sorted(set(y_true) | set(y_pred))`), a
            false positive on a non-existent target class `c'` additionally introduces
            `F1_{c'} = 0.0` into the macro average!
      - Both `macro_f1` (computed over `sorted(set(y_true))` where every true target
        sample is evaluated and out-of-target predictions penalize the true class FN,
        plus any unpredicted true target class gets `F1 = 0.0`) and
        `macro_f1_including_phantom_fp_classes` (`sorted(set(y_true) | set(y_pred))`)
        are reported transparently along with `out_of_target_prediction_count`.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    if len(y_t) != len(y_p) or len(y_t) == 0:
        raise ValueError("y_true and y_pred must have the same non-zero length.")

    target_present_classes = sorted(int(c) for c in np.unique(y_t))
    active_union_classes = sorted(int(c) for c in (set(y_t.tolist()) | set(y_p.tolist())))

    out_of_target_mask = ~np.isin(y_p, target_present_classes)
    out_of_target_count = int(np.sum(out_of_target_mask))
    out_of_target_rate = float(out_of_target_count) / float(len(y_t))

    acc = float(accuracy_score(y_t, y_p))
    # Balanced accuracy is macro recall across all true classes present in the target test set
    bal_acc = float(
        balanced_accuracy_score(y_t, y_p)
        if out_of_target_count == 0
        else np.mean(
            [
                float(np.sum((y_t == c) & (y_p == c))) / float(np.sum(y_t == c))
                for c in target_present_classes
            ]
        )
    )
    macro_f1_present = float(
        f1_score(y_t, y_p, labels=target_present_classes, average="macro", zero_division=0)
    )
    macro_f1_active_union = float(
        f1_score(y_t, y_p, labels=active_union_classes, average="macro", zero_division=0)
    )

    per_class_f1 = {
        ID_TO_HARMONIZED_CLASS[c]: float(
            f1_score(y_t, y_p, labels=[c], average="macro", zero_division=0)
        )
        for c in range(NUM_HARMONIZED_CLASSES)
    }

    roc_auc_val: Optional[float] = None
    roc_auc_valid = False
    if (
        y_proba is not None
        and y_proba.shape == (len(y_t), NUM_HARMONIZED_CLASSES)
        and len(target_present_classes) >= 2
    ):
        # Extract columns for present target classes and renormalize row probabilities
        sub_probs = np.asarray(y_proba[:, target_present_classes], dtype=np.float64)
        row_sums = np.sum(sub_probs, axis=1, keepdims=True)
        if np.all(np.isfinite(sub_probs)) and np.all(row_sums > 0):
            sub_probs_norm = sub_probs / row_sums
            y_t_contiguous = np.searchsorted(target_present_classes, y_t)
            try:
                if len(target_present_classes) == 2:
                    roc_auc_val = float(roc_auc_score(y_t_contiguous, sub_probs_norm[:, 1]))
                else:
                    roc_auc_val = float(
                        roc_auc_score(
                            y_t_contiguous,
                            sub_probs_norm,
                            multi_class="ovr",
                            average="macro",
                        )
                    )
                roc_auc_valid = True
            except ValueError:
                roc_auc_val = None
                roc_auc_valid = False

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1_present,
        "macro_f1_including_phantom_classes": macro_f1_active_union,
        "roc_auc_ovr": roc_auc_val,
        "roc_auc_valid": roc_auc_valid,
        "target_present_classes": [ID_TO_HARMONIZED_CLASS[c] for c in target_present_classes],
        "predicted_classes": [ID_TO_HARMONIZED_CLASS[c] for c in sorted(set(y_p.tolist()))],
        "out_of_target_prediction_count": out_of_target_count,
        "out_of_target_prediction_rate": out_of_target_rate,
        "per_class_f1_all_5": per_class_f1,
    }


def allocate_non_overwriting_run_dir(
    output_root: str | Path,
    experiment_prefix: str = "run",
) -> Path:
    """
    Create a guaranteed-new timestamped output directory under `output_root`
    (`<output_root>/<experiment_prefix>_<YYYYMMDD_HHMMSS>[_NN]`), ensuring
    previous experimental results are never overwritten (Section 12 & Section 13.11).
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"{experiment_prefix}_{ts}"
    candidate = root / base_name
    suffix = 1
    while candidate.exists():
        candidate = root / f"{base_name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate
