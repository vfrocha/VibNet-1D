"""
 Harmonized 5-Class Vocabulary, Variable-Space Classification Heads, DA Head Expansion,
 Recording-Level Partitioning, Leak-Free Physical RMS/Feature Scaling, 126/15 Scale-Invariance
 Feature Catalog, Hierarchical Source Balancing, MFPT Isolation Remediation, and
 TabNet/Classical DA, DG & TL Protocol Engine (`src.vibnet.evaluation.da_dg_tl_protocol`).

Implements the revised methodological decisions for the VibNet-1D Domain Adaptation (DA),
Domain Generalization (DG), and Transfer Learning (TL) experimental matrix:
  1. Source configurations (`UOC` excluded):
     - `S1`: `["UORED"]` -> available classes: `("Normal", "Ball")` (2 outputs)
     - `S2`: `["HUST"]` (HUST Bearing, distinct from `HUST_Gearbox`) -> available classes:
       `("Normal", "Ball", "Inner", "Outer", "Inner & Outer")` (5 outputs)
     - `S3`: `["MFPT"]` -> available classes: `("Inner", "Outer")` (2 outputs)
     - `S4`: `["UORED", "HUST", "MFPT"]` -> available classes:
       `("Normal", "Ball", "Inner", "Outer", "Inner & Outer")` (5 outputs)
  2. Variable-output classification heads & canonical name-based DA head expansion:
     - Initial source heads do NOT impose 5 outputs on all configurations; each source head
       has `out_features == len(source_present_classes)` (`S1`: 2, `S2`: 5, `S3`: 2, `S4`: 5).
     - In DG: the encoder and source head remain strictly frozen with the exact output space
       learned on the sources (`no head expansion`). Any target sample whose true class is absent
       from the source head is necessarily misclassified (`0` correct predictions, never excluded).
     - In DA: the head is expanded for each target (`expand_head_for_da_target`) to cover the
       union of source classes and target adaptation classes (`source_classes ∪ target_classes`).
       Shared classes preserve their exact learned weight rows and bias values by canonical class
       name (never by assumed index position), while newly introduced target classes are
       initialized reproducibly using the experiment `seed`. Adaptation updates only the expanded
       head on the 20% target adaptation partition while keeping the TabNet encoder 100% frozen.
  3. Recording-level grouping (`recording_key = f"{dataset}::{condition}::{class_dir}::{sNNNNN}"`):
     - Zero recording overlap across training, validation, and test partitions.
     - In DA: 20% of recordings per class for adaptation and 80% for test. No mandatory target
       validation split inside the 20% adaptation partition (`CWRU_12k` `Class_0 / Normal` uses
       all 2 adaptation recordings for training and 6 for test; epochs and hyperparameters are
       selected strictly on the sources without consulting test data).
  4. Leak-free physical reference RMS (`detrended_rms`) and feature scaling:
     - Physical unit conversion (`1 g = 9.80665 m/s^2`) before geometric mean aggregation across
       active sources in DG: `RMS_global = exp(mean(log(RMS_ref_i_ms2)))`.
     - Explicit audit of physical comparability across `UORED`, `HUST`, and `MFPT`.
     - Explicit mathematical catalog of the 126 scale-invariant features and 15 scale-sensitive
       features among the 141 TabVibNet features (`get_141_feature_scale_invariance_catalog`).
     - In DA: `RMS_ref` estimated exclusively from `Normal` windows in the target's 20% adaptation
       partition and applied to both adaptation and test (`CWRU_48k` uses external calibration from
       the 2 `Normal` adaptation recordings of `CWRU_12k`, never from validation or test).
  5. Isolated MFPT preparation & validation (`prepare_isolated_mfpt_detrended_rms`):
     - Identifies the 6 recordings at `97,656 Hz` (`Load_270/Class_23/s00000..s00002` and
       `Load_270/Class_25/s00003..s00005`), preserves original files in backup, decimates from
       `97,656 Hz` to `48,828 Hz` via polyphase anti-aliasing filtering (`resample_poly(up=1, down=2)`),
       applies 1-second windowing (`48,828` samples/window -> `78` total 1-second windows),
       generates `data/processed_tl/detrended_rms/MFPT`, and updates manifests transactionally.
  6. Hierarchical source balancing (`compute_hierarchical_source_balance`) and classical
     baselines (`RandomForest` and `SVM` for DG, DA Target-only 20%, DA Source + Target 20%,
     and TL preservation).
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
import shutil
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import detrend as scipy_detrend, resample_poly
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.vibnet.evaluation.balancing import trim_single_remainder_sample
from src.vibnet.preprocessing.signal_norm import (
    EPSILON_RMS,
    compute_reference_rms,
    detrend_linear_window,
)

# =============================================================================
# 1. CANONICAL CONSTANTS, SOURCE CONFIGURATIONS & HARMONIZED VOCABULARY
# =============================================================================

STANDARD_GRAVITY_MS2: float = 9.80665

SOURCE_CONFIGS: Dict[str, Tuple[str, ...]] = {
    "S1": ("UORED",),
    "S2": ("HUST",),
    "S3": ("MFPT",),
    "S4": ("UORED", "HUST", "MFPT"),
}

DA_DG_TARGETS: Tuple[str, ...] = ("CWRU_12k", "CWRU_48k", "PU")
TL_TARGETS: Tuple[str, ...] = ("CWRU_12k", "CWRU_48k", "PU", "HUST_Gearbox", "UOEMD")
CANONICAL_SEEDS: Tuple[int, ...] = (42, 123, 2026)

DATASET_PHYSICAL_UNITS: Dict[str, str] = {
    "UORED": "m/s^2",
    "HUST": "m/s^2",
    "MFPT": "g",
    "CWRU_12k": "g",
    "CWRU_48k": "g",
    "PU": "V",
    "HUST_Gearbox": "m/s^2",
    "UOEMD": "m/s^2",
}

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

# Canonical classes available per source configuration (Section 3):
#   S1 (UORED): Normal, Ball (2 classes)
#   S2 (HUST Bearing): Normal, Ball, Inner, Outer, Inner & Outer (5 classes)
#   S3 (MFPT): Inner, Outer (2 classes)
#   S4 (Combined): Normal, Ball, Inner, Outer, Inner & Outer (5 classes)
SOURCE_CONFIG_AVAILABLE_CLASSES: Dict[str, Tuple[str, ...]] = {
    "S1": ("Normal", "Ball"),
    "S2": ("Normal", "Ball", "Inner", "Outer", "Inner & Outer"),
    "S3": ("Inner", "Outer"),
    "S4": ("Normal", "Ball", "Inner", "Outer", "Inner & Outer"),
}

# Verified mapping from on-disk folder names (`Class_*`) to the 5 harmonized classes.
# Severities of the same fault are grouped into the same canonical fault class;
# `Inner & Outer` remains an independent compound class.
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
        "Class_23": None,  # Per Section 3: MFPT supervised head classes are Inner & Outer; Class_23 is used for RMS_ref
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
    to perform a recording-grouped split. Never falls back silently to window-level splitting.
    """


class IncomparableSourceRMSReferenceError(ValueError):
    """
    Raised when attempting an unapproved direct aggregation of source RMS references
    across datasets with incompatible physical units or amplitude scales.
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
    recording_sha256: str = ""


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


def canonicalize_class_sequence(class_names: Iterable[str]) -> Tuple[str, ...]:
    """
    Order any subset of harmonized class names according to their canonical order in
    `HARMONIZED_CLASSES` (`Normal`, `Ball`, `Inner`, `Outer`, `Inner & Outer`),
    removing duplicates and validating every class name.
    """
    unique_names = set(str(c) for c in class_names)
    unknown = unique_names - set(HARMONIZED_CLASSES)
    if unknown:
        raise ValueError(
            f"Unknown class names outside harmonized vocabulary {HARMONIZED_CLASSES}: "
            f"{sorted(unknown)}"
        )
    return tuple(c for c in HARMONIZED_CLASSES if c in unique_names)


def map_to_harmonized_class(
    dataset: str,
    raw_class: str,
    allow_mfpt_normal: bool = False,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Map a dataset-native folder label (`raw_class`) to `(harmonized_name, harmonized_id)`
    in the canonical 5-class vocabulary (`0: Normal`, `1: Ball`, `2: Inner`, `3: Outer`,
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


def get_source_config_classes(
    source_config_id: str,
    allow_mfpt_normal: bool = False,
) -> Tuple[str, ...]:
    """
    Return the exact canonical tuple of classes available for `source_config_id` (`S1..S4`):
      - `S1`: `("Normal", "Ball")` (2 classes)
      - `S2`: `("Normal", "Ball", "Inner", "Outer", "Inner & Outer")` (5 classes)
      - `S3`: `("Inner", "Outer")` (2 classes, unless `allow_mfpt_normal=True`)
      - `S4`: `("Normal", "Ball", "Inner", "Outer", "Inner & Outer")` (5 classes)
    """
    if source_config_id not in SOURCE_CONFIGS:
        raise KeyError(
            f"Unknown source configuration '{source_config_id}'. "
            f"Valid configurations: {sorted(SOURCE_CONFIGS.keys())}"
        )
    if not allow_mfpt_normal and source_config_id in SOURCE_CONFIG_AVAILABLE_CLASSES:
        return SOURCE_CONFIG_AVAILABLE_CLASSES[source_config_id]

    sources = SOURCE_CONFIGS[source_config_id]
    present_names: set[str] = set()
    for s_name in sources:
        for raw_c in DATASET_TO_HARMONIZED_CLASS_MAP[s_name]:
            harm, _ = map_to_harmonized_class(
                s_name, raw_c, allow_mfpt_normal=allow_mfpt_normal
            )
            if harm is not None:
                present_names.add(harm)
    return canonicalize_class_sequence(present_names)


def diagnose_source_config_class_coverage(
    source_config_id: str,
    allow_mfpt_normal: bool = False,
) -> Dict[str, Any]:
    """
    Analyze which harmonized classes are available in source configuration `S1..S4`
    and report the exact variable head output dimension (`len(present_class_names)`).
    """
    if source_config_id not in SOURCE_CONFIGS:
        raise KeyError(
            f"Unknown source configuration '{source_config_id}'. "
            f"Valid configurations: {sorted(SOURCE_CONFIGS.keys())}"
        )
    sources = SOURCE_CONFIGS[source_config_id]
    by_source: Dict[str, List[str]] = {}
    for s_name in sources:
        s_classes = [
            c
            for c in HARMONIZED_CLASSES
            if any(
                map_to_harmonized_class(
                    s_name, raw_c, allow_mfpt_normal=allow_mfpt_normal
                )[0]
                == c
                for raw_c in DATASET_TO_HARMONIZED_CLASS_MAP[s_name]
            )
        ]
        by_source[s_name] = s_classes

    present_names = list(
        get_source_config_classes(source_config_id, allow_mfpt_normal=allow_mfpt_normal)
    )
    present_ids = [HARMONIZED_CLASS_TO_ID[c] for c in present_names]
    absent_names = [c for c in HARMONIZED_CLASSES if c not in present_names]
    absent_ids = [HARMONIZED_CLASS_TO_ID[c] for c in absent_names]

    head_index_to_class = {idx: name for idx, name in enumerate(present_names)}
    class_to_head_index = {name: idx for idx, name in enumerate(present_names)}

    return {
        "source_config_id": source_config_id,
        "sources": list(sources),
        "classes_by_source": by_source,
        "present_class_ids": present_ids,
        "present_class_names": present_names,
        "absent_class_ids": absent_ids,
        "absent_class_names": absent_names,
        "full_5_class_coverage": len(present_names) == NUM_HARMONIZED_CLASSES,
        "head_output_dim": len(present_names),
        "head_index_to_class": head_index_to_class,
        "class_to_head_index": class_to_head_index,
    }


def load_dataset_window_records(
    dataset_name: str,
    data_root: str | Path,
    load_waveforms: bool = False,
    allow_mfpt_normal: bool = False,
    deduplicate_identical_recordings_by_sha256: bool = True,
) -> Tuple[List[WindowRecord], Optional[List[np.ndarray]]]:
    """
    Scan `<data_root>/<dataset_name>/<condition>/<raw_class>/sNNNNN_wNN.npy` in
    deterministic sorted order and construct `WindowRecord` items (and optional
    1D waveforms). Ignores any nested depth != 3 anomalies if present, raising
    an explicit error if a file violates the `<condition>/<class>/<file.npy>` contract.

    When `deduplicate_identical_recordings_by_sha256=True` (default):
      - Computes the SHA-256 digest of every original recording (across its ordered
        window `.npy` byte payloads).
      - If two or more recordings in `(dataset, condition, raw_class)` have identical
        SHA-256 digests (e.g., in `PU` `Class_27`: `C2: s00194 == s00198`,
        `C4: s00180 == s00204 == s00224`, `C4: s00196 == s00220`), all members of
        the duplicate group receive the same canonical `recording_key`
        (`f"{dataset}::{condition}::{raw_class}::sha256grp_{ '_'.join(sorted_prefixes) }"`)
        so that the entire group is treated as a single independent partitioning unit
        and never split across adaptation/training and validation/test, while
        preserving every individual `recording_prefix` (`sNNNNN`) and every `.npy` file on disk.
    """
    ds_dir = Path(data_root) / dataset_name
    if not ds_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")

    raw_items: List[Tuple[Path, str, str, str, str, int, Optional[str], Optional[int], str]] = []
    waveforms: Optional[List[np.ndarray]] = [] if load_waveforms else None
    files_by_recording: Dict[Tuple[str, str, str], List[Tuple[int, Path]]] = defaultdict(list)

    for fpath in sorted(ds_dir.rglob("*.npy")):
        rel_parts = fpath.relative_to(ds_dir).parts
        if len(rel_parts) != 3:
            raise ValueError(
                f"Invalid directory nesting depth ({len(rel_parts)} != 3) for '{fpath}'."
            )
        cond_name, raw_cls, fname = rel_parts
        rec_prefix, w_idx = parse_window_filename(fname)
        harm_name, harm_id = map_to_harmonized_class(
            dataset_name, raw_cls, allow_mfpt_normal=allow_mfpt_normal
        )
        sample_id = f"{dataset_name}/{cond_name}/{raw_cls}/{fname}"
        raw_items.append(
            (fpath, cond_name, raw_cls, fname, rec_prefix, w_idx, harm_name, harm_id, sample_id)
        )
        files_by_recording[(cond_name, raw_cls, rec_prefix)].append((w_idx, fpath))
        if waveforms is not None:
            arr = np.load(fpath, allow_pickle=False)
            waveforms.append(np.asarray(arr, dtype=np.float64).reshape(-1))

    if not raw_items:
        raise ValueError(f"No .npy files found under {ds_dir}")

    recording_sha_map: Dict[Tuple[str, str, str], str] = {}
    sha_to_prefixes: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

    if deduplicate_identical_recordings_by_sha256:
        for (cond_name, raw_cls, rec_prefix), w_files in sorted(files_by_recording.items()):
            hasher = hashlib.sha256()
            for w_idx, fp in sorted(w_files, key=lambda x: x[0]):
                hasher.update(f"w{w_idx:04d}:".encode("utf-8"))
                hasher.update(fp.read_bytes())
            digest = hasher.hexdigest()
            recording_sha_map[(cond_name, raw_cls, rec_prefix)] = digest
            sha_to_prefixes[(cond_name, raw_cls, digest)].append(rec_prefix)

    records: List[WindowRecord] = []
    for (
        _fpath,
        cond_name,
        raw_cls,
        _fname,
        rec_prefix,
        w_idx,
        harm_name,
        harm_id,
        sample_id,
    ) in raw_items:
        rec_sha = recording_sha_map.get((cond_name, raw_cls, rec_prefix), "")
        dup_prefixes = (
            sorted(set(sha_to_prefixes.get((cond_name, raw_cls, rec_sha), [rec_prefix])))
            if rec_sha
            else [rec_prefix]
        )
        if len(dup_prefixes) > 1:
            grp_suffix = "sha256grp_" + "_".join(dup_prefixes)
            rec_key = build_recording_key(dataset_name, cond_name, raw_cls, grp_suffix)
        else:
            rec_key = build_recording_key(dataset_name, cond_name, raw_cls, rec_prefix)

        records.append(
            WindowRecord(
                sample_id=sample_id,
                dataset=dataset_name,
                condition=cond_name,
                raw_class=raw_cls,
                harmonized_class=harm_name,
                harmonized_class_id=harm_id,
                recording_prefix=rec_prefix,
                recording_key=rec_key,
                window_index=w_idx,
                recording_sha256=rec_sha,
            )
        )

    return records, waveforms


# =============================================================================
# 3. RECORDING-GROUPED SPLITTING & MANIFEST EXPORT (DA & DG)
# =============================================================================


@dataclass(frozen=True)
class RecordingEntryManifest:
    """Single recording row for reproducible partition manifests (Section 8)."""

    dataset: str
    recording_key: str
    recording_prefix: str
    raw_class: str
    harmonized_class: str
    condition: str
    partition: str
    seed: int
    window_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "recording_key": self.recording_key,
            "recording_prefix": self.recording_prefix,
            "raw_class": self.raw_class,
            "harmonized_class": self.harmonized_class,
            "condition": self.condition,
            "partition": self.partition,
            "seed": self.seed,
            "window_count": self.window_count,
        }


@dataclass(frozen=True)
class RecordingSplitManifest:
    """
    Serializable, deterministic manifest of recording-grouped partitions
    (`adapt`, `test` [or `adapt_train`, `adapt_val`, `test`] for DA;
    `source_train`, `source_val` for DG).
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
    recording_entries: Tuple[RecordingEntryManifest, ...] = ()

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
            "recording_entries": [e.to_dict() for e in self.recording_entries],
        }


def assert_zero_recording_overlap(
    records: Sequence[WindowRecord],
    partition_indices: Mapping[str, Sequence[int]],
) -> None:
    """
    Verify that no `recording_key` (or `sample_id`) appears in more than one
    mutually exclusive partition.
    Note: if `partition_indices` contains both `"adapt"` and `"adapt_train"`/`"adapt_val"`,
    `"adapt"` is checked against `"test"`, while `"adapt_train"`, `"adapt_val"`, and `"test"`
    are checked pairwise.
    """
    all_keys = set(partition_indices.keys())
    composite_keys: set[str] = set()
    if "adapt" in all_keys and ("adapt_train" in all_keys or "adapt_val" in all_keys):
        # Verify that adapt is the exact disjoint union of adapt_train and adapt_val
        adapt_set = set(partition_indices["adapt"])
        tr_set = set(partition_indices.get("adapt_train", ()))
        val_set = set(partition_indices.get("adapt_val", ()))
        if tr_set & val_set:
            raise ValueError("Index overlap between adapt_train and adapt_val!")
        if adapt_set != (tr_set | val_set):
            raise ValueError("Partition 'adapt' must equal 'adapt_train' union 'adapt_val'!")
        composite_keys.add("adapt")

    if "pretrain_train" in all_keys and (
        "source_train" in all_keys or "pretrain_only_train" in all_keys
    ):
        # Verify that pretrain_train is the exact disjoint union of source_train and pretrain_only_train
        pre_set = set(partition_indices["pretrain_train"])
        src_tr_set = set(partition_indices.get("source_train", ()))
        un_tr_set = set(partition_indices.get("pretrain_only_train", ()))
        if src_tr_set & un_tr_set:
            raise ValueError("Index overlap between source_train and pretrain_only_train!")
        if pre_set != (src_tr_set | un_tr_set):
            raise ValueError(
                "Partition 'pretrain_train' must equal 'source_train' union 'pretrain_only_train'!"
            )
        composite_keys.add("pretrain_train")

    part_names = sorted(k for k in all_keys if k not in composite_keys)

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
    adapt_val_fraction: float = 0.0,
    min_adapt_train_recordings_per_class: int = 1,
    min_adapt_val_recordings_per_class: int = 0,
    min_test_recordings_per_class: int = 1,
) -> Tuple[Dict[str, Tuple[int, ...]], RecordingSplitManifest]:
    """
    Partition target dataset windows by original recording (`recording_key`) for
    Domain Adaptation (DA) per Section 8.1:
      - For each present harmonized class, reserve 20% of recordings for adaptation
        and 80% of recordings exclusively for test.
      - Per Section 8.1 ("Dentro da adaptação, não é obrigatório criar uma validação
        independente do destino quando isso resultar em partições inadequadas. A seleção
        de épocas e hiperparâmetros deve ser realizada nas fontes, sem consultar o teste.
        Exceção CWRU_12k: A classe Normal possui oito gravações: duas gravações para
        adaptação, seis gravações para teste. Utilizar integralmente as duas gravações
        normais na adaptação. Não dividir as duas gravações em adapt_train e adapt_val."),
        the default is `adapt_val_fraction = 0.0` and `min_adapt_val_recordings_per_class = 0`,
        putting 100% of the 20% adaptation recordings into `adapt` (and `adapt_train`,
        with `adapt_val = ()`).
      - If a caller explicitly requests an internal target validation split
        (`adapt_val_fraction > 0` and `min_adapt_val_recordings_per_class >= 1`),
        any class without enough recordings (or CWRU_12k Normal if 100% adapt is enforced)
        will raise `InsufficientRecordingsError` rather than splitting windows.
    """
    if not (0.0 < adapt_fraction < 1.0):
        raise ValueError("adapt_fraction must be in (0, 1).")
    if not (0.0 <= adapt_val_fraction < 1.0):
        raise ValueError("adapt_val_fraction must be in [0, 1).")

    supervised_indices = [
        idx for idx, r in enumerate(records) if r.harmonized_class_id is not None
    ]
    if not supervised_indices:
        raise ValueError("No records with valid harmonized classes found for DA split.")

    datasets_in_records = sorted({records[i].dataset for i in supervised_indices})
    ds_label = "|".join(datasets_in_records)

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
    recording_entries: List[RecordingEntryManifest] = []

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

        n_adapt = max(1, int(round(n_recs * adapt_fraction)))
        # Explicit Section 8.1 rule for CWRU_12k Normal: 8 recordings -> 2 adaptation, 6 test
        if ds_label == "CWRU_12k" and cls_name == "Normal" and n_recs == 8:
            n_adapt = 2

        min_adapt = (
            min_adapt_train_recordings_per_class + min_adapt_val_recordings_per_class
        )
        if n_adapt < min_adapt:
            raise InsufficientRecordingsError(
                f"Dataset '{ds_label}' class '{cls_name}' has {n_recs} recordings; "
                f"adaptation fraction {adapt_fraction:.2f} yields {n_adapt} recordings "
                f"(< {min_adapt} required)."
            )
        if (n_recs - n_adapt) < min_test_recordings_per_class:
            raise InsufficientRecordingsError(
                f"Dataset '{ds_label}' class '{cls_name}' leaves fewer than "
                f"{min_test_recordings_per_class} test recordings."
            )

        cls_seed_token = f"DA::{ds_label}::{cls_name}::{int(seed)}"
        cls_seed = int(
            hashlib.sha256(cls_seed_token.encode("utf-8")).hexdigest()[:8], 16
        )
        rng = random.Random(cls_seed)
        shuffled_recs = list(unique_recs)
        rng.shuffle(shuffled_recs)

        adapt_recs = shuffled_recs[:n_adapt]
        test_recs = shuffled_recs[n_adapt:]

        # Section 8.1: For CWRU_12k Normal (2 adaptation recordings), never divide into adapt_train/adapt_val
        is_cwru12k_normal_exception = (
            ds_label == "CWRU_12k" and cls_name == "Normal" and len(adapt_recs) == 2
        )
        if (
            adapt_val_fraction <= 0.0
            and min_adapt_val_recordings_per_class == 0
        ) or is_cwru12k_normal_exception:
            if is_cwru12k_normal_exception and min_adapt_val_recordings_per_class > 0:
                raise InsufficientRecordingsError(
                    "CWRU_12k Normal has 2 adaptation recordings and cannot be split into "
                    "adapt_train and adapt_val per Section 8.1."
                )
            n_adapt_val = 0
        else:
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

        for r_key in adapt_train_recs:
            w_idxs = rec_map[r_key]
            r0 = records[w_idxs[0]]
            member_prefixes = sorted({records[i].recording_prefix for i in w_idxs})
            recording_entries.append(
                RecordingEntryManifest(
                    dataset=r0.dataset,
                    recording_key=r_key,
                    recording_prefix="+".join(member_prefixes),
                    raw_class=r0.raw_class,
                    harmonized_class=cls_name,
                    condition=r0.condition,
                    partition="adapt" if n_adapt_val == 0 else "adapt_train",
                    seed=int(seed),
                    window_count=len(w_idxs),
                )
            )
        for r_key in adapt_val_recs:
            w_idxs = rec_map[r_key]
            r0 = records[w_idxs[0]]
            member_prefixes = sorted({records[i].recording_prefix for i in w_idxs})
            recording_entries.append(
                RecordingEntryManifest(
                    dataset=r0.dataset,
                    recording_key=r_key,
                    recording_prefix="+".join(member_prefixes),
                    raw_class=r0.raw_class,
                    harmonized_class=cls_name,
                    condition=r0.condition,
                    partition="adapt_val",
                    seed=int(seed),
                    window_count=len(w_idxs),
                )
            )
        for r_key in test_recs_sorted:
            w_idxs = rec_map[r_key]
            r0 = records[w_idxs[0]]
            member_prefixes = sorted({records[i].recording_prefix for i in w_idxs})
            recording_entries.append(
                RecordingEntryManifest(
                    dataset=r0.dataset,
                    recording_key=r_key,
                    recording_prefix="+".join(member_prefixes),
                    raw_class=r0.raw_class,
                    harmonized_class=cls_name,
                    condition=r0.condition,
                    partition="test",
                    seed=int(seed),
                    window_count=len(w_idxs),
                )
            )

        rec_counts_by_cls[cls_name] = {
            "total": n_recs,
            "adapt": len(adapt_train_recs) + len(adapt_val_recs),
            "adapt_train": len(adapt_train_recs),
            "adapt_val": len(adapt_val_recs),
            "adapt_total": len(adapt_train_recs) + len(adapt_val_recs),
            "test": len(test_recs_sorted),
        }
        win_counts_by_cls[cls_name] = {
            "total": len(cls_tr_win) + len(cls_val_win) + len(cls_te_win),
            "adapt": len(cls_tr_win) + len(cls_val_win),
            "adapt_train": len(cls_tr_win),
            "adapt_val": len(cls_val_win),
            "adapt_total": len(cls_tr_win) + len(cls_val_win),
            "test": len(cls_te_win),
        }

    adapt_all_indices = sorted(adapt_train_indices + adapt_val_indices)
    partitions: Dict[str, Tuple[int, ...]] = {
        "adapt": tuple(adapt_all_indices),
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

    total_recs = len(part_recs["adapt"]) + len(part_recs["test"])
    total_wins = len(part_samples["adapt"]) + len(part_samples["test"])

    rec_props = {
        "adapt": (len(part_recs["adapt"]) / float(total_recs) if total_recs > 0 else 0.0),
        "adapt_train": (len(part_recs["adapt_train"]) / float(total_recs) if total_recs > 0 else 0.0),
        "adapt_val": (len(part_recs["adapt_val"]) / float(total_recs) if total_recs > 0 else 0.0),
        "adapt_total": (len(part_recs["adapt"]) / float(total_recs) if total_recs > 0 else 0.0),
        "test": (len(part_recs["test"]) / float(total_recs) if total_recs > 0 else 0.0),
    }
    win_props = {
        "adapt": (len(part_samples["adapt"]) / float(total_wins) if total_wins > 0 else 0.0),
        "adapt_train": (len(part_samples["adapt_train"]) / float(total_wins) if total_wins > 0 else 0.0),
        "adapt_val": (len(part_samples["adapt_val"]) / float(total_wins) if total_wins > 0 else 0.0),
        "adapt_total": (len(part_samples["adapt"]) / float(total_wins) if total_wins > 0 else 0.0),
        "test": (len(part_samples["test"]) / float(total_wins) if total_wins > 0 else 0.0),
    }

    sig_payload = json.dumps(
        {
            "protocol": "DA",
            "dataset": ds_label,
            "seed": int(seed),
            "recordings": {
                "adapt_train": list(part_recs["adapt_train"]),
                "adapt_val": list(part_recs["adapt_val"]),
                "test": list(part_recs["test"]),
            },
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
        recording_entries=tuple(recording_entries),
    )
    return partitions, manifest


def create_source_recording_split(
    records: Sequence[WindowRecord],
    seed: int,
    val_fraction: float = 0.20,
    source_config_id: str = "S4",
    include_additional_unsupervised_for_pretraining: bool = True,
) -> Tuple[Dict[str, Tuple[int, ...]], RecordingSplitManifest]:
    """
    Partition source domain records (`UORED`, `HUST`, `MFPT`) strictly by original
    recording (`recording_key`) before any pretraining or head training:
      1. Supervised head classes (`harmonized_class_id is not None`) are partitioned
         stratified by `(dataset, harmonized_class)` into `source_train` (80%) and
         `source_val` (20%).
      2. Additional unsupervised-only classes present in the active sources
         (`harmonized_class_id is None`, namely `UORED/Class_Cage`, `HUST/Class_44`,
         `HUST/Class_46`, and `MFPT/Class_23`) are partitioned stratified by
         `(dataset, raw_class)` into `pretrain_only_train` (80%) and `pretrain_only_val` (20%)
         before self-supervised pretraining.
      3. `pretrain_train = sorted(source_train + pretrain_only_train)` contains ALL
         training windows of the active source recordings (both supervised head classes
         and additional unsupervised-only classes) while excluding 100% of validation
         recordings (`source_val` and `pretrain_only_val`), test recordings, and target data.
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
    pretrain_only_train_indices: List[int] = []
    pretrain_only_val_indices: List[int] = []

    rec_counts_by_cls: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "source_train": 0,
            "source_val": 0,
            "pretrain_only_train": 0,
            "pretrain_only_val": 0,
            "pretrain_train": 0,
        }
    )
    win_counts_by_cls: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "source_train": 0,
            "source_val": 0,
            "pretrain_only_train": 0,
            "pretrain_only_val": 0,
            "pretrain_train": 0,
        }
    )
    recording_entries: List[RecordingEntryManifest] = []

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

        for r_key in tr_recs:
            w_idxs = rec_map[r_key]
            r0 = records[w_idxs[0]]
            member_prefixes = sorted({records[i].recording_prefix for i in w_idxs})
            recording_entries.append(
                RecordingEntryManifest(
                    dataset=r0.dataset,
                    recording_key=r_key,
                    recording_prefix="+".join(member_prefixes),
                    raw_class=r0.raw_class,
                    harmonized_class=cls_name,
                    condition=r0.condition,
                    partition="source_train",
                    seed=int(seed),
                    window_count=len(w_idxs),
                )
            )
        for r_key in val_recs:
            w_idxs = rec_map[r_key]
            r0 = records[w_idxs[0]]
            member_prefixes = sorted({records[i].recording_prefix for i in w_idxs})
            recording_entries.append(
                RecordingEntryManifest(
                    dataset=r0.dataset,
                    recording_key=r_key,
                    recording_prefix="+".join(member_prefixes),
                    raw_class=r0.raw_class,
                    harmonized_class=cls_name,
                    condition=r0.condition,
                    partition="source_val",
                    seed=int(seed),
                    window_count=len(w_idxs),
                )
            )

        rec_counts_by_cls[cls_name]["total"] += n_recs
        rec_counts_by_cls[cls_name]["source_train"] += len(tr_recs)
        rec_counts_by_cls[cls_name]["source_val"] += len(val_recs)
        rec_counts_by_cls[cls_name]["pretrain_train"] += len(tr_recs)

        win_counts_by_cls[cls_name]["total"] += len(tr_wins) + len(val_wins)
        win_counts_by_cls[cls_name]["source_train"] += len(tr_wins)
        win_counts_by_cls[cls_name]["source_val"] += len(val_wins)
        win_counts_by_cls[cls_name]["pretrain_train"] += len(tr_wins)

    if include_additional_unsupervised_for_pretraining:
        unsup_indices = [
            idx for idx, r in enumerate(records) if r.harmonized_class_id is None
        ]
        unsup_group_to_rec_windows: Dict[Tuple[str, str], Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for idx in unsup_indices:
            rec = records[idx]
            unsup_group_to_rec_windows[(rec.dataset, rec.raw_class)][
                rec.recording_key
            ].append(idx)

        for (ds_name, raw_cls_name), rec_map in sorted(unsup_group_to_rec_windows.items()):
            unique_recs = sorted(rec_map.keys())
            n_recs = len(unique_recs)
            if n_recs < 2:
                raise InsufficientRecordingsError(
                    f"Unsupervised pretraining class '{ds_name}/{raw_cls_name}' has only "
                    f"{n_recs} recording(s); at least 2 are required for a recording-level split."
                )
            n_val = max(1, int(round(n_recs * val_fraction)))
            if n_recs - n_val < 1:
                n_val = n_recs - 1

            seed_token = f"SRC_PRETRAIN::{source_config_id}::{ds_name}::{raw_cls_name}::{int(seed)}"
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

            pretrain_only_train_indices.extend(tr_wins)
            pretrain_only_val_indices.extend(val_wins)

            unsup_label = f"UNSUPERVISED_PRETRAIN_ONLY::{ds_name}::{raw_cls_name}"
            for r_key in tr_recs:
                w_idxs = rec_map[r_key]
                r0 = records[w_idxs[0]]
                member_prefixes = sorted({records[i].recording_prefix for i in w_idxs})
                recording_entries.append(
                    RecordingEntryManifest(
                        dataset=r0.dataset,
                        recording_key=r_key,
                        recording_prefix="+".join(member_prefixes),
                        raw_class=r0.raw_class,
                        harmonized_class=unsup_label,
                        condition=r0.condition,
                        partition="pretrain_only_train",
                        seed=int(seed),
                        window_count=len(w_idxs),
                    )
                )
            for r_key in val_recs:
                w_idxs = rec_map[r_key]
                r0 = records[w_idxs[0]]
                member_prefixes = sorted({records[i].recording_prefix for i in w_idxs})
                recording_entries.append(
                    RecordingEntryManifest(
                        dataset=r0.dataset,
                        recording_key=r_key,
                        recording_prefix="+".join(member_prefixes),
                        raw_class=r0.raw_class,
                        harmonized_class=unsup_label,
                        condition=r0.condition,
                        partition="pretrain_only_val",
                        seed=int(seed),
                        window_count=len(w_idxs),
                    )
                )

            rec_counts_by_cls[unsup_label]["total"] += n_recs
            rec_counts_by_cls[unsup_label]["pretrain_only_train"] += len(tr_recs)
            rec_counts_by_cls[unsup_label]["pretrain_only_val"] += len(val_recs)
            rec_counts_by_cls[unsup_label]["pretrain_train"] += len(tr_recs)

            win_counts_by_cls[unsup_label]["total"] += len(tr_wins) + len(val_wins)
            win_counts_by_cls[unsup_label]["pretrain_only_train"] += len(tr_wins)
            win_counts_by_cls[unsup_label]["pretrain_only_val"] += len(val_wins)
            win_counts_by_cls[unsup_label]["pretrain_train"] += len(tr_wins)

    pretrain_train_all = sorted(train_indices + pretrain_only_train_indices)
    partitions: Dict[str, Tuple[int, ...]] = {
        "source_train": tuple(sorted(train_indices)),
        "source_val": tuple(sorted(val_indices)),
        "pretrain_only_train": tuple(sorted(pretrain_only_train_indices)),
        "pretrain_only_val": tuple(sorted(pretrain_only_val_indices)),
        "pretrain_train": tuple(pretrain_train_all),
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
    total_sup_recs = len(part_recs["source_train"]) + len(part_recs["source_val"])
    total_sup_wins = len(part_samples["source_train"]) + len(part_samples["source_val"])
    total_all_recs = total_sup_recs + len(part_recs["pretrain_only_train"]) + len(part_recs["pretrain_only_val"])
    total_all_wins = total_sup_wins + len(part_samples["pretrain_only_train"]) + len(part_samples["pretrain_only_val"])

    rec_props = {
        "source_train": (len(part_recs["source_train"]) / float(total_sup_recs) if total_sup_recs > 0 else 0.0),
        "source_val": (len(part_recs["source_val"]) / float(total_sup_recs) if total_sup_recs > 0 else 0.0),
        "pretrain_train_of_all_source": (
            len(part_recs["pretrain_train"]) / float(total_all_recs) if total_all_recs > 0 else 0.0
        ),
    }
    win_props = {
        "source_train": (len(part_samples["source_train"]) / float(total_sup_wins) if total_sup_wins > 0 else 0.0),
        "source_val": (len(part_samples["source_val"]) / float(total_sup_wins) if total_sup_wins > 0 else 0.0),
        "pretrain_train_of_all_source": (
            len(part_samples["pretrain_train"]) / float(total_all_wins) if total_all_wins > 0 else 0.0
        ),
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
        recording_entries=tuple(recording_entries),
    )
    return partitions, manifest


# =============================================================================
# 4. LEAK-FREE RMS NORMALIZATION FOR DA & DG (WITH UNIT CONVERSION TO m/s^2)
# =============================================================================


def convert_rms_to_ms2(rms_value: float, unit: str) -> float:
    """
    Convert an RMS reference from its native physical unit (`"m/s^2"` or `"g"`)
    to `m/s^2` using `1 g = 9.80665 m/s^2` (Section 5).
    Raises `ValueError` for unknown or uncalibrated units (e.g. `"V"` or `"unknown"`).
    """
    u = str(unit).strip().lower()
    val = float(rms_value)
    if val <= 0.0 or not math.isfinite(val):
        raise ValueError(f"Invalid RMS value: {rms_value}")
    if u in ("m/s^2", "m/s2", "ms^-2", "ms-2"):
        return val
    if u in ("g", "g-force", "g_force"):
        return val * STANDARD_GRAVITY_MS2
    raise ValueError(
        f"Cannot convert physical unit '{unit}' to m/s^2 without sensor calibration factor."
    )


def compute_dg_global_rms_reference_ms2(
    source_rms_refs: Mapping[str, float],
    source_units: Mapping[str, str],
    active_sources: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compute the Domain Generalization global RMS reference in `m/s^2` across the
    active sources of configuration `S1`, `S2`, `S3`, or `S4` (Section 5):
      1. Convert every active source reference to `m/s^2` using `1 g = 9.80665 m/s^2`.
      2. Compute the geometric mean:
           `RMS_global = exp(mean(log(RMS_ref_i_ms2)))`
      3. Never use arithmetic mean for DG global aggregation.
    """
    selected = list(active_sources) if active_sources is not None else list(source_rms_refs.keys())
    if not selected:
        raise ValueError("active_sources cannot be empty.")

    raw_refs: Dict[str, float] = {}
    units_used: Dict[str, str] = {}
    converted_ms2: Dict[str, float] = {}

    for s_name in selected:
        if s_name not in source_rms_refs:
            raise KeyError(f"Active source '{s_name}' missing from source_rms_refs.")
        u = str(source_units.get(s_name, "unknown"))
        raw_val = float(source_rms_refs[s_name])
        raw_refs[s_name] = raw_val
        units_used[s_name] = u
        converted_ms2[s_name] = convert_rms_to_ms2(raw_val, u)

    log_vals = [math.log(v) for v in converted_ms2.values()]
    rms_global_ms2 = float(math.exp(sum(log_vals) / float(len(log_vals))))
    min_ms2 = min(converted_ms2.values())
    max_ms2 = max(converted_ms2.values())
    scale_ratio_ms2 = max_ms2 / max(min_ms2, EPSILON_RMS)

    return {
        "active_sources": selected,
        "raw_rms_refs": raw_refs,
        "raw_units": units_used,
        "converted_rms_refs_ms2": converted_ms2,
        "conversion_constant_1g_ms2": STANDARD_GRAVITY_MS2,
        "rms_global_geometric_ms2": rms_global_ms2,
        "scale_ratio_ms2_max_over_min": float(scale_ratio_ms2),
        "formula": "RMS_global = exp(mean(log(RMS_ref_i_ms2)))",
    }


@dataclass(frozen=True)
class DARMSNormalizer:
    """
    Leak-free Domain Adaptation RMS reference normalizer (Section 6):
      - Estimates `rms_ref` exclusively from `Normal` windows in the target's
        20% adaptation partition (`adapt`).
      - Applies the exact same `rms_ref` to both adaptation and test windows.
      - Never touches test windows during `fit`.
      - Records `calibration_mode` (`"internal_adaptation_normal"` or
        `"external_related_domain_adaptation_normal"` for `CWRU_48k` calibrated
        from `CWRU_12k` adaptation normal recordings).
    """

    target_dataset: str
    rms_ref: float
    n_reference_windows: int
    reference_recording_keys: Tuple[str, ...]
    apply_detrend: bool = True
    calibration_mode: str = "internal_adaptation_normal"
    reference_domain: str = ""

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
    cross_reference_domain: str = "CWRU_12k",
    cross_reference_forbidden_recording_keys: Optional[Sequence[str]] = None,
) -> DARMSNormalizer:
    """
    Fit `DARMSNormalizer` exclusively on `Normal` windows inside `adapt_indices`
    (or, for `CWRU_48k` which has 0 native Normal recordings, exclusively on the
    `Normal` windows from the 20% adaptation partition of `CWRU_12k`, verifying zero
    overlap with `cross_reference_forbidden_recording_keys` [validation/test of `CWRU_12k`]).
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
        or records[i].raw_class in ("Class_0", "Class_Normal", "Class_23", "Class_26")
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
            calibration_mode="internal_adaptation_normal",
            reference_domain=ds_name,
        )

    if cross_reference_normal_windows is not None and len(cross_reference_normal_windows) > 0:
        ref_recs = tuple(sorted(set(cross_reference_recording_keys or ())))
        if cross_reference_forbidden_recording_keys:
            forbidden = set(cross_reference_forbidden_recording_keys)
            leaked = set(ref_recs) & forbidden
            if leaked:
                raise ValueError(
                    f"External calibration leakage detected! Cross-reference recording keys "
                    f"overlap with forbidden validation/test recordings: {sorted(leaked)}"
                )
        ref_windows = []
        for w in cross_reference_normal_windows:
            arr = np.asarray(w, dtype=np.float64).reshape(-1)
            if apply_detrend:
                arr = np.asarray(detrend_linear_window(arr), dtype=np.float64)
            ref_windows.append(arr)
        rms_val = compute_reference_rms(ref_windows)
        return DARMSNormalizer(
            target_dataset=ds_name,
            rms_ref=float(rms_val),
            n_reference_windows=len(ref_windows),
            reference_recording_keys=ref_recs,
            apply_detrend=apply_detrend,
            calibration_mode="external_related_domain_adaptation_normal",
            reference_domain=cross_reference_domain,
        )

    raise ValueError(
        f"Target dataset '{ds_name}' has 0 Normal windows in the adaptation partition "
        "and no explicit cross_reference_normal_windows were provided."
    )


def audit_cwru_12k_48k_calibration_compatibility() -> Dict[str, Any]:
    """
    Return the physical and acquisition compatibility diagnosis between `CWRU_12k`
    and `CWRU_48k` (Section 6 & Deliverable 8).
    """
    return {
        "source_reference_dataset": "CWRU_12k",
        "target_calibrated_dataset": "CWRU_48k",
        "test_rig": "Case Western Reserve University (CWRU) 2 HP Reliance Electric motor test stand",
        "accelerometer_position": "Drive End (DE) bearing housing (12 o'clock position)",
        "bearing_model": "6205-2RS JEM SKF deep-groove ball bearing",
        "physical_unit": "g (calibrated gravitational acceleration)",
        "motor_loads_hp": [0, 1, 2, 3],
        "cwru_12k_sampling_rate_hz": 12000,
        "cwru_48k_sampling_rate_hz": 48000,
        "cwru_12k_normal_rms_by_load_g": {
            "Load_0HP": 0.075655,
            "Load_1HP": 0.066215,
            "Load_2HP": 0.064112,
            "Load_3HP": 0.069971,
            "overall_mean_g": 0.068036,
        },
        "cwru_12k_fault_rms_range_g": [0.195, 0.271],
        "cwru_48k_fault_rms_range_g": [0.114, 0.746],
        "is_physically_compatible": True,
        "calibration_rule": (
            "Use exclusively the 2 Normal recordings assigned to the 20% adaptation "
            "partition of CWRU_12k (for the active seed) to compute RMS_ref for CWRU_48k. "
            "Never use the 6 test recordings of CWRU_12k. Record operation as "
            "'external_related_domain_adaptation_normal'."
        ),
    }


@dataclass(frozen=True)
class DGRMSNormalizer:
    """
    Strict Domain Generalization RMS normalizer (Section 5):
      - Computed exclusively from active source datasets (`source_rms_refs`).
      - Converts all active source references to `m/s^2` (`1 g = 9.80665 m/s^2`)
        before computing the geometric mean `RMS_global = exp(mean(log(RMS_ref_i_ms2)))`.
      - Zero target data or statistics are ever inspected.
    """

    source_rms_refs: Dict[str, float]
    source_rms_refs_ms2: Dict[str, float]
    source_units: Dict[str, str]
    aggregation_mode: str
    global_target_rms_ref: float
    comparability_report: Dict[str, Any]
    apply_detrend: bool = True

    def transform_source_windows(
        self,
        source_name: str,
        windows: Sequence[np.ndarray],
        source_unit: Optional[str] = None,
        feature_mode: str = "full_141",
        source_norm_mode: str = "global_rms",
    ) -> List[np.ndarray]:
        """
        Normalize source domain windows in DG:
          - `source_norm_mode="per_source_rms"`: normalizes each source domain `i` by its
            own healthy training `RMS_ref_i` (`self.source_rms_refs[source_name]`), aligning
            all active sources to `RMS_normal == 1.0` while preserving linear `detrend` and
            never touching any target domain statistics.
          - `source_norm_mode="global_rms"` (legacy default for backward compatibility):
            divides every source by the unified `self.global_target_rms_ref` in `m/s^2`
            (converting `"g"` sources by `* 9.80665`).
        """
        if source_name not in self.source_rms_refs:
            raise KeyError(f"Unknown source dataset '{source_name}'.")
        if source_norm_mode not in ("global_rms", "per_source_rms"):
            raise ValueError(
                f"Unsupported source_norm_mode '{source_norm_mode}'. "
                "Expected 'per_source_rms' or 'global_rms'."
            )
        resolved_unit = (
            str(source_unit).strip()
            if source_unit is not None
            else str(
                self.source_units.get(
                    source_name, DATASET_PHYSICAL_UNITS.get(source_name, "")
                )
            ).strip()
        )
        u_low = resolved_unit.lower()
        is_ms2_ref = self.aggregation_mode in (
            "geometric_mean_ms2",
            "strict_check_geometric_ms2",
        )
        if is_ms2_ref and u_low in ("v", "volt", "volts", "voltage"):
            if feature_mode == "full_141":
                raise IncomparableSourceRMSReferenceError(
                    f"DG 141-feature normalization is blocked for source '{source_name}' "
                    f"(physical unit '{resolved_unit}' = Volts) under m/s^2 global reference."
                )

        if source_norm_mode == "per_source_rms":
            # Dividing in native unit by source_rms_refs[source_name] is mathematically
            # identical to converting both numerator and denominator to m/s^2.
            unit_multiplier = 1.0
            denom = max(float(self.source_rms_refs[source_name]), EPSILON_RMS)
        else:
            unit_multiplier = (
                STANDARD_GRAVITY_MS2
                if (is_ms2_ref and u_low in ("g", "g-force", "g_force"))
                else 1.0
            )
            denom = max(float(self.global_target_rms_ref), EPSILON_RMS)

        out: List[np.ndarray] = []
        for w in windows:
            arr = np.asarray(w, dtype=np.float64).reshape(-1) * unit_multiplier
            if self.apply_detrend:
                arr = np.asarray(detrend_linear_window(arr), dtype=np.float64)
            out.append((arr / denom).astype(np.float32))
        return out

    def transform_target_windows(
        self,
        windows: Sequence[np.ndarray],
        target_dataset: Optional[str] = None,
        target_unit: Optional[str] = None,
        feature_mode: str = "full_141",
    ) -> List[np.ndarray]:
        """
        Normalize target domain windows in DG using `self.global_target_rms_ref` (which is in
        `m/s^2` when `aggregation_mode in ("geometric_mean_ms2", "strict_check_geometric_ms2")`):
          - If `target_dataset` or `target_unit` has physical unit `"V"` (`PU`) and
            `feature_mode == "full_141"`, raises `IncomparableSourceRMSReferenceError`
            because applying an `m/s^2` reference to a signal in Volts without a calibrated
            sensor sensitivity factor (`V -> m/s^2`) is physically invalid (Pending Item 1).
          - If `target_dataset` or `target_unit` has physical unit `"g"` (`CWRU_12k`, `CWRU_48k`)
            and `self.aggregation_mode` is in `m/s^2`, converts the target waveform from `g`
            to `m/s^2` (`* 9.80665`) before dividing by `global_target_rms_ref` (`m/s^2`).
        """
        resolved_unit = (
            str(target_unit).strip()
            if target_unit is not None
            else DATASET_PHYSICAL_UNITS.get(str(target_dataset or ""), "")
        )
        u_low = resolved_unit.lower()
        is_ms2_ref = self.aggregation_mode in (
            "geometric_mean_ms2",
            "strict_check_geometric_ms2",
        )
        if is_ms2_ref and u_low in ("v", "volt", "volts", "voltage"):
            if feature_mode == "full_141":
                raise IncomparableSourceRMSReferenceError(
                    f"DG 141-feature evaluation is blocked for target '{target_dataset or 'PU'}' "
                    f"(physical unit '{resolved_unit}' = Volts) because source RMS_global "
                    f"({self.global_target_rms_ref:.4f} m/s^2) is in m/s^2 and no verified "
                    "sensor sensitivity factor (V -> m/s^2) is available. Use the 126-feature "
                    "scale-invariant ablation or provide a verified physical calibration factor."
                )

        unit_multiplier = (
            STANDARD_GRAVITY_MS2
            if (is_ms2_ref and u_low in ("g", "g-force", "g_force"))
            else 1.0
        )
        denom = max(float(self.global_target_rms_ref), EPSILON_RMS)
        out: List[np.ndarray] = []
        for w in windows:
            arr = np.asarray(w, dtype=np.float64).reshape(-1) * unit_multiplier
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
    Audit physical units, conversion to `m/s^2` (`1 g = 9.80665 m/s^2`), and scale
    ratios across source datasets (Section 5 & Deliverable 5).
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

    # Attempt conversion of all active sources to m/s^2
    converted_ms2: Dict[str, float] = {}
    conversion_possible = True
    for k, v in source_rms_refs.items():
        try:
            converted_ms2[k] = convert_rms_to_ms2(v, source_units.get(k, "unknown"))
        except ValueError:
            conversion_possible = False

    if conversion_possible and converted_ms2:
        ms2_vals = list(converted_ms2.values())
        min_ms2 = min(ms2_vals)
        max_ms2 = max(ms2_vals)
        scale_ratio_ms2 = max_ms2 / max(min_ms2, EPSILON_RMS)
        geom_mean_ms2 = math.exp(sum(math.log(v) for v in ms2_vals) / float(len(ms2_vals)))
        arith_mean_ms2 = sum(ms2_vals) / float(len(ms2_vals))
    else:
        scale_ratio_ms2 = float("nan")
        geom_mean_ms2 = float("nan")
        arith_mean_ms2 = float("nan")

    units_compatible_raw = len(units_set) == 1 and "unknown" not in units_set
    scales_comparable_raw = scale_ratio <= float(max_allowed_ratio)
    scales_comparable_ms2 = (
        conversion_possible and scale_ratio_ms2 <= float(max_allowed_ratio)
    )
    is_directly_comparable = bool(units_compatible_raw and scales_comparable_raw)

    arith_mean = sum(vals) / float(len(vals))
    geom_mean = math.exp(sum(math.log(v) for v in vals) / float(len(vals)))

    reasons: List[str] = []
    if not units_compatible_raw:
        reasons.append(
            f"Heterogeneous raw physical units across sources: {dict(source_units)}"
        )
    if not scales_comparable_raw:
        reasons.append(
            f"Raw source healthy RMS_ref spans a {scale_ratio:.2f}x factor "
            f"(min={min_rms:.6f}, max={max_rms:.6f} > max_allowed_ratio={max_allowed_ratio:.2f}x)"
        )
    if conversion_possible and not scales_comparable_ms2:
        reasons.append(
            f"Even after converting 1 g = {STANDARD_GRAVITY_MS2} m/s^2, source healthy "
            f"RMS_ref in m/s^2 spans a {scale_ratio_ms2:.2f}x factor "
            f"({ {k: round(v, 4) for k, v in converted_ms2.items()} })"
        )

    # Precompute geometric mean in m/s^2 per source configuration S1..S4 when available
    per_config_geom_ms2: Dict[str, Optional[float]] = {}
    for sc_id, sc_sources in SOURCE_CONFIGS.items():
        if all(s in converted_ms2 for s in sc_sources):
            per_config_geom_ms2[sc_id] = float(
                math.exp(
                    sum(math.log(converted_ms2[s]) for s in sc_sources)
                    / float(len(sc_sources))
                )
            )
        else:
            per_config_geom_ms2[sc_id] = None

    return {
        "is_directly_comparable": is_directly_comparable,
        "units_compatible": units_compatible_raw,
        "scales_comparable": scales_comparable_raw,
        "conversion_to_ms2_possible": conversion_possible,
        "scales_comparable_after_ms2_conversion": scales_comparable_ms2,
        "scale_ratio_max_over_min": float(scale_ratio),
        "scale_ratio_ms2_max_over_min": float(scale_ratio_ms2),
        "arithmetic_mean_rms_ref": float(arith_mean),
        "geometric_mean_rms_ref": float(geom_mean),
        "geometric_mean_rms_ref_ms2": float(geom_mean_ms2),
        "arithmetic_mean_rms_ref_ms2": float(arith_mean_ms2),
        "source_rms_refs": {k: float(v) for k, v in source_rms_refs.items()},
        "source_rms_refs_ms2": converted_ms2,
        "per_config_geometric_mean_ms2": per_config_geom_ms2,
        "source_units": {k: str(source_units.get(k, "unknown")) for k in source_rms_refs},
        "incomparability_reasons": reasons,
    }


def fit_dg_rms_normalizer(
    source_normal_windows: Mapping[str, Sequence[np.ndarray]],
    source_units: Mapping[str, str],
    aggregation_mode: str = "geometric_mean_ms2",
    max_allowed_ratio: float = 5.0,
    apply_detrend: bool = True,
) -> DGRMSNormalizer:
    """
    Fit `DGRMSNormalizer` using ONLY source-domain healthy windows (zero target data).

    Supported `aggregation_mode` values:
      - `"geometric_mean_ms2"` (default per Section 5): converts every active source's
        healthy `RMS_ref` to `m/s^2` (`1 g = 9.80665 m/s^2`) and computes the geometric
        mean `RMS_global = exp(mean(log(RMS_ref_i_ms2)))`.
      - `"strict_check_geometric_ms2"`: converts to `m/s^2` and checks if the scale ratio
        in `m/s^2` is within `max_allowed_ratio`; raises `IncomparableSourceRMSReferenceError`
        if the physical scales in `m/s^2` still diverge beyond `max_allowed_ratio`.
      - `"strict_check_arithmetic"`: raises `IncomparableSourceRMSReferenceError` when raw
        units or scales are incompatible (kept for backward-compatibility tests).
      - `"explicit_geometric_mean"`: geometric mean on raw values.
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

    if aggregation_mode == "geometric_mean_ms2":
        if not report["conversion_to_ms2_possible"]:
            raise IncomparableSourceRMSReferenceError(
                f"Cannot convert all source units to m/s^2: {dict(source_units)}"
            )
        global_ref = report["geometric_mean_rms_ref_ms2"]
    elif aggregation_mode == "strict_check_geometric_ms2":
        if not report["scales_comparable_after_ms2_conversion"]:
            raise IncomparableSourceRMSReferenceError(
                "Source RMS references remain incomparable even after conversion to m/s^2: "
                + "; ".join(report["incomparability_reasons"])
            )
        global_ref = report["geometric_mean_rms_ref_ms2"]
    elif aggregation_mode == "strict_check_arithmetic":
        if not report["is_directly_comparable"]:
            raise IncomparableSourceRMSReferenceError(
                "Cannot silently average source RMS_ref values in DG: "
                + "; ".join(report["incomparability_reasons"])
            )
        global_ref = report["arithmetic_mean_rms_ref"]
    elif aggregation_mode == "explicit_geometric_mean":
        global_ref = (
            report["geometric_mean_rms_ref_ms2"]
            if report["conversion_to_ms2_possible"]
            else report["geometric_mean_rms_ref"]
        )
    elif aggregation_mode == "explicit_arithmetic_mean":
        global_ref = report["arithmetic_mean_rms_ref"]
    else:
        raise ValueError(f"Unknown DG aggregation_mode '{aggregation_mode}'.")

    return DGRMSNormalizer(
        source_rms_refs=source_rms_refs,
        source_rms_refs_ms2=dict(report.get("source_rms_refs_ms2", {})),
        source_units={k: str(source_units.get(k, "unknown")) for k in source_rms_refs},
        aggregation_mode=aggregation_mode,
        global_target_rms_ref=float(global_ref),
        comparability_report=report,
        apply_detrend=apply_detrend,
    )


def fit_source_only_standard_scaler(
    X_source_train: np.ndarray,
    X_source_val: Optional[np.ndarray] = None,
    X_target_dict: Optional[Mapping[str, np.ndarray]] = None,
) -> Tuple[StandardScaler, np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
    """
    Fit `StandardScaler` exclusively on `X_source_train` (Section 5: "O StandardScaler
    deve ser ajustado exclusivamente com os dados de treinamento das fontes. Aplicar os
    mesmos parâmetros de padronização aos destinos, sem reajuste. Não utilizar dados dos
    destinos para estimar estatísticas de normalização DG.").
    """
    X_tr = np.asarray(X_source_train, dtype=np.float32)
    if X_tr.ndim != 2 or len(X_tr) == 0:
        raise ValueError(f"X_source_train must be a non-empty 2D array, got {X_tr.shape}.")

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_scaled = (
        scaler.transform(np.asarray(X_source_val, dtype=np.float32)).astype(np.float32)
        if X_source_val is not None and len(X_source_val) > 0
        else None
    )
    scaled_targets: Dict[str, np.ndarray] = {}
    if X_target_dict is not None:
        for t_name, X_t in X_target_dict.items():
            scaled_targets[t_name] = scaler.transform(
                np.asarray(X_t, dtype=np.float32)
            ).astype(np.float32)

    return scaler, X_tr_scaled, X_val_scaled, scaled_targets


# =============================================================================
# 5. MATHEMATICAL CATALOG OF 126 SCALE-INVARIANT & 15 SCALE-SENSITIVE FEATURES
# =============================================================================


@dataclass(frozen=True)
class FeatureScaleInvarianceItem:
    """Metadata and mathematical scale-dependence proof for a single feature (0..140)."""

    index: int
    name: str
    domain_group: str
    source_module: str
    is_scale_invariant: bool
    scaling_law: str
    mathematical_justification: str


def get_141_feature_scale_invariance_catalog() -> List[FeatureScaleInvarianceItem]:
    """
    Return the complete, ordered 141-item catalog (`indices 0..140`) identifying the
    exact **126 scale-invariant features** (`f(c * x) == f(x)` for `c > 0`) and
    **15 scale-sensitive features** (`f(c * x) = c^k * f(x)` or `f(x) + 2 log(c)`),
    with their mathematical justification and code implementation location (Section 5 &
    Deliverable 6).

    Structure of the 141-feature vector (`extract_fusion_features`):
      - `0..15` (`16` features): `src/features/extractors_v2.py::extract_advanced_features`
        * `0..6` (7 Time features):
          - `0: v2_time_abs_mean` (Sensitive, `c^1`)
          - `1: v2_time_shannon_entropy` (Invariant, `c^0`: normalized histogram bin probabilities)
          - `2: v2_time_hilbert_envelope_mean` (Sensitive, `c^1`)
          - `3: v2_time_hilbert_envelope_rms` (Sensitive, `c^1`)
          - `4: v2_time_hilbert_envelope_kurtosis` (Invariant, `c^0`: fourth standardized moment)
          - `5: v2_time_tkeo_mean` (Sensitive, `c^2`: `Ψ[c x] = c^2 Ψ[x]`)
          - `6: v2_time_tkeo_std` (Sensitive, `c^2`)
        * `7..12` (6 Frequency Welch PSD features):
          - `7: v2_freq_welch_peak_freq` (Invariant, `c^0`: `argmax_f(c^2 Pxx(f)) = argmax_f Pxx(f)`)
          - `8: v2_freq_welch_peak_mag` (Sensitive, `c^2`: `max_f Pxx(f)`)
          - `9: v2_freq_welch_total_power` (Sensitive, `c^2`: `sum_f Pxx(f)`)
          - `10: v2_freq_welch_relative_low_band_power` (Invariant, `c^0`: `P_low / P_total`)
          - `11: v2_freq_welch_high_low_energy_ratio` (Invariant, `c^0`: `P_high / P_low`)
          - `12: v2_freq_welch_legacy_band_power` (Sensitive, `c^2`: `sum_f Pxx(f)`)
        * `13..15` (3 Time-Frequency STFT/PSD features):
          - `13: v2_tf_log_stft_power_mean` (Sensitive, `+2 log(c)` shift: `mean(log(|c Z|^2)) = mean(log(|Z|^2)) + 2 log(c)`)
          - `14: v2_tf_log_stft_power_std` (Invariant, `c^0`: `std(log(|Z|^2) + 2 log(c)) = std(log(|Z|^2))`)
          - `15: v2_tf_psd_peak_freq_1_200hz` (Invariant, `c^0`: `argmax_{f in [1,200]} Pxx(f)`)
        -> Subtotal `extractors_v2.py` (16 features): **7 Invariant, 9 Sensitive**.

      - `16..140` (`125` features): `src/features/signalai_wrapper.py::SignalAIWrapper(pipeline_name="all")`
        * `16..26` (11 Time-Domain features):
          - `16..19` (4 Sensitive): `signai_time_rms` (`c^1`), `signai_time_peak` (`c^1`),
            `signai_time_peak_to_peak` (`c^1`), `signai_time_variance` (`c^2`).
          - `20..26` (7 Invariant): `signai_time_skewness` (`c^0`), `signai_time_kurtosis` (`c^0`),
            `signai_time_crest_factor` (`Peak/RMS`), `signai_time_shape_factor` (`RMS/MeanAbs`),
            `signai_time_impulse_factor` (`Peak/MeanAbs`), `signai_time_clearance_factor`
            (`Peak/MeanSqrtAbs^2`), `signai_time_zero_crossing_rate` (`sign(c x_t) = sign(x_t)`).
        * `27..41` (15 Frequency-Domain features):
          - `27` (1 Sensitive): `signai_freq_mean_spectral_power` (`c^2`).
          - `28..41` (14 Invariant): normalized spectral moments and ratios (`spectral_centroid`,
            `spectral_rms_freq`, `spectral_spread_std`, `spectral_skewness`, `spectral_kurtosis`,
            `spectral_flatness`, `spectral_rolloff_85`, `spectral_entropy`, `dominant_freq_hz`,
            `relative_band_power_b1..b4`, `harmonic_energy_ratio`).
        * `42..114` (73 Wavelet Packet Decomposition `WPD db4` level-6 features):
          - `42..105` (64 Invariant): relative sub-band energy ratios `E_{6,k} / E_total` (`k=0..63`).
          - `106` (1 Invariant): `wpd_shannon_entropy` (`-sum p_k log p_k`).
          - `107..113` (7 Invariant): macro-band relative ratios, sub-band kurtosis and skewness.
          - `114` (1 Sensitive): `wpd_total_energy` (`sum_k E_{6,k}`, scales as `c^2`).
        * `115..140` (26 Empirical Mode Decomposition `EMD` features across 6 IMFs):
          - `115..120` (6 Invariant): relative IMF energy ratios `E(IMF_m) / E_total` (`m=1..6`).
          - `121..126` (6 Invariant): Pearson correlation `corr(IMF_m, x)` (`c` cancels in covariance/std).
          - `127..132` (6 Invariant): IMF kurtosis (`m=1..6`).
          - `133..138` (6 Invariant): IMF skewness (`m=1..6`).
          - `139` (1 Invariant): `emd_energy_entropy`.
          - `140` (1 Invariant): `emd_imf1_zero_crossing_rate`.
        -> Subtotal `SignalAIWrapper` (125 features): **119 Invariant, 6 Sensitive**.

      - Grand Total (`0..140` = 141 features):
        **`7 + 119 = 126` Scale-Invariant features** and **`9 + 6 = 15` Scale-Sensitive features**.
    """
    catalog: List[FeatureScaleInvarianceItem] = [
        # --- 0..15: src/features/extractors_v2.py (16 features: 7 invariant, 9 sensitive) ---
        FeatureScaleInvarianceItem(
            0, "v2_time_abs_mean", "Time (v2)", "src/features/extractors_v2.py:L30",
            False, "f(c*x) = c * f(x)", "Mean absolute amplitude E[|x|] is homogeneous of degree 1."
        ),
        FeatureScaleInvarianceItem(
            1, "v2_time_shannon_entropy", "Time (v2)", "src/features/extractors_v2.py:L12-L17",
            True, "f(c*x) = f(x)",
            "scipy.stats.entropy(hist) normalizes hist / sum(hist); bin counts over min-max range are scale-invariant."
        ),
        FeatureScaleInvarianceItem(
            2, "v2_time_hilbert_envelope_mean", "Time (v2)", "src/features/extractors_v2.py:L34-L36",
            False, "f(c*x) = c * f(x)", "Hilbert transform is linear, so |H(c*x)| = c*|H(x)|."
        ),
        FeatureScaleInvarianceItem(
            3, "v2_time_hilbert_envelope_rms", "Time (v2)", "src/features/extractors_v2.py:L37",
            False, "f(c*x) = c * f(x)", "RMS of Hilbert amplitude envelope scales linearly with c > 0."
        ),
        FeatureScaleInvarianceItem(
            4, "v2_time_hilbert_envelope_kurtosis", "Time (v2)", "src/features/extractors_v2.py:L38",
            True, "f(c*x) = f(x)", "Kurtosis E[(A - μ)^4] / σ^4 is homogeneous of degree 0."
        ),
        FeatureScaleInvarianceItem(
            5, "v2_time_tkeo_mean", "Time (v2)", "src/features/extractors_v2.py:L5-L10,L41-L42",
            False, "f(c*x) = c^2 * f(x)", "Teager-Kaiser operator Ψ[x_n] = x_n^2 - x_{n-1}x_{n+1} is homogeneous of degree 2."
        ),
        FeatureScaleInvarianceItem(
            6, "v2_time_tkeo_std", "Time (v2)", "src/features/extractors_v2.py:L43",
            False, "f(c*x) = c^2 * f(x)", "Standard deviation of TKEO signal scales with c^2."
        ),
        FeatureScaleInvarianceItem(
            7, "v2_freq_welch_peak_freq", "Frequency (v2)", "src/features/extractors_v2.py:L53-L56",
            True, "f(c*x) = f(x)", "argmax_f Pxx(f) is invariant to positive scalar multiplication c^2 * Pxx(f)."
        ),
        FeatureScaleInvarianceItem(
            8, "v2_freq_welch_peak_mag", "Frequency (v2)", "src/features/extractors_v2.py:L57",
            False, "f(c*x) = c^2 * f(x)", "Maximum Welch PSD magnitude scales with c^2."
        ),
        FeatureScaleInvarianceItem(
            9, "v2_freq_welch_total_power", "Frequency (v2)", "src/features/extractors_v2.py:L59",
            False, "f(c*x) = c^2 * f(x)", "Sum of Welch PSD values sum(Pxx) scales with c^2."
        ),
        FeatureScaleInvarianceItem(
            10, "v2_freq_welch_relative_low_band_power", "Frequency (v2)", "src/features/extractors_v2.py:L68",
            True, "f(c*x) = f(x)", "Ratio P_low / P_total cancels c^2 in numerator and denominator."
        ),
        FeatureScaleInvarianceItem(
            11, "v2_freq_welch_high_low_energy_ratio", "Frequency (v2)", "src/features/extractors_v2.py:L69",
            True, "f(c*x) = f(x)", "Ratio P_high / P_low cancels c^2 in numerator and denominator."
        ),
        FeatureScaleInvarianceItem(
            12, "v2_freq_welch_legacy_band_power", "Frequency (v2)", "src/features/extractors_v2.py:L70",
            False, "f(c*x) = c^2 * f(x)", "Absolute spectral band power scales with c^2."
        ),
        FeatureScaleInvarianceItem(
            13, "v2_tf_log_stft_power_mean", "Time-Frequency (v2)", "src/features/extractors_v2.py:L79-L83",
            False, "f(c*x) = f(x) + 2*log(c)", "Mean log STFT power shifts additively by 2*log(c) under amplitude scaling."
        ),
        FeatureScaleInvarianceItem(
            14, "v2_tf_log_stft_power_std", "Time-Frequency (v2)", "src/features/extractors_v2.py:L84",
            True, "f(c*x) = f(x)", "Standard deviation of log(|c*Z|^2) = log(|Z|^2) + 2*log(c) is invariant to additive constant 2*log(c)."
        ),
        FeatureScaleInvarianceItem(
            15, "v2_tf_psd_peak_freq_1_200hz", "Time-Frequency (v2)", "src/features/extractors_v2.py:L87-L92",
            True, "f(c*x) = f(x)", "Peak frequency in 1-200 Hz band argmax_{f in [1,200]} Pxx(f) is scale-invariant."
        ),
    ]

    # --- 16..26: SignalAI Time-Domain features (11 features: 4 sensitive, 7 invariant) ---
    signai_time_sensitive = [
        (16, "signai_time_rms", "f(c*x) = c * f(x)", "Root mean square sqrt(mean(x^2)) is homogeneous of degree 1."),
        (17, "signai_time_peak", "f(c*x) = c * f(x)", "Peak amplitude max(|x|) is homogeneous of degree 1."),
        (18, "signai_time_peak_to_peak", "f(c*x) = c * f(x)", "Peak-to-peak max(x) - min(x) is homogeneous of degree 1."),
        (19, "signai_time_variance", "f(c*x) = c^2 * f(x)", "Signal variance Var(x) is homogeneous of degree 2."),
    ]
    for idx, name, law, just in signai_time_sensitive:
        catalog.append(
            FeatureScaleInvarianceItem(
                idx, name, "Time (SignalAI)", "src/features/signalai_wrapper.py:L37",
                False, law, just
            )
        )

    signai_time_invariant = [
        (20, "signai_time_skewness", "Third standardized moment E[(x-μ)^3]/σ^3 cancels c^3."),
        (21, "signai_time_kurtosis", "Fourth standardized moment E[(x-μ)^4]/σ^4 cancels c^4."),
        (22, "signai_time_crest_factor", "Crest factor Peak / RMS cancels c in numerator and denominator."),
        (23, "signai_time_shape_factor", "Shape factor RMS / Mean(|x|) cancels c in numerator and denominator."),
        (24, "signai_time_impulse_factor", "Impulse factor Peak / Mean(|x|) cancels c in numerator and denominator."),
        (25, "signai_time_clearance_factor", "Clearance factor Peak / Mean(sqrt(|x|))^2 cancels c in numerator and denominator."),
        (26, "signai_time_zero_crossing_rate", "Zero-crossing indicator I(x_t * x_{t+1} < 0) is invariant for c > 0."),
    ]
    for idx, name, just in signai_time_invariant:
        catalog.append(
            FeatureScaleInvarianceItem(
                idx, name, "Time (SignalAI)", "src/features/signalai_wrapper.py:L37",
                True, "f(c*x) = f(x)", just
            )
        )

    # --- 27..41: SignalAI Frequency-Domain features (15 features: 1 sensitive, 14 invariant) ---
    catalog.append(
        FeatureScaleInvarianceItem(
            27, "signai_freq_mean_spectral_power", "Frequency (SignalAI)", "src/features/signalai_wrapper.py:L37",
            False, "f(c*x) = c^2 * f(x)", "Mean spectral power (1/K) sum_k P(f_k) scales with c^2."
        )
    )
    signai_freq_invariant_names = [
        ("signai_freq_spectral_centroid", "Weighted frequency mean sum(f_k P(f_k)) / sum(P(f_k)) cancels c^2."),
        ("signai_freq_spectral_rms_freq", "Root mean square frequency sqrt(sum(f_k^2 P(f_k)) / sum(P(f_k))) cancels c^2."),
        ("signai_freq_spectral_spread_std", "Spectral standard deviation around spectral centroid cancels c^2."),
        ("signai_freq_spectral_skewness", "Third standardized spectral moment around centroid is dimensionless."),
        ("signai_freq_spectral_kurtosis", "Fourth standardized spectral moment around centroid is dimensionless."),
        ("signai_freq_spectral_flatness", "Ratio of geometric mean to arithmetic mean of spectrum cancels c."),
        ("signai_freq_spectral_rolloff_85", "85% cumulative spectral energy quantile frequency is invariant to c^2."),
        ("signai_freq_spectral_entropy", "Shannon entropy of normalized PSD p_k = P(f_k)/sum(P) is invariant to c^2."),
        ("signai_freq_dominant_frequency", "Dominant FFT bin frequency argmax_f |X(f)| is invariant to c > 0."),
        ("signai_freq_rel_band_power_b1", "Relative spectral power in band B1 divided by total power cancels c^2."),
        ("signai_freq_rel_band_power_b2", "Relative spectral power in band B2 divided by total power cancels c^2."),
        ("signai_freq_rel_band_power_b3", "Relative spectral power in band B3 divided by total power cancels c^2."),
        ("signai_freq_rel_band_power_b4", "Relative spectral power in band B4 divided by total power cancels c^2."),
        ("signai_freq_harmonic_energy_ratio", "Ratio of rotational harmonic energy to total spectral energy cancels c^2."),
    ]
    for offset, (name, just) in enumerate(signai_freq_invariant_names, start=28):
        catalog.append(
            FeatureScaleInvarianceItem(
                offset, name, "Frequency (SignalAI)", "src/features/signalai_wrapper.py:L37",
                True, "f(c*x) = f(x)", just
            )
        )

    # --- 42..114: SignalAI WPD db4 features (73 features: 72 invariant, 1 sensitive) ---
    for subband_idx in range(64):
        idx = 42 + subband_idx
        catalog.append(
            FeatureScaleInvarianceItem(
                idx,
                f"signai_wpd_db4_rel_energy_subband_{subband_idx:02d}",
                "WPD (SignalAI)",
                "src/features/signalai_wrapper.py:L37",
                True,
                "f(c*x) = f(x)",
                f"Relative WPD node energy ||d_{{6,{subband_idx}}}||_2^2 / sum_j ||d_{{6,j}}||_2^2 cancels c^2 by linearity of wavelet filters.",
            )
        )
    wpd_extra_invariant = [
        (106, "signai_wpd_db4_shannon_entropy", "Shannon entropy of the 64 normalized WPD relative energy ratios."),
        (107, "signai_wpd_db4_high_low_macroband_ratio", "Ratio of upper to lower WPD macro-band energies cancels c^2."),
        (108, "signai_wpd_db4_macroband_1_kurtosis", "Kurtosis of WPD coefficients in macro-band 1 cancels c^4."),
        (109, "signai_wpd_db4_macroband_2_kurtosis", "Kurtosis of WPD coefficients in macro-band 2 cancels c^4."),
        (110, "signai_wpd_db4_macroband_3_kurtosis", "Kurtosis of WPD coefficients in macro-band 3 cancels c^4."),
        (111, "signai_wpd_db4_macroband_4_kurtosis", "Kurtosis of WPD coefficients in macro-band 4 cancels c^4."),
        (112, "signai_wpd_db4_macroband_skewness_mean", "Mean skewness across WPD macro-bands cancels c^3."),
        (113, "signai_wpd_db4_max_subband_energy_ratio", "Maximum relative sub-band energy max_k (E_k / E_total) cancels c^2."),
    ]
    for idx, name, just in wpd_extra_invariant:
        catalog.append(
            FeatureScaleInvarianceItem(
                idx, name, "WPD (SignalAI)", "src/features/signalai_wrapper.py:L37",
                True, "f(c*x) = f(x)", just
            )
        )
    catalog.append(
        FeatureScaleInvarianceItem(
            114, "signai_wpd_db4_total_energy", "WPD (SignalAI)", "src/features/signalai_wrapper.py:L37",
            False, "f(c*x) = c^2 * f(x)", "Total WPD energy sum_k ||d_{6,k}||_2^2 scales with c^2."
        )
    )

    # --- 115..140: SignalAI EMD features (26 features: all 26 invariant) ---
    # By homogeneity of extremal cubic spline envelopes in sifting, IMF_m(c*x) = c * IMF_m(x).
    for imf_k in range(1, 7):
        catalog.append(
            FeatureScaleInvarianceItem(
                114 + imf_k,
                f"signai_emd_imf{imf_k}_rel_energy",
                "EMD (SignalAI)",
                "src/features/signalai_wrapper.py:L37",
                True,
                "f(c*x) = f(x)",
                f"Relative energy ||IMF_{imf_k}||_2^2 / sum_m ||IMF_m||_2^2 cancels c^2 because IMF_m(c*x) = c*IMF_m(x).",
            )
        )
    for imf_k in range(1, 7):
        catalog.append(
            FeatureScaleInvarianceItem(
                120 + imf_k,
                f"signai_emd_imf{imf_k}_corr_with_signal",
                "EMD (SignalAI)",
                "src/features/signalai_wrapper.py:L37",
                True,
                "f(c*x) = f(x)",
                f"Pearson correlation corr(c*IMF_{imf_k}, c*x) cancels c^2 in covariance and product of stds.",
            )
        )
    for imf_k in range(1, 7):
        catalog.append(
            FeatureScaleInvarianceItem(
                126 + imf_k,
                f"signai_emd_imf{imf_k}_kurtosis",
                "EMD (SignalAI)",
                "src/features/signalai_wrapper.py:L37",
                True,
                "f(c*x) = f(x)",
                f"Kurtosis of IMF_{imf_k}(c*x) = c*IMF_{imf_k}(x) is homogeneous of degree 0.",
            )
        )
    for imf_k in range(1, 7):
        catalog.append(
            FeatureScaleInvarianceItem(
                132 + imf_k,
                f"signai_emd_imf{imf_k}_skewness",
                "EMD (SignalAI)",
                "src/features/signalai_wrapper.py:L37",
                True,
                "f(c*x) = f(x)",
                f"Skewness of IMF_{imf_k}(c*x) = c*IMF_{imf_k}(x) is homogeneous of degree 0.",
            )
        )
    catalog.append(
        FeatureScaleInvarianceItem(
            139, "signai_emd_energy_entropy", "EMD (SignalAI)", "src/features/signalai_wrapper.py:L37",
            True, "f(c*x) = f(x)", "Shannon entropy of normalized relative IMF energies is invariant to c > 0."
        )
    )
    catalog.append(
        FeatureScaleInvarianceItem(
            140, "signai_emd_imf1_zero_crossing_rate", "EMD (SignalAI)", "src/features/signalai_wrapper.py:L37",
            True, "f(c*x) = f(x)", "Zero-crossing rate of IMF_1(c*x) = c*IMF_1(x) is invariant for c > 0."
        )
    )

    assert len(catalog) == 141
    assert sum(1 for item in catalog if item.is_scale_invariant) == 126
    assert sum(1 for item in catalog if not item.is_scale_invariant) == 15
    return catalog


def get_scale_invariant_126_indices() -> Tuple[int, ...]:
    """Return the exact 126 feature indices (out of 0..140) that are invariant to c > 0."""
    return tuple(
        item.index
        for item in get_141_feature_scale_invariance_catalog()
        if item.is_scale_invariant
    )


def get_scale_sensitive_15_indices() -> Tuple[int, ...]:
    """Return the exact 15 feature indices (out of 0..140) that depend on signal amplitude scale."""
    return tuple(
        item.index
        for item in get_141_feature_scale_invariance_catalog()
        if not item.is_scale_invariant
    )


def select_feature_subset_for_ablation(
    X_141: np.ndarray,
    mode: str = "full_141",
    expected_checkpoint_input_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Project a `(N, 141)` feature matrix to either `"full_141"` (main configuration)
    or `"invariant_126"` (scale-invariant ablation, Section 5).

    Enforces Section 5: "Manter as 141 features como configuração principal, garantindo
    compatibilidade com o encoder TabNet existente. Preparar uma possível ablação com
    126 features, mas sem reutilizar indevidamente o checkpoint de 141 entradas."
    """
    arr = np.asarray(X_141, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 141:
        raise ValueError(f"Expected input feature matrix of shape (N, 141), got {arr.shape}.")

    if mode == "full_141":
        out = arr
    elif mode == "invariant_126":
        idx_126 = get_scale_invariant_126_indices()
        out = arr[:, idx_126]
    else:
        raise ValueError(f"Unsupported feature subset mode '{mode}'. Expected 'full_141' or 'invariant_126'.")

    if expected_checkpoint_input_dim is not None and int(expected_checkpoint_input_dim) != out.shape[1]:
        raise ValueError(
            f"Checkpoint input dimension mismatch: checkpoint expects {expected_checkpoint_input_dim} "
            f"features, but mode '{mode}' provides {out.shape[1]} features. "
            "Never reuse a 141-input TabNet checkpoint on 126-feature ablation inputs!"
        )
    return out


# =============================================================================
# 6. HIERARCHICAL SOURCE BALANCING (NO CROSS-DOMAIN SMOTE, NO VAL/TEST BALANCING)
# =============================================================================


@dataclass(frozen=True)
class HierarchicalBalanceResult:
    """
    Contains hierarchical source-domain and class-within-domain balancing weights
    and/or resampled indices, along with full distribution audit metadata (Section 9).
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
    class_ids: Sequence[int | str],
    mode: str = "hierarchical_weights",
    seed: int = 42,
    partition_name: str = "train",
) -> HierarchicalBalanceResult:
    """
    Implement the hierarchical balancing protocol (Section 9):
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
      5. Rejects any call where `partition_name` is `"val"`, `"validation"`, `"adapt_val"`, or `"test"`.
    """
    if partition_name.strip().lower() in ("val", "validation", "test", "adapt_val", "source_val"):
        raise ValueError(
            f"Balancing is strictly prohibited on partition '{partition_name}' (Section 9.4)."
        )
    if len(datasets) != len(class_ids) or len(datasets) == 0:
        raise ValueError("datasets and class_ids must have the same non-zero length.")
    if mode not in (
        "hierarchical_weights",
        "hierarchical_class_equalized",
        "hierarchical_resample",
        "none",
    ):
        raise ValueError(
            f"Unsupported hierarchical balance mode '{mode}'. "
            "Expected 'hierarchical_weights', 'hierarchical_class_equalized', "
            "'hierarchical_resample', or 'none'."
        )

    def _resolve_cls_name(c_val: int | str) -> str:
        if isinstance(c_val, (int, np.integer)):
            return ID_TO_HARMONIZED_CLASS.get(int(c_val), str(c_val))
        c_str = str(c_val)
        if c_str.isdigit() and int(c_str) in ID_TO_HARMONIZED_CLASS:
            return ID_TO_HARMONIZED_CLASS[int(c_str)]
        return c_str

    n_total = len(datasets)
    counts_by_ds_cls: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    indices_by_ds_cls: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    resolved_cls_names = [_resolve_cls_name(c) for c in class_ids]

    for idx, (ds, c_name) in enumerate(zip(datasets, resolved_cls_names)):
        ds_str = str(ds)
        counts_by_ds_cls[ds_str][c_name] += 1
        indices_by_ds_cls[(ds_str, c_name)].append(idx)

    unique_datasets = sorted(counts_by_ds_cls.keys())
    n_datasets = len(unique_datasets)

    dist_before: Dict[str, Dict[str, int]] = {
        ds: dict(sorted(counts_by_ds_cls[ds].items()))
        for ds in unique_datasets
    }

    if mode == "none":
        weights = np.ones(n_total, dtype=np.float64)
        resampled_idx = np.arange(n_total, dtype=np.int64)
    elif mode == "hierarchical_weights":
        weights = np.zeros(n_total, dtype=np.float64)
        for idx, (ds, c_name) in enumerate(zip(datasets, resolved_cls_names)):
            ds_str = str(ds)
            n_cls_in_ds = len(counts_by_ds_cls[ds_str])
            n_dc = counts_by_ds_cls[ds_str][c_name]
            weights[idx] = float(n_total) / float(n_datasets * n_cls_in_ds * n_dc)
        resampled_idx = np.arange(n_total, dtype=np.int64)
    elif mode == "hierarchical_class_equalized":
        # Equalize total weight across every canonical class (`N_total / |C_global|`),
        # and within each canonical class split weight equally across the datasets
        # that contain that class (`1 / |D_c|`). Prevents classes present in only one
        # dataset (e.g., `Inner & Outer` in S4) from being down-weighted by 3.5x.
        datasets_by_class: Dict[str, set[str]] = defaultdict(set)
        for ds_str, cls_counts in counts_by_ds_cls.items():
            for c_name in cls_counts:
                datasets_by_class[c_name].add(ds_str)
        n_global_classes = len(datasets_by_class)
        weights = np.zeros(n_total, dtype=np.float64)
        for idx, (ds, c_name) in enumerate(zip(datasets, resolved_cls_names)):
            ds_str = str(ds)
            n_ds_for_cls = len(datasets_by_class[c_name])
            n_dc = counts_by_ds_cls[ds_str][c_name]
            weights[idx] = float(n_total) / float(
                n_global_classes * n_ds_for_cls * n_dc
            )
        resampled_idx = np.arange(n_total, dtype=np.int64)
    else:  # hierarchical_resample
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
            for c_name in present_c:
                cell_pool = sorted(indices_by_ds_cls[(ds, c_name)])
                full_reps, rem = divmod(per_cell_quota, len(cell_pool))
                cell_chosen = cell_pool * full_reps + rng.sample(cell_pool, rem)
                sampled.extend(cell_chosen)

        resampled_idx = np.asarray(sampled, dtype=np.int64)
        weights = np.ones(len(resampled_idx), dtype=np.float64)

    eff_ds_weight: Dict[str, float] = {ds: 0.0 for ds in unique_datasets}
    eff_ds_cls_weight: Dict[str, Dict[str, float]] = {
        ds: {c: 0.0 for c in sorted(counts_by_ds_cls[ds].keys())}
        for ds in unique_datasets
    }
    dist_after: Dict[str, Dict[str, int]] = {
        ds: {c: 0 for c in sorted(counts_by_ds_cls[ds].keys())}
        for ds in unique_datasets
    }

    for pos, orig_idx in enumerate(resampled_idx):
        ds_str = str(datasets[orig_idx])
        c_name = resolved_cls_names[orig_idx]
        w_val = (
            float(weights[pos])
            if len(weights) == len(resampled_idx)
            else float(weights[orig_idx])
        )
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
# 7. SELF-SUPERVISED TABNET PRETRAINING (141 MAIN & 126 SCALE-INVARIANT ABLATION)
#    AND VARIABLE-SPACE CLASSIFICATION HEADS / DA HEAD EXPANSION
# =============================================================================


@dataclass(frozen=True)
class SelfSupervisedPretrainingInputs:
    """
    Leak-free self-supervised pretraining package for `TabNetPretrainer` (Pending Item 2):
      - Built exclusively from `pretrain_train = sorted(source_train + pretrain_only_train)`
        of the active source configuration (`S1`, `S2`, `S3`, or `S4`).
      - Zero validation recordings (`source_val` and `pretrain_only_val`), zero test
        recordings, zero target domain data, and zero class labels are exposed to
        `TabNetPretrainer`.
      - Supports both `"full_141"` (main configuration, `input_dim=141`) and
        `"invariant_126"` (scale-invariant ablation, `input_dim=126`, with its own
        dedicated `TabNetPretrainer`).
    """

    source_config_id: str
    active_sources: Tuple[str, ...]
    feature_mode: str
    input_dim: int
    X_pretrain_unlabeled: np.ndarray
    pretrain_recording_keys: Tuple[str, ...]
    pretrain_sample_ids: Tuple[str, ...]
    supervised_train_recording_count: int
    supervised_train_window_count: int
    additional_unsupervised_recording_count: int
    additional_unsupervised_window_count: int
    additional_unsupervised_classes_by_source: Dict[str, List[str]]
    additional_unsupervised_counts_by_class: Dict[str, Dict[str, int]]
    excluded_validation_recording_keys: Tuple[str, ...]
    labels_exposed_to_pretrainer: bool = False

    def to_audit_dict(self) -> Dict[str, Any]:
        return {
            "source_config_id": self.source_config_id,
            "active_sources": list(self.active_sources),
            "feature_mode": self.feature_mode,
            "input_dim": int(self.input_dim),
            "n_pretrain_windows": int(self.X_pretrain_unlabeled.shape[0]),
            "n_pretrain_recordings": len(self.pretrain_recording_keys),
            "supervised_train_recording_count": int(self.supervised_train_recording_count),
            "supervised_train_window_count": int(self.supervised_train_window_count),
            "additional_unsupervised_recording_count": int(self.additional_unsupervised_recording_count),
            "additional_unsupervised_window_count": int(self.additional_unsupervised_window_count),
            "additional_unsupervised_classes_by_source": self.additional_unsupervised_classes_by_source,
            "additional_unsupervised_counts_by_class": self.additional_unsupervised_counts_by_class,
            "excluded_validation_recording_count": len(self.excluded_validation_recording_keys),
            "labels_exposed_to_pretrainer": bool(self.labels_exposed_to_pretrainer),
        }


def prepare_self_supervised_pretraining_inputs(
    records: Sequence[WindowRecord],
    partitions: Mapping[str, Sequence[int]],
    X_all_141: np.ndarray,
    source_config_id: str,
    feature_mode: str = "full_141",
    scaler: Optional[StandardScaler] = None,
) -> SelfSupervisedPretrainingInputs:
    """
    Construct the unlabeled feature matrix and provenance audit log for `TabNetPretrainer`
    (Pending Item 2 & Pending Item 1):
      1. Requires `records` and `partitions` produced by `create_source_recording_split(...)`
         prior to pretraining.
      2. Verifies that every record in `pretrain_train` belongs strictly to the active
         sources of `source_config_id` (`SOURCE_CONFIGS[source_config_id]`) — never target
         datasets or inactive sources.
      3. Verifies zero recording and sample overlap between `pretrain_train` and
         `source_val` / `pretrain_only_val`.
      4. Projects features according to `feature_mode` (`"full_141"` -> 141 features;
         `"invariant_126"` -> 126 scale-invariant features).
      5. Never passes or stores labels for `TabNetPretrainer`.
    """
    if source_config_id not in SOURCE_CONFIGS:
        raise KeyError(f"Unknown source_config_id '{source_config_id}'.")
    active_sources = SOURCE_CONFIGS[source_config_id]
    active_set = set(active_sources)

    arr_141 = np.asarray(X_all_141, dtype=np.float32)
    if arr_141.ndim != 2 or arr_141.shape[0] != len(records) or arr_141.shape[1] != 141:
        raise ValueError(
            f"X_all_141 shape {arr_141.shape} must match (len(records)={len(records)}, 141)."
        )

    pretrain_idxs = tuple(int(i) for i in partitions.get("pretrain_train", ()))
    sup_train_idxs = set(int(i) for i in partitions.get("source_train", ()))
    unsup_train_idxs = set(int(i) for i in partitions.get("pretrain_only_train", ()))
    sup_val_idxs = set(int(i) for i in partitions.get("source_val", ()))
    unsup_val_idxs = set(int(i) for i in partitions.get("pretrain_only_val", ()))
    all_val_idxs = sup_val_idxs | unsup_val_idxs

    if not pretrain_idxs:
        raise ValueError("pretrain_train partition cannot be empty.")
    if set(pretrain_idxs) != (sup_train_idxs | unsup_train_idxs):
        raise ValueError(
            "pretrain_train must equal the disjoint union of source_train and pretrain_only_train."
        )
    if set(pretrain_idxs) & all_val_idxs:
        raise ValueError("Validation window leakage detected in pretrain_train!")

    pretrain_recs = {records[i].recording_key for i in pretrain_idxs}
    val_recs = {records[i].recording_key for i in all_val_idxs}
    if pretrain_recs & val_recs:
        raise ValueError(
            f"Validation recording leakage detected in pretrain_train: {sorted(pretrain_recs & val_recs)}"
        )

    non_active_datasets = {
        records[i].dataset for i in pretrain_idxs if records[i].dataset not in active_set
    }
    if non_active_datasets:
        raise ValueError(
            f"Non-active or target dataset(s) detected in pretrain_train for {source_config_id}: "
            f"{sorted(non_active_datasets)}"
        )

    X_selected = select_feature_subset_for_ablation(arr_141, mode=feature_mode)
    if scaler is not None:
        X_pre = scaler.transform(X_selected[list(pretrain_idxs)]).astype(np.float32)
    else:
        X_pre = X_selected[list(pretrain_idxs)].astype(np.float32)

    # Trim single remainder sample if len(X_pre) % batch_size == 1 is handled at fit time
    add_classes_by_src: Dict[str, List[str]] = defaultdict(list)
    add_counts_by_cls: Dict[str, Dict[str, int]] = {}
    for s_name in active_sources:
        raw_unsup_classes = sorted(
            {
                records[i].raw_class
                for i in range(len(records))
                if records[i].dataset == s_name and records[i].harmonized_class_id is None
            }
        )
        add_classes_by_src[s_name] = raw_unsup_classes
        for raw_c in raw_unsup_classes:
            key = f"{s_name}::{raw_c}"
            tr_w = [
                i
                for i in unsup_train_idxs
                if records[i].dataset == s_name and records[i].raw_class == raw_c
            ]
            val_w = [
                i
                for i in unsup_val_idxs
                if records[i].dataset == s_name and records[i].raw_class == raw_c
            ]
            add_counts_by_cls[key] = {
                "pretrain_train_recordings": len({records[i].recording_key for i in tr_w}),
                "pretrain_train_windows": len(tr_w),
                "excluded_val_recordings": len({records[i].recording_key for i in val_w}),
                "excluded_val_windows": len(val_w),
            }

    return SelfSupervisedPretrainingInputs(
        source_config_id=source_config_id,
        active_sources=tuple(active_sources),
        feature_mode=feature_mode,
        input_dim=int(X_pre.shape[1]),
        X_pretrain_unlabeled=X_pre,
        pretrain_recording_keys=tuple(sorted(pretrain_recs)),
        pretrain_sample_ids=tuple(records[i].sample_id for i in pretrain_idxs),
        supervised_train_recording_count=len({records[i].recording_key for i in sup_train_idxs}),
        supervised_train_window_count=len(sup_train_idxs),
        additional_unsupervised_recording_count=len({records[i].recording_key for i in unsup_train_idxs}),
        additional_unsupervised_window_count=len(unsup_train_idxs),
        additional_unsupervised_classes_by_source=dict(add_classes_by_src),
        additional_unsupervised_counts_by_class=add_counts_by_cls,
        excluded_validation_recording_keys=tuple(sorted(val_recs)),
        labels_exposed_to_pretrainer=False,
    )


def fit_self_supervised_tabnet_pretrainer(
    pretrain_inputs: SelfSupervisedPretrainingInputs,
    seed: int = 42,
    max_epochs: int = 30,
    batch_size: int = 64,
    pretraining_ratio: float = 0.5,
    tabnet_params: Optional[Mapping[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Instantiate and fit a dedicated `TabNetPretrainer` (`141` features for `"full_141"`
    or `126` features for `"invariant_126"`, never sharing checkpoints across dimensions)
    using ONLY `pretrain_inputs.X_pretrain_unlabeled` (zero labels, zero validation
    recordings, zero target data).
    """
    from pytorch_tabnet.pretraining import TabNetPretrainer
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    hp = dict(tabnet_params or {})
    pretrainer = TabNetPretrainer(
        n_d=int(hp.get("n_d", 16)),
        n_a=int(hp.get("n_a", 16)),
        n_steps=int(hp.get("n_steps", 4)),
        gamma=float(hp.get("gamma", 1.3)),
        n_independent=int(hp.get("n_independent", 2)),
        n_shared=int(hp.get("n_shared", 2)),
        mask_type=str(hp.get("mask_type", "sparsemax")),
        seed=int(seed),
        verbose=0,
    )
    X_fit = np.asarray(pretrain_inputs.X_pretrain_unlabeled, dtype=np.float32)
    eff_bs = min(int(batch_size), max(2, len(X_fit)))
    trimmed = False
    if len(X_fit) > 1 and (len(X_fit) % eff_bs) == 1:
        X_fit = X_fit[:-1]
        trimmed = True
    trim_meta = {
        "trimmed_single_remainder": trimmed,
        "original_len": int(len(pretrain_inputs.X_pretrain_unlabeled)),
        "effective_len": int(len(X_fit)),
        "batch_size": int(eff_bs),
    }
    eff_vbs = min(eff_bs, max(2, len(X_fit) // 2))
    pretrainer.fit(
        X_train=X_fit,
        max_epochs=int(max_epochs),
        batch_size=eff_bs,
        virtual_batch_size=eff_vbs,
        pretraining_ratio=float(pretraining_ratio),
        drop_last=False,
    )
    audit = pretrain_inputs.to_audit_dict()
    audit["input_dim_verified"] = int(pretrainer.network.input_dim)
    audit["batch_trim_metadata"] = trim_meta
    if int(pretrainer.network.input_dim) != int(pretrain_inputs.input_dim):
        raise AssertionError(
            f"TabNetPretrainer input_dim ({pretrainer.network.input_dim}) != "
            f"expected ({pretrain_inputs.input_dim})"
        )
    return pretrainer, audit



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
        x_t = torch.tensor(
            np.asarray(X_scaled, dtype=np.float32), dtype=torch.float32, device=device
        )
        embedded = embedder(x_t)
        steps_out, _ = encoder(embedded)
        latent = torch.sum(torch.stack(steps_out, dim=0), dim=0)
        return latent.detach().cpu().numpy().astype(np.float32)


def _snapshot_torch_module(
    module: Any,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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


def _normalize_labels_to_class_names(
    y: Sequence[int | str],
    label_space: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Convert a sequence of labels `y` (which may be canonical class names `"Normal"`,
    global harmonized IDs `0..4`, or local indices `0..K-1` when `label_space` is given)
    into canonical harmonized class names.
    """
    out: List[str] = []
    for item in y:
        if isinstance(item, str):
            if item not in HARMONIZED_CLASS_TO_ID:
                raise ValueError(f"Unknown class name '{item}'.")
            out.append(item)
        elif isinstance(item, (int, np.integer)):
            idx = int(item)
            if label_space is not None and 0 <= idx < len(label_space):
                out.append(str(label_space[idx]))
            elif idx in ID_TO_HARMONIZED_CLASS:
                out.append(ID_TO_HARMONIZED_CLASS[idx])
            else:
                raise ValueError(
                    f"Integer label {idx} cannot be mapped to a class name."
                )
        else:
            raise TypeError(f"Unsupported label type: {type(item)}")
    return out


@dataclass
class VariableClassHeadCheckpoint:
    """
    Stores a classification head (`nn.Linear(latent_dim, len(class_names))`) with
    an explicit, variable class output space (`class_names`), per Section 3:
      - `S1` (`UORED`): `class_names = ("Normal", "Ball")`, `num_classes = 2`
      - `S2` (`HUST`): `class_names = ("Normal", "Ball", "Inner", "Outer", "Inner & Outer")`, `num_classes = 5`
      - `S3` (`MFPT`): `class_names = ("Inner", "Outer")`, `num_classes = 2`
      - `S4` (`Combined`): `class_names = ("Normal", "Ball", "Inner", "Outer", "Inner & Outer")`, `num_classes = 5`

    Also tracks provenance of each class for DA expansion (Section 3.2):
      - `transferred_from_source_classes`: classes whose initial weights/biases came from the source head
      - `introduced_by_expansion_classes`: new classes added when expanding the head for a DA target
      - `learned_during_adaptation_classes`: classes with training examples during target DA adaptation
    """

    latent_dim: int
    class_names: Tuple[str, ...]
    head_state_dict: Dict[str, Any]
    source_config_id: str
    best_val_loss: float
    selected_epoch: int
    encoder_param_max_abs_diff: float
    encoder_buffer_max_abs_diff: float
    head_param_max_abs_diff: float
    transferred_from_source_classes: Tuple[str, ...] = ()
    introduced_by_expansion_classes: Tuple[str, ...] = ()
    learned_during_adaptation_classes: Tuple[str, ...] = ()
    target_dataset: Optional[str] = None

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def class_to_index(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.class_names)}

    @property
    def index_to_class(self) -> Dict[int, str]:
        return {idx: name for idx, name in enumerate(self.class_names)}

    @property
    def observed_class_ids(self) -> Tuple[int, ...]:
        return tuple(HARMONIZED_CLASS_TO_ID[c] for c in self.class_names)


# Alias for backward compatibility
FiveClassHeadCheckpoint = VariableClassHeadCheckpoint


def expand_head_for_da_target(
    source_head_checkpoint: VariableClassHeadCheckpoint,
    target_present_classes: Sequence[int | str],
    seed: int,
    target_dataset: Optional[str] = None,
) -> VariableClassHeadCheckpoint:
    """
    Create a target-specific expanded classification head for Domain Adaptation (Section 3.2):
      1. Determine the expanded class space as the canonical union of the source head's
         classes and the classes present in the target adaptation partition.
      2. Match classes strictly by canonical class name (`"Normal"`, `"Ball"`, `"Inner"`,
         `"Outer"`, `"Inner & Outer"`), NEVER by assumed positional index.
      3. For every shared class `c in source_classes`, copy its exact weight row
         `weight[src_idx, :]` and bias `bias[src_idx]` into `expanded_weight[new_idx, :]`
         and `expanded_bias[new_idx]`.
      4. For every new class `c in target_classes \\ source_classes`, initialize its weight
         row and bias reproducibly using a dedicated `torch.Generator` seeded with
         `(seed, canonical_class_name)`.
      5. Record `transferred_from_source_classes` and `introduced_by_expansion_classes`.
    """
    import torch
    import torch.nn as nn

    target_cls_names = canonicalize_class_sequence(
        _normalize_labels_to_class_names(target_present_classes)
    )
    source_cls_names = tuple(source_head_checkpoint.class_names)
    expanded_cls_names = canonicalize_class_sequence(
        list(source_cls_names) + list(target_cls_names)
    )

    latent_dim = int(source_head_checkpoint.latent_dim)
    src_weight = source_head_checkpoint.head_state_dict["weight"].detach().cpu().clone()
    src_bias = source_head_checkpoint.head_state_dict["bias"].detach().cpu().clone()

    if src_weight.shape != (len(source_cls_names), latent_dim):
        raise ValueError(
            f"Source head weight shape {tuple(src_weight.shape)} does not match "
            f"({len(source_cls_names)}, {latent_dim})."
        )

    new_head = nn.Linear(latent_dim, len(expanded_cls_names))
    new_weight = torch.empty(
        (len(expanded_cls_names), latent_dim), dtype=src_weight.dtype
    )
    new_bias = torch.empty((len(expanded_cls_names),), dtype=src_bias.dtype)

    src_name_to_idx = source_head_checkpoint.class_to_index
    bound = 1.0 / math.sqrt(float(latent_dim))

    transferred: List[str] = []
    introduced: List[str] = []

    for new_idx, cls_name in enumerate(expanded_cls_names):
        if cls_name in src_name_to_idx:
            old_idx = src_name_to_idx[cls_name]
            new_weight[new_idx, :] = src_weight[old_idx, :]
            new_bias[new_idx] = src_bias[old_idx]
            transferred.append(cls_name)
        else:
            # Deterministic per-(seed, class_name) Kaiming-uniform initialization
            token = f"DA_EXPAND::{int(seed)}::{cls_name}"
            cls_seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(cls_seed)
            w_row = torch.empty((1, latent_dim), dtype=src_weight.dtype).uniform_(
                -bound, bound, generator=gen
            )
            b_val = torch.empty((1,), dtype=src_bias.dtype).uniform_(
                -bound, bound, generator=gen
            )
            new_weight[new_idx, :] = w_row[0]
            new_bias[new_idx] = b_val[0]
            introduced.append(cls_name)

    with torch.no_grad():
        new_head.weight.copy_(new_weight)
        new_head.bias.copy_(new_bias)

    return VariableClassHeadCheckpoint(
        latent_dim=latent_dim,
        class_names=expanded_cls_names,
        head_state_dict=copy.deepcopy(new_head.state_dict()),
        source_config_id=source_head_checkpoint.source_config_id,
        best_val_loss=float("inf"),
        selected_epoch=0,
        encoder_param_max_abs_diff=0.0,
        encoder_buffer_max_abs_diff=0.0,
        head_param_max_abs_diff=0.0,
        transferred_from_source_classes=tuple(transferred),
        introduced_by_expansion_classes=tuple(introduced),
        learned_during_adaptation_classes=(),
        target_dataset=target_dataset,
    )


def train_variable_class_head_with_frozen_tabnet_encoder(
    pretrained_tabnet: Any,
    X_train_scaled: np.ndarray,
    y_train: Sequence[int | str],
    X_val_scaled: Optional[np.ndarray] = None,
    y_val: Optional[Sequence[int | str]] = None,
    seed: int = 42,
    sample_weights: Optional[np.ndarray] = None,
    head_class_names: Optional[Sequence[str]] = None,
    init_head_checkpoint: Optional[VariableClassHeadCheckpoint] = None,
    expand_for_target_classes: bool = True,
    target_dataset: Optional[str] = None,
    source_config_id: str = "S4",
    max_epochs: int = 40,
    patience: int = 10,
    fixed_epochs: Optional[int] = None,
    batch_size: int = 64,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> VariableClassHeadCheckpoint:
    """
    Train a source classification head (for DG / initial DA) or adapt an expanded
    target classification head (for DA) on top of a completely frozen `TabNet` encoder:
      - Never forces 5 outputs when training a source head for `S1` (2 outputs: `Normal`,
        `Ball`) or `S3` (2 outputs: `Inner`, `Outer`). The head output dimension is strictly
        `len(active_class_names)`.
      - When `init_head_checkpoint` is provided (DA mode):
        * If `expand_for_target_classes=True`, automatically expands `init_head_checkpoint`
          via `expand_head_for_da_target` to include any new classes present in `y_train`
          (preserving shared class weights/biases by canonical name and initializing new
          outputs reproducibly with `seed`).
        * Adapts only the expanded head on `(X_train_scaled, y_train)`.
        * Records `transferred_from_source_classes`, `introduced_by_expansion_classes`,
          and `learned_during_adaptation_classes`.
      - Freezes all parameters and GhostBatchNorm buffers of `embedder` and `encoder`
        (`requires_grad=False`, `.eval()`) and asserts `encoder_param_max_abs_diff == 0.0`
        and `encoder_buffer_max_abs_diff == 0.0`.
      - Asserts `head_param_max_abs_diff > 0.0`.
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

    with torch.no_grad():
        probe_x = torch.tensor(
            np.asarray(X_train_scaled[:2], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        probe_steps, _ = encoder(embedder(probe_x))
        latent_dim = int(probe_steps[0].shape[1])

    y_train_names = _normalize_labels_to_class_names(y_train, label_space=head_class_names)
    train_unique_names = canonicalize_class_sequence(y_train_names)

    if init_head_checkpoint is not None:
        if expand_for_target_classes:
            working_init = expand_head_for_da_target(
                source_head_checkpoint=init_head_checkpoint,
                target_present_classes=train_unique_names,
                seed=seed,
                target_dataset=target_dataset,
            )
        else:
            working_init = init_head_checkpoint
        active_class_names = working_init.class_names
        transferred_classes = (
            working_init.transferred_from_source_classes
            if working_init.transferred_from_source_classes
            else tuple(init_head_checkpoint.class_names)
        )
        introduced_classes = working_init.introduced_by_expansion_classes
        learned_da_classes = train_unique_names
    else:
        if head_class_names is not None:
            active_class_names = canonicalize_class_sequence(head_class_names)
        elif source_config_id in SOURCE_CONFIG_AVAILABLE_CLASSES:
            expected_cfg_classes = SOURCE_CONFIG_AVAILABLE_CLASSES[source_config_id]
            if set(train_unique_names).issubset(set(expected_cfg_classes)):
                active_class_names = expected_cfg_classes
            else:
                active_class_names = train_unique_names
        else:
            active_class_names = train_unique_names
        working_init = None
        transferred_classes = active_class_names
        introduced_classes = ()
        learned_da_classes = ()

    missing_in_head = set(train_unique_names) - set(active_class_names)
    if missing_in_head:
        raise ValueError(
            f"Training data contains classes {sorted(missing_in_head)} not present in "
            f"head class space {active_class_names}."
        )

    class_to_local_idx = {name: idx for idx, name in enumerate(active_class_names)}
    y_train_local = np.asarray(
        [class_to_local_idx[name] for name in y_train_names], dtype=np.int64
    )

    has_val = (
        X_val_scaled is not None
        and y_val is not None
        and len(X_val_scaled) > 0
        and len(y_val) > 0
    )
    if has_val:
        assert X_val_scaled is not None and y_val is not None
        y_val_names = _normalize_labels_to_class_names(y_val, label_space=head_class_names)
        # Filter or map validation samples within the active head class space
        val_keep_mask = [name in class_to_local_idx for name in y_val_names]
        X_val_arr = np.asarray(X_val_scaled, dtype=np.float32)[val_keep_mask]
        y_val_local = np.asarray(
            [class_to_local_idx[name] for name in y_val_names if name in class_to_local_idx],
            dtype=np.int64,
        )
        has_val = len(X_val_arr) > 0
    else:
        X_val_arr = np.empty((0, X_train_scaled.shape[1]), dtype=np.float32)
        y_val_local = np.empty((0,), dtype=np.int64)

    torch.manual_seed(seed)
    head = nn.Linear(latent_dim, len(active_class_names)).to(device)
    if working_init is not None:
        head.load_state_dict(copy.deepcopy(working_init.head_state_dict))

    for p in head.parameters():
        p.requires_grad = True
    head_p_before, _ = _snapshot_torch_module(head)

    if sample_weights is not None:
        w_arr = np.asarray(sample_weights, dtype=np.float32)
        if len(w_arr) != len(X_train_scaled):
            raise ValueError("sample_weights length must match X_train_scaled.")
    else:
        w_arr = np.ones(len(X_train_scaled), dtype=np.float32)

    ds_train = TensorDataset(
        torch.tensor(np.asarray(X_train_scaled, dtype=np.float32), dtype=torch.float32),
        torch.tensor(y_train_local, dtype=torch.long),
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

    X_val_t = torch.tensor(X_val_arr, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val_local, dtype=torch.long, device=device)

    total_epochs = int(fixed_epochs) if fixed_epochs is not None else int(max_epochs)
    best_val_loss = float("inf")
    best_epoch = 1
    best_head_state = copy.deepcopy(head.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, total_epochs + 1):
        embedder.eval()
        encoder.eval()
        head.train()
        last_train_loss = 0.0
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
            per_sample_loss = criterion_none(logits, by)
            loss = torch.sum(per_sample_loss * bw) / torch.clamp(torch.sum(bw), min=1e-8)
            loss.backward()
            optimizer.step()
            last_train_loss = float(loss.item())

        embedder.eval()
        encoder.eval()
        head.eval()
        if fixed_epochs is not None or not has_val:
            best_val_loss = last_train_loss
            best_epoch = epoch
            best_head_state = copy.deepcopy(head.state_dict())
        else:
            with torch.no_grad():
                val_steps, _ = encoder(embedder(X_val_t))
                val_latent = torch.sum(torch.stack(val_steps, dim=0), dim=0)
                val_logits = head(val_latent)
                val_loss = float(torch.mean(criterion_none(val_logits, y_val_t)).item())

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
            "Classification head parameters were not updated during training!"
        )

    return VariableClassHeadCheckpoint(
        latent_dim=latent_dim,
        class_names=active_class_names,
        head_state_dict={k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        source_config_id=source_config_id,
        best_val_loss=float(best_val_loss),
        selected_epoch=int(best_epoch),
        encoder_param_max_abs_diff=float(enc_param_diff),
        encoder_buffer_max_abs_diff=float(enc_buf_diff),
        head_param_max_abs_diff=float(head_param_diff),
        transferred_from_source_classes=tuple(transferred_classes),
        introduced_by_expansion_classes=tuple(introduced_classes),
        learned_during_adaptation_classes=tuple(learned_da_classes),
        target_dataset=target_dataset,
    )


# Backward-compatible alias
train_five_class_head_with_frozen_tabnet_encoder = (
    train_variable_class_head_with_frozen_tabnet_encoder
)


def predict_with_variable_class_head(
    pretrained_tabnet: Any,
    head_checkpoint: VariableClassHeadCheckpoint,
    X_test_scaled: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Evaluate `X_test_scaled` using the frozen TabNet encoder and the variable-space
    classification head (`head_checkpoint`, with `K = len(head_checkpoint.class_names)`
    outputs).

    Returns:
      - `y_pred_canonical_ids`: integer predictions mapped to canonical harmonized IDs
        (`0: Normal, 1: Ball, 2: Inner, 3: Outer, 4: Inner & Outer`), shape `(N,)`
      - `y_proba_head`: raw softmax probability matrix over the head's exact `K` outputs,
        shape `(N, K)`, where column `j` corresponds to `head_checkpoint.class_names[j]`
      - `y_pred_class_names`: predicted canonical class names, length `N`
    """
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_net = pretrained_tabnet.network.to(device)
    embedder = backbone_net.embedder
    encoder = backbone_net.encoder
    embedder.eval()
    encoder.eval()

    k_outputs = head_checkpoint.num_classes
    head = nn.Linear(head_checkpoint.latent_dim, k_outputs).to(device)
    head.load_state_dict(copy.deepcopy(head_checkpoint.head_state_dict))
    head.eval()

    with torch.no_grad():
        x_t = torch.tensor(
            np.asarray(X_test_scaled, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        steps_out, _ = encoder(embedder(x_t))
        latent = torch.sum(torch.stack(steps_out, dim=0), dim=0)
        logits = head(latent)
        probs_head = (
            torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
        )
        local_preds = np.argmax(probs_head, axis=1).astype(np.int64)

    pred_names = [head_checkpoint.class_names[int(idx)] for idx in local_preds]
    pred_canonical_ids = np.asarray(
        [HARMONIZED_CLASS_TO_ID[name] for name in pred_names], dtype=np.int64
    )
    return pred_canonical_ids, probs_head, pred_names


def predict_with_five_class_head(
    pretrained_tabnet: Any,
    head_checkpoint: VariableClassHeadCheckpoint,
    X_test_scaled: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper around `predict_with_variable_class_head` returning `(y_pred_canonical_ids, y_proba_head)`,
    where `y_proba_head` has shape `(N, head_checkpoint.num_classes)`.
    """
    preds, probs_head, _ = predict_with_variable_class_head(
        pretrained_tabnet=pretrained_tabnet,
        head_checkpoint=head_checkpoint,
        X_test_scaled=X_test_scaled,
    )
    return preds, probs_head


def evaluate_variable_space_predictions(
    y_true: Sequence[int | str],
    y_pred: Sequence[int | str],
    head_class_names: Optional[Sequence[str]] = None,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Evaluate multiclass predictions with variable source and target class spaces
    (Sections 3.1, 3.2 & 12):
      - Never excludes unknown target test examples (`len(y_true)` is evaluated in full).
      - If a true target class does not exist in the head (`c not in head_class_names`),
        every sample of that class is necessarily misclassified (`TP_c = 0`, `Recall_c = 0.0`,
        `F1_c = 0.0`), reducing `Accuracy`, `Balanced_Accuracy`, and `Macro_F1`.
      - If the head predicts a class that does not exist in the target ground truth
        (`c_pred not in target_present_classes`), it is counted as a classification error.
      - Computes and returns:
        * `macro_f1`: Macro-F1 across all true classes present in the target test set
          (`labels = target_present_classes`), where unknown target classes receive `F1 = 0.0`.
        * `macro_f1_including_phantom_classes`: Macro-F1 over `target_present_classes ∪ predicted_classes`.
        * `accuracy`: overall accuracy across 100% of target test samples.
        * `balanced_accuracy`: mean recall across all `target_present_classes` (where
          unknown target classes have `Recall = 0.0`).
        * `confusion_matrix`: full confusion matrix over `union_classes` (`target ∪ head ∪ predicted`)
          and `confusion_matrix_5class` over the canonical 5 classes.
        * `roc_auc_ovr` and `roc_auc_valid`: computed when all `target_present_classes`
          are covered by `head_class_names` and `len(target_present_classes) >= 2`.
    """
    true_names = _normalize_labels_to_class_names(y_true)
    pred_names = _normalize_labels_to_class_names(y_pred)
    if len(true_names) != len(pred_names) or len(true_names) == 0:
        raise ValueError("y_true and y_pred must have the same non-zero length.")

    y_t = np.asarray([HARMONIZED_CLASS_TO_ID[c] for c in true_names], dtype=np.int64)
    y_p = np.asarray([HARMONIZED_CLASS_TO_ID[c] for c in pred_names], dtype=np.int64)

    target_present_ids = sorted(int(c) for c in np.unique(y_t))
    target_present_names = [ID_TO_HARMONIZED_CLASS[c] for c in target_present_ids]

    if head_class_names is not None:
        resolved_head_names = list(canonicalize_class_sequence(head_class_names))
    else:
        resolved_head_names = [
            ID_TO_HARMONIZED_CLASS[c]
            for c in sorted(int(x) for x in np.unique(y_p))
        ]
    head_class_ids = [HARMONIZED_CLASS_TO_ID[c] for c in resolved_head_names]

    unknown_target_class_names = [
        c for c in target_present_names if c not in resolved_head_names
    ]
    unknown_target_class_ids = [
        HARMONIZED_CLASS_TO_ID[c] for c in unknown_target_class_names
    ]
    unknown_target_sample_mask = np.isin(y_t, unknown_target_class_ids)
    unknown_target_sample_count = int(np.sum(unknown_target_sample_mask))

    active_union_ids = sorted(
        int(c) for c in (set(y_t.tolist()) | set(y_p.tolist()))
    )
    full_eval_ids = sorted(
        int(c) for c in (set(y_t.tolist()) | set(y_p.tolist()) | set(head_class_ids))
    )

    out_of_target_mask = ~np.isin(y_p, target_present_ids)
    out_of_target_count = int(np.sum(out_of_target_mask))
    out_of_target_rate = float(out_of_target_count) / float(len(y_t))

    acc = float(accuracy_score(y_t, y_p))
    per_target_class_recall = {
        ID_TO_HARMONIZED_CLASS[c]: float(np.sum((y_t == c) & (y_p == c)))
        / float(np.sum(y_t == c))
        for c in target_present_ids
    }
    bal_acc = float(np.mean(list(per_target_class_recall.values())))

    macro_f1_present = float(
        f1_score(
            y_t, y_p, labels=target_present_ids, average="macro", zero_division=0
        )
    )
    macro_f1_active_union = float(
        f1_score(
            y_t, y_p, labels=active_union_ids, average="macro", zero_division=0
        )
    )

    per_class_f1_all_5 = {
        ID_TO_HARMONIZED_CLASS[c]: float(
            f1_score(y_t, y_p, labels=[c], average="macro", zero_division=0)
        )
        for c in range(NUM_HARMONIZED_CLASSES)
    }

    cm_eval = confusion_matrix(y_t, y_p, labels=full_eval_ids).tolist()
    cm_5class = confusion_matrix(
        y_t, y_p, labels=list(range(NUM_HARMONIZED_CLASSES))
    ).tolist()

    # Project variable-dim y_proba (N, K) to 5-class probability matrix if needed
    proba_5: Optional[np.ndarray] = None
    if y_proba is not None:
        p_arr = np.asarray(y_proba, dtype=np.float64)
        if p_arr.shape == (len(y_t), len(resolved_head_names)):
            proba_5 = np.zeros((len(y_t), NUM_HARMONIZED_CLASSES), dtype=np.float64)
            for local_col, c_name in enumerate(resolved_head_names):
                proba_5[:, HARMONIZED_CLASS_TO_ID[c_name]] = p_arr[:, local_col]
        elif p_arr.shape == (len(y_t), NUM_HARMONIZED_CLASSES):
            proba_5 = p_arr

    roc_auc_val: Optional[float] = None
    roc_auc_valid = False
    if (
        proba_5 is not None
        and len(target_present_ids) >= 2
        and len(unknown_target_class_ids) == 0
    ):
        sub_probs = np.asarray(proba_5[:, target_present_ids], dtype=np.float64)
        row_sums = np.sum(sub_probs, axis=1, keepdims=True)
        if np.all(np.isfinite(sub_probs)) and np.all(row_sums > 0):
            sub_probs_norm = sub_probs / row_sums
            y_t_contiguous = np.searchsorted(target_present_ids, y_t)
            try:
                if len(target_present_ids) == 2:
                    roc_auc_val = float(
                        roc_auc_score(y_t_contiguous, sub_probs_norm[:, 1])
                    )
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
        "head_classes": resolved_head_names,
        "target_present_classes": target_present_names,
        "unknown_target_classes": unknown_target_class_names,
        "unknown_target_sample_count": unknown_target_sample_count,
        "predicted_classes": [
            ID_TO_HARMONIZED_CLASS[c] for c in sorted(set(y_p.tolist()))
        ],
        "out_of_target_prediction_count": out_of_target_count,
        "out_of_target_prediction_rate": out_of_target_rate,
        "per_target_class_recall": per_target_class_recall,
        "per_class_f1_all_5": per_class_f1_all_5,
        "confusion_matrix_labels": [ID_TO_HARMONIZED_CLASS[c] for c in full_eval_ids],
        "confusion_matrix": cm_eval,
        "confusion_matrix_5class_labels": list(HARMONIZED_CLASSES),
        "confusion_matrix_5class": cm_5class,
        "n_test_samples": int(len(y_t)),
    }


# Backward-compatible alias
evaluate_five_class_predictions = evaluate_variable_space_predictions


# =============================================================================
# 8. CLASSICAL BASELINES (RF & SVM FOR DG AND DA: TARGET-ONLY 20%, SOURCE+TARGET 20%)
# =============================================================================


@dataclass(frozen=True)
class ClassicalBaselineRunResult:
    """
    Encapsulates predictions, probabilities, fitted scaler, and evaluation metrics
    for a classical baseline (`RandomForest` or `SVM`) in DG or DA (Section 10).
    """

    protocol: str
    variant: str
    model_name: str
    head_class_names: Tuple[str, ...]
    y_pred_canonical_ids: np.ndarray
    y_pred_class_names: Tuple[str, ...]
    y_proba_head: np.ndarray
    metrics: Dict[str, Any]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    hyperparameters: Dict[str, Any]
    joint_learning_note: str


def train_eval_classical_da_dg_baseline(
    protocol: str,
    variant: str,
    model_name: str,
    X_test_raw: np.ndarray,
    y_test: Sequence[int | str],
    seed: int = 42,
    X_source_train_raw: Optional[np.ndarray] = None,
    y_source_train: Optional[Sequence[int | str]] = None,
    source_sample_weights: Optional[np.ndarray] = None,
    X_target_adapt_raw: Optional[np.ndarray] = None,
    y_target_adapt: Optional[Sequence[int | str]] = None,
    target_sample_weights: Optional[np.ndarray] = None,
    hyperparameters: Optional[Mapping[str, Any]] = None,
) -> ClassicalBaselineRunResult:
    """
    Train and evaluate the classical baselines required by Section 10:
      - **DG** (`protocol="DG"`, `variant="Source-Only"`):
        * `RandomForest` or `SVM` trained exclusively on source training data.
        * `StandardScaler` fitted exclusively on `X_source_train_raw`.
      - **DA Target-only 20%** (`protocol="DA"`, `variant="Target-Only 20%"`):
        * `RandomForest` or `SVM` trained exclusively on the target's 20% labeled
          adaptation partition (`X_target_adapt_raw`, `y_target_adapt`).
        * `StandardScaler` fitted exclusively on `X_target_adapt_raw`.
      - **DA Source + Target 20%** (`protocol="DA"`, `variant="Source + Target 20%"`):
        * `RandomForest` or `SVM` trained jointly on the pooled source training data
          and the target's 20% adaptation partition (`vstack([X_source_train_raw, X_target_adapt_raw])`).
        * `StandardScaler` fitted exclusively on the combined training set (`source + target 20%`).
        * Explicitly documented as a joint-learning baseline (not equivalent to explicit
          TabNet head transfer/expansion).
      - No variant ever uses `X_test_raw` or `y_test` for training, normalization,
        hyperparameter selection, or adjustment.
      - Both `SVM` (where standardization is mandatory for RBF kernel distances) and
        `RandomForest` (where standardization is kept for strict pipeline uniformity,
        as monotonic linear rescaling does not alter decision tree split thresholds)
        receive the training-only `StandardScaler`.
    """
    proto_norm = protocol.strip().upper()
    model_norm = model_name.strip()
    if model_norm not in ("RandomForest", "SVM"):
        raise ValueError(
            f"Unsupported classical model '{model_name}'. Expected 'RandomForest' or 'SVM'."
        )

    if proto_norm == "DG" or variant == "Source-Only":
        if X_source_train_raw is None or y_source_train is None:
            raise ValueError("DG / Source-Only baseline requires X_source_train_raw and y_source_train.")
        X_tr_raw = np.asarray(X_source_train_raw, dtype=np.float32)
        y_tr_names = _normalize_labels_to_class_names(y_source_train)
        w_tr = (
            np.asarray(source_sample_weights, dtype=np.float64)
            if source_sample_weights is not None
            else None
        )
        note = "Trained exclusively on source domains (zero target adaptation or target statistics)."
    elif proto_norm == "DA" and variant == "Target-Only 20%":
        if X_target_adapt_raw is None or y_target_adapt is None:
            raise ValueError("DA Target-Only 20% baseline requires X_target_adapt_raw and y_target_adapt.")
        X_tr_raw = np.asarray(X_target_adapt_raw, dtype=np.float32)
        y_tr_names = _normalize_labels_to_class_names(y_target_adapt)
        w_tr = (
            np.asarray(target_sample_weights, dtype=np.float64)
            if target_sample_weights is not None
            else None
        )
        note = "Trained exclusively on the 20% labeled target adaptation partition."
    elif proto_norm == "DA" and variant == "Source + Target 20%":
        if (
            X_source_train_raw is None
            or y_source_train is None
            or X_target_adapt_raw is None
            or y_target_adapt is None
        ):
            raise ValueError(
                "DA Source + Target 20% baseline requires both source and 20% target adaptation data."
            )
        X_s = np.asarray(X_source_train_raw, dtype=np.float32)
        X_t = np.asarray(X_target_adapt_raw, dtype=np.float32)
        X_tr_raw = np.vstack([X_s, X_t])
        y_tr_names = _normalize_labels_to_class_names(y_source_train) + _normalize_labels_to_class_names(
            y_target_adapt
        )
        if source_sample_weights is not None or target_sample_weights is not None:
            w_s = (
                np.asarray(source_sample_weights, dtype=np.float64)
                if source_sample_weights is not None
                else np.ones(len(X_s), dtype=np.float64)
            )
            w_t = (
                np.asarray(target_sample_weights, dtype=np.float64)
                if target_sample_weights is not None
                else np.ones(len(X_t), dtype=np.float64)
            )
            w_tr = np.concatenate([w_s, w_t])
        else:
            w_tr = None
        note = (
            "Joint learning baseline combining source training data and 20% labeled target "
            "adaptation data; not equivalent to explicit TabNet head transfer/expansion."
        )
    else:
        raise ValueError(
            f"Unsupported protocol/variant combination: protocol='{protocol}', variant='{variant}'."
        )

    head_class_names = canonicalize_class_sequence(y_tr_names)
    if len(head_class_names) < 2:
        raise ValueError(
            f"Classical baseline '{model_norm}' requires at least 2 distinct training classes, "
            f"got {head_class_names}."
        )
    cls_to_local = {name: idx for idx, name in enumerate(head_class_names)}
    y_tr_local = np.asarray([cls_to_local[name] for name in y_tr_names], dtype=np.int64)

    # Fit StandardScaler strictly on X_tr_raw (zero test data leakage)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_te_scaled = scaler.transform(np.asarray(X_test_raw, dtype=np.float32)).astype(np.float32)

    hp = dict(hyperparameters or {})
    if model_norm == "RandomForest":
        eff_hp = {
            "n_estimators": int(hp.get("n_estimators", 300)),
            "max_depth": hp.get("max_depth", None),
            "min_samples_split": int(hp.get("min_samples_split", 2)),
            "min_samples_leaf": int(hp.get("min_samples_leaf", 1)),
            "max_features": hp.get("max_features", "sqrt"),
            "class_weight": hp.get("class_weight", "balanced" if w_tr is None else None),
            "random_state": int(seed),
            "n_jobs": int(hp.get("n_jobs", -1)),
        }
        clf = RandomForestClassifier(**eff_hp)
    else:  # SVM
        eff_hp = {
            "kernel": str(hp.get("kernel", "rbf")),
            "C": float(hp.get("C", 10.0)),
            "gamma": hp.get("gamma", "scale"),
            "probability": True,
            "class_weight": hp.get("class_weight", "balanced" if w_tr is None else None),
            "random_state": int(seed),
        }
        clf = SVC(**eff_hp)

    clf.fit(X_tr_scaled, y_tr_local, sample_weight=w_tr)
    preds_local = np.asarray(clf.predict(X_te_scaled), dtype=np.int64)
    probs_head = np.asarray(clf.predict_proba(X_te_scaled), dtype=np.float64)

    pred_names = tuple(head_class_names[int(idx)] for idx in preds_local)
    pred_canonical_ids = np.asarray(
        [HARMONIZED_CLASS_TO_ID[name] for name in pred_names], dtype=np.int64
    )

    metrics = evaluate_variable_space_predictions(
        y_true=y_test,
        y_pred=pred_canonical_ids,
        head_class_names=head_class_names,
        y_proba=probs_head,
    )

    return ClassicalBaselineRunResult(
        protocol=proto_norm,
        variant=variant,
        model_name=model_norm,
        head_class_names=head_class_names,
        y_pred_canonical_ids=pred_canonical_ids,
        y_pred_class_names=pred_names,
        y_proba_head=probs_head,
        metrics=metrics,
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=scaler.scale_.copy(),
        hyperparameters={k: v for k, v in eff_hp.items() if k != "n_jobs"},
        joint_learning_note=note,
    )


# =============================================================================
# 9. ISOLATED MFPT DECIMATION (97,656 Hz -> 48,828 Hz) & NORMALIZATION (SECTION 7)
# =============================================================================


MFPT_97K_RECORDINGS: Tuple[Tuple[str, str, str], ...] = (
    ("Load_270", "Class_23", "s00000"),
    ("Load_270", "Class_23", "s00001"),
    ("Load_270", "Class_23", "s00002"),
    ("Load_270", "Class_25", "s00003"),
    ("Load_270", "Class_25", "s00004"),
    ("Load_270", "Class_25", "s00005"),
)


def prepare_isolated_mfpt_detrended_rms(
    processed_dir: str | Path,
    processed_tl_dir: str | Path,
    backup_root: Optional[str | Path] = None,
    target_fs_hz: int = 48828,
) -> Dict[str, Any]:
    """
    Execute the isolated remediation and preparation of `MFPT` (Section 7):
      1. Identify the 6 recordings in `data/processed/MFPT` acquired at `97,656 Hz`
         (`Load_270/Class_23/s00000..s00002` and `Load_270/Class_25/s00003..s00005`,
         each currently split into 12 windows of 48,828 points = 0.5 s).
      2. Preserve 100% of the original files in `data/processed/MFPT` (never overwriting
         `data/processed/MFPT`) and create a timestamped backup copy + SHA-256 manifest
         under `backup_root`.
      3. Reconstruct each of the 6 recordings (`12 * 48,828 = 585,936` samples at `97,656 Hz`),
         decimate from `97,656 Hz` to `48,828 Hz` using polyphase FIR anti-aliasing filtering
         (`scipy.signal.resample_poly(sig, up=1, down=2)` -> `292,968` samples), apply linear
         `detrend`, and slice into `6` non-overlapping 1-second windows of `48,828` samples
         (`w00..w05`).
      4. Copy the remaining 14 recordings (`42` windows in `Load_0..300`, already at `48,828 Hz`
         = 1.0 s per window), producing `18 + 18 + 42 = 78` harmonized 1-second windows in
         `data/processed_tl/detrended/MFPT`.
      5. Compute `RMS_ref` over the `18` harmonized `Class_23` (`Normal`) windows and generate
         `data/processed_tl/detrended_rms/MFPT` (`78` windows) via the transactional pipeline,
         updating `data/processed_tl/detrended_rms/normalization_manifest.csv` (merging with
         existing datasets without touching any other dataset).
      6. Validate frequency, dimensions (`(48828,)`), window counts (`78` total: `18` `Class_23`,
         `21` `Class_24`, `39` `Class_25`), labels, recordings (`20` unique recordings), and
         mathematical RMS parity.
    """
    from src.preprocessing import normalize_dataset_tl as norm_tl
    from src.vibnet.data.transactional_io import (
        commit_staged_dataset_dir,
        create_staging_dataset_dir,
        validate_staged_dataset_dir,
    )

    proc_root = Path(processed_dir).expanduser().resolve()
    tl_root = Path(processed_tl_dir).expanduser().resolve()
    mfpt_src_dir = proc_root / "MFPT"
    if not mfpt_src_dir.is_dir():
        raise FileNotFoundError(f"MFPT source directory not found: {mfpt_src_dir}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    resolved_bkp_root = (
        Path(backup_root).expanduser().resolve()
        if backup_root is not None
        else (proc_root.parent / "backups" / f"remediation_mfpt_{ts}")
    )
    resolved_bkp_root.mkdir(parents=True, exist_ok=True)

    # 1. Backup original data/processed/MFPT and existing normalization_manifest.csv
    bkp_mfpt_dir = resolved_bkp_root / "processed_MFPT_original_114_backup"
    if not bkp_mfpt_dir.exists():
        shutil.copytree(mfpt_src_dir, bkp_mfpt_dir)

    norm_manifest_path = tl_root / "detrended_rms" / "normalization_manifest.csv"
    if norm_manifest_path.is_file():
        bkp_man_dir = resolved_bkp_root / "manifests_before"
        bkp_man_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(norm_manifest_path, bkp_man_dir / "normalization_manifest.csv")

    # 2. Group all existing files in data/processed/MFPT by (condition, class, recording_prefix)
    src_files = sorted(mfpt_src_dir.rglob("*.npy"))
    grouped_files: Dict[Tuple[str, str, str], List[Path]] = defaultdict(list)
    for fp in src_files:
        rel = fp.relative_to(mfpt_src_dir).parts
        if len(rel) != 3:
            raise ValueError(f"Unexpected nesting in {fp}")
        cond_name, cls_name, fname = rel
        rec_prefix, _ = parse_window_filename(fname)
        grouped_files[(cond_name, cls_name, rec_prefix)].append(fp)

    # 3. Build harmonized data/processed_tl/detrended/MFPT transactionally in staging
    detrended_variant_root = tl_root / "detrended"
    detrended_variant_root.mkdir(parents=True, exist_ok=True)
    tx_bkp_root = tl_root / ".backups"

    staging_det_dir = create_staging_dataset_dir(
        allowed_variant_root=detrended_variant_root,
        dataset_name="MFPT",
        backup_root=tx_bkp_root,
        allowed_datasets=norm_tl.ALL_SUPPORTED_DATASETS,
    )

    decimated_recordings_audit: List[Dict[str, Any]] = []
    copied_recordings_audit: List[Dict[str, Any]] = []
    expected_rel_paths: List[str] = []
    window_counts_by_class: Dict[str, int] = defaultdict(int)
    recording_counts_by_class: Dict[str, int] = defaultdict(int)

    for (cond_name, cls_name, rec_prefix), f_list in sorted(grouped_files.items()):
        f_list_sorted = sorted(
            f_list, key=lambda p: parse_window_filename(p.name)[1]
        )
        out_cls_dir = staging_det_dir / cond_name / cls_name
        out_cls_dir.mkdir(parents=True, exist_ok=True)
        recording_counts_by_class[cls_name] += 1

        if (cond_name, cls_name, rec_prefix) in MFPT_97K_RECORDINGS or len(f_list_sorted) == 12:
            # 97,656 Hz recording (12 half-second windows of 48,828 points = 585,936 points)
            raw_parts = [
                np.asarray(np.load(p, allow_pickle=False), dtype=np.float64).reshape(-1)
                for p in f_list_sorted
            ]
            for arr in raw_parts:
                if arr.shape != (target_fs_hz,):
                    raise ValueError(
                        f"Unexpected window shape {arr.shape} in {rec_prefix} (expected {(target_fs_hz,)})"
                    )
            concat_97k = np.concatenate(raw_parts, axis=0)
            decimated_48k = resample_poly(concat_97k, up=1, down=2)
            detrended_48k = np.asarray(
                scipy_detrend(decimated_48k, type="linear"), dtype=np.float64
            )
            n_full_windows = len(detrended_48k) // int(target_fs_hz)
            for w_idx in range(n_full_windows):
                win = detrended_48k[
                    w_idx * int(target_fs_hz) : (w_idx + 1) * int(target_fs_hz)
                ]
                fname = f"{rec_prefix}_w{w_idx:02d}.npy"
                np.save(out_cls_dir / fname, win)
                expected_rel_paths.append(f"{cond_name}/{cls_name}/{fname}")
                window_counts_by_class[cls_name] += 1

            decimated_recordings_audit.append(
                {
                    "condition": cond_name,
                    "raw_class": cls_name,
                    "recording_prefix": rec_prefix,
                    "original_fs_hz": 97656,
                    "target_fs_hz": int(target_fs_hz),
                    "windows_before": len(f_list_sorted),
                    "samples_before": int(len(concat_97k)),
                    "samples_after_decimation": int(len(detrended_48k)),
                    "windows_after": int(n_full_windows),
                }
            )
        else:
            # Already at 48,828 Hz (1.0 s windows of 48,828 points, already detrended)
            for w_idx, p in enumerate(f_list_sorted):
                arr = np.asarray(
                    np.load(p, allow_pickle=False), dtype=np.float64
                ).reshape(-1)
                if arr.shape != (target_fs_hz,):
                    raise ValueError(
                        f"Unexpected window shape {arr.shape} in {p} (expected {(target_fs_hz,)})"
                    )
                fname = f"{rec_prefix}_w{w_idx:02d}.npy"
                np.save(out_cls_dir / fname, arr)
                expected_rel_paths.append(f"{cond_name}/{cls_name}/{fname}")
                window_counts_by_class[cls_name] += 1

            copied_recordings_audit.append(
                {
                    "condition": cond_name,
                    "raw_class": cls_name,
                    "recording_prefix": rec_prefix,
                    "original_fs_hz": int(target_fs_hz),
                    "target_fs_hz": int(target_fs_hz),
                    "windows_before": len(f_list_sorted),
                    "windows_after": len(f_list_sorted),
                }
            )

    total_harmonized_windows = sum(window_counts_by_class.values())
    val_det = validate_staged_dataset_dir(
        staging_dir=staging_det_dir,
        expected_window_count=total_harmonized_windows,
        expected_window_len=int(target_fs_hz),
        expected_relative_paths=expected_rel_paths,
    )
    if not val_det["valid"]:
        raise RuntimeError(
            f"Validation of staged harmonized MFPT failed: {val_det['reason']}"
        )

    commit_staged_dataset_dir(
        staging_dir=staging_det_dir,
        target_dir=detrended_variant_root / "MFPT",
        allowed_variant_root=detrended_variant_root,
        backup_root=tx_bkp_root,
        dataset_name="MFPT",
        variant_name="detrended",
        allowed_datasets=norm_tl.ALL_SUPPORTED_DATASETS,
        keep_backup=True,
    )

    # 4. Run official RMS normalization for MFPT only (updates detrended_rms/MFPT and merges normalization_manifest.csv)
    old_input_root = norm_tl.INPUT_ROOT
    old_fallback_root = norm_tl.PROCESSED_FALLBACK_ROOT
    old_variants = copy.deepcopy(norm_tl.VARIANTS)
    try:
        norm_tl.INPUT_ROOT = tl_root
        norm_tl.PROCESSED_FALLBACK_ROOT = proc_root
        norm_tl.VARIANTS = {
            "raw": {
                "input_dir": tl_root / "raw",
                "output_dir": tl_root / "raw_rms",
            },
            "detrended": {
                "input_dir": tl_root / "detrended",
                "output_dir": tl_root / "detrended_rms",
            },
        }
        (tl_root / "detrended_rms").mkdir(parents=True, exist_ok=True)
        refs = norm_tl.build_reference_map()
        manifest_rows: List[Dict[str, Any]] = []
        norm_tl.process_variant(
            variant_name="detrended",
            dataset_name="MFPT",
            references=refs,
            manifest_rows=manifest_rows,
            fallback_processed_root=proc_root,
            backup_root=tx_bkp_root,
            keep_backup=True,
        )
        norm_tl.write_manifest(
            manifest_rows,
            tl_root / "detrended_rms",
            merge_existing=True,
            backup_root=tx_bkp_root,
            keep_backup=True,
        )
    finally:
        norm_tl.INPUT_ROOT = old_input_root
        norm_tl.PROCESSED_FALLBACK_ROOT = old_fallback_root
        norm_tl.VARIANTS = old_variants

    # 5. Verify mathematical and structural integrity of detrended_rms/MFPT
    from scripts.validate_dataset_tl import compare_pair, validate_reference_class

    mfpt_rms_row = manifest_rows[0]
    rms_ref_harmonized = float(mfpt_rms_row["reference_rms"])
    rms_ref_ms2 = convert_rms_to_ms2(rms_ref_harmonized, "g")

    det_rms_mfpt_dir = tl_root / "detrended_rms" / "MFPT"
    det_rms_files = sorted(det_rms_mfpt_dir.rglob("*.npy"))
    normal_rms_after = [
        float(np.sqrt(np.mean(np.square(np.load(f, allow_pickle=False)))))
        for f in det_rms_files
        if f.parent.name == "Class_23"
    ]
    mean_normal_rms_after = float(np.mean(normal_rms_after))

    return {
        "dataset": "MFPT",
        "original_processed_dir_preserved": str(mfpt_src_dir),
        "original_processed_window_count": len(src_files),
        "backup_dir": str(bkp_mfpt_dir),
        "harmonized_detrended_dir": str(detrended_variant_root / "MFPT"),
        "harmonized_detrended_rms_dir": str(det_rms_mfpt_dir),
        "target_fs_hz": int(target_fs_hz),
        "window_shape": (int(target_fs_hz),),
        "n_unique_recordings": len(grouped_files),
        "n_decimated_97k_recordings": len(decimated_recordings_audit),
        "n_native_48k_recordings": len(copied_recordings_audit),
        "total_harmonized_windows": int(total_harmonized_windows),
        "window_counts_by_raw_class": dict(window_counts_by_class),
        "recording_counts_by_raw_class": dict(recording_counts_by_class),
        "reference_class": str(mfpt_rms_row["reference_classes"]),
        "reference_windows": int(mfpt_rms_row["reference_windows"]),
        "reference_rms_g": rms_ref_harmonized,
        "reference_rms_ms2": rms_ref_ms2,
        "mean_normal_rms_after_normalization": mean_normal_rms_after,
        "decimated_recordings": decimated_recordings_audit,
        "native_48k_recordings": copied_recordings_audit,
    }


# =============================================================================
# 10. SAFE NON-OVERWRITING OUTPUT DIRECTORY ALLOCATOR
# =============================================================================


def allocate_non_overwriting_run_dir(
    output_root: str | Path,
    experiment_prefix: str = "run",
) -> Path:
    """
    Create a new timestamped directory under `output_root` (`<prefix>_YYYYMMDD_HHMMSS`)
    that is guaranteed never to overwrite any existing directory or file.
    """
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{experiment_prefix}_{ts}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{experiment_prefix}_{ts}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate
