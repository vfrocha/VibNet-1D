#!/usr/bin/env python3
"""
Experimento de Transfer Learning Cross-Dataset: MLP vs. TabNet (detrended_rms + SMOTE).

Avalia a transferência de aprendizado entre múltiplos datasets de vibração utilizando
exclusivamente sinais processados na variante `detrended_rms`, balanceamento SMOTE
auditado no split de treino de cada fold LOCO, e comparação controlada entre:
  - MLP (HybridDLModel: MLPEncoder compartilhado + cabeças lineares específicas por domínio)
  - TabNet (TabNetPretrainer não supervisionado nas fontes + TabNetClassifier no target)

Datasets de destino (targets) padrão:
  - CWRU_12k, CWRU_48k, PU, HUST_Gearbox, UOEMD

Datasets de origem (sources) candidatos padrão:
  - UORED, HUST, Mechanical_Gear, Electric_Motor

Modos de combinação de fontes (--source-modes):
  - all          : todas as fontes elegíveis em conjunto
  - single       : cada fonte elegível individualmente
  - combinations : pares e trincas de fontes elegíveis (análise complementar)
  - power_set    : todos os 2^K - 1 subconjuntos não vazios

Saídas geradas em:
  results/tl_cross_dataset_mlp_tabnet/run_<YYYYMMDD_HHMMSS>/
"""

import argparse
import csv
import hashlib
import itertools
import json
import logging
import os
import random
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 1. CONFIGURAÇÃO DOS DATASETS E SEMÂNTICA EXPLÍCITA DE CLASSES
# =============================================================================

DEFAULT_TARGETS: List[str] = [
    "CWRU_12k",
    "CWRU_48k",
    "PU",
    "HUST_Gearbox",
    "UOEMD",
]

DEFAULT_SOURCES: List[str] = [
    "UORED",
    "HUST",
    "Mechanical_Gear",
    "Electric_Motor",
]

DATASETS_CONFIG: Dict[str, int] = {
    "CWRU_12k": 12000,
    "CWRU_48k": 48000,
    "PU": 64000,
    "HUST_Gearbox": 25600,
    "UOEMD": 42000,
    "UORED": 42000,
    "HUST": 51200,
    "Mechanical_Gear": 10000,
    "Electric_Motor": 42000,
    "MFPT": 48828,
}

# Vocabulário explícito e semântica física das classes por dataset
# Fonte de verdade: docs/distribuicao_classes_datasets.csv e docs/dataset_mapping.md
DATASET_CLASS_SEMANTICS: Dict[str, Dict[str, Any]] = {
    "CWRU_12k": {
        "role_default": "target",
        "component": "bearing",
        "fs_hz": 12000,
        "normal_class": "Class_0",
        "expected_classes": ["Class_0", "Class_1", "Class_2", "Class_3"],
        "class_descriptions": {
            "Class_0": "Normal (Healthy Bearing)",
            "Class_1": "Inner Race Fault (IR)",
            "Class_2": "Ball / Rolling Element Fault (B)",
            "Class_3": "Outer Race Fault (OR)",
        },
        "loco_axis": "Load_{0,1,2,3}HP x Sev_{0.007,0.014,0.021,0.028} (Normal replicated across conditions)",
    },
    "CWRU_48k": {
        "role_default": "target",
        "component": "bearing",
        "fs_hz": 48000,
        "normal_class": "Class_0",
        "expected_classes": ["Class_0", "Class_1", "Class_2", "Class_3"],
        "class_descriptions": {
            "Class_0": "Normal (Healthy Bearing - shared reference from CWRU_12k)",
            "Class_1": "Inner Race Fault (IR)",
            "Class_2": "Ball / Rolling Element Fault (B)",
            "Class_3": "Outer Race Fault (OR)",
        },
        "loco_axis": "Load_{0,1,2,3}HP x Sev_{0.007,0.014,0.021}",
    },
    "PU": {
        "role_default": "target",
        "component": "bearing",
        "fs_hz": 64000,
        "normal_class": "Normal",
        "expected_classes": [
            "Normal",
            "IR_EDM", "IR_Fatigue", "IR_Indent",
            "OR_EDM", "OR_Electric", "OR_Fatigue",
            "Combo_Fatigue", "Combo_Plastic",
        ],
        "class_descriptions": {
            "Normal": "Healthy Bearing",
            "IR_EDM": "Inner Race Fault (EDM artificial)",
            "IR_Fatigue": "Inner Race Fault (Fatigue pitting)",
            "IR_Indent": "Inner Race Fault (Plastic indentation)",
            "OR_EDM": "Outer Race Fault (EDM artificial)",
            "OR_Electric": "Outer Race Fault (Electric engraver)",
            "OR_Fatigue": "Outer Race Fault (Fatigue pitting)",
            "Combo_Fatigue": "Compound Fault IR+OR (Fatigue)",
            "Combo_Plastic": "Compound Fault IR+OR (Plastic deformation)",
        },
        "loco_axis": "N09_M07_F10, N15_M01_F10, N15_M07_F04, N15_M07_F10",
    },
    "HUST_Gearbox": {
        "role_default": "target",
        "component": "gearbox",
        "fs_hz": 25600,
        "normal_class": "Normal",
        "expected_classes": ["Normal", "Broken", "Missing"],
        "class_descriptions": {
            "Normal": "Healthy Gear",
            "Broken": "Broken Tooth Gear Fault",
            "Missing": "Missing Tooth Gear Fault",
        },
        "loco_axis": "Cond_{20,25,30,35,40}Hz x {1..6}",
    },
    "UOEMD": {
        "role_default": "target",
        "component": "motor_bearing",
        "fs_hz": 42000,
        "normal_class": "H_1",
        "expected_classes": [
            "H_1", "B_1", "C_1", "I_1", "O_1",
            "R_1", "S_1", "U_1", "V_1", "W_1",
        ],
        "class_descriptions": {
            "H_1": "Healthy Motor/Bearing",
            "B_1": "Bowed Rotor / Ball Fault",
            "C_1": "Broken Rotor Bars / Cage Fault",
            "I_1": "Inner Race Bearing Fault",
            "O_1": "Outer Race Bearing Fault",
            "R_1": "Rotor Unbalance",
            "S_1": "Stator Winding Fault",
            "U_1": "Voltage Unbalance",
            "V_1": "Misalignment",
            "W_1": "Fault Condition W_1",
        },
        "loco_axis": "Load_{0,1,2,3}",
    },
    "UORED": {
        "role_default": "source",
        "component": "bearing",
        "fs_hz": 42000,
        "normal_class": "H",
        "expected_classes": ["H", "B", "C", "I", "O"],
        "class_descriptions": {
            "H": "Healthy Bearing",
            "B": "Ball Fault",
            "C": "Cage Fault",
            "I": "Inner Race Fault",
            "O": "Outer Race Fault",
        },
        "loco_axis": "Cond_1_1 .. Cond_5_4",
    },
    "HUST": {
        "role_default": "source",
        "component": "bearing",
        "fs_hz": 51200,
        "normal_class": "N",
        "expected_classes": ["N", "B", "I", "O", "IB", "IO", "OB"],
        "class_descriptions": {
            "N": "Normal Bearing",
            "B": "Ball Fault",
            "I": "Inner Race Fault",
            "O": "Outer Race Fault",
            "IB": "Compound Inner + Ball Fault",
            "IO": "Compound Inner + Outer Fault",
            "OB": "Compound Outer + Ball Fault",
        },
        "loco_axis": "Load_{0,200,400}W",
    },
    "Mechanical_Gear": {
        "role_default": "source",
        "component": "gearbox",
        "fs_hz": 10000,
        "normal_class": "Normal",
        "expected_classes": ["Normal", "Chip5a", "Miss", "Root", "Surface"],
        "class_descriptions": {
            "Normal": "Healthy Gear",
            "Chip5a": "Chipped Tooth Gear Fault",
            "Miss": "Missing Tooth Gear Fault",
            "Root": "Gear Root Crack Fault",
            "Surface": "Gear Surface Wear Fault",
        },
        "loco_axis": "Speed_{10,15,20,25}_Load_{0,1,2}",
    },
    "Electric_Motor": {
        "role_default": "source",
        "component": "motor_bearing",
        "fs_hz": 42000,
        "normal_class": "Normal",
        "expected_classes": ["Normal", "Inner", "Outer"],
        "class_descriptions": {
            "Normal": "Healthy Motor Bearing",
            "Inner": "Inner Race Fault",
            "Outer": "Outer Race Fault",
        },
        "loco_axis": "Load_{1,2,3}",
    },
    "MFPT": {
        "role_default": "source",
        "component": "bearing",
        "fs_hz": 48828,
        "normal_class": "Class_23",
        "expected_classes": [
            f"Class_{i}" for i in range(24) if i != 9
        ],
        "class_descriptions": {
            "Class_23": "Normal Baseline Condition",
        },
        "loco_axis": "Load_*lbs",
    },
}


# =============================================================================
# 2. UTILITÁRIOS DE REPRODUTIBILIDADE, HASH DE CÓDIGO E COMBINATÓRIA DE FONTES
# =============================================================================

def set_global_seed(seed: int) -> None:
    """Fixa as seeds de Python, NumPy e PyTorch para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_code_identification() -> Dict[str, Any]:
    """
    Retorna a identificação do código utilizado:
      - hash git (se disponível)
      - hashes SHA-256 dos scripts e módulos principais do experimento
    """
    info: Dict[str, Any] = {
        "git_commit": None,
        "git_dirty": None,
        "file_sha256": {},
    }
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if res.returncode == 0:
            info["git_commit"] = res.stdout.strip()
        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if status_res.returncode == 0:
            info["git_dirty"] = bool(status_res.stdout.strip())
    except Exception:
        pass

    tracked_files = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "models" / "build_tabnet_resnet_balance.py",
        PROJECT_ROOT / "src" / "preprocessing" / "normalize_dataset_tl.py",
        PROJECT_ROOT / "src" / "data" / "validate_dataset_tl.py",
    ]
    for fp in tracked_files:
        if fp.exists():
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            rel = str(fp.relative_to(PROJECT_ROOT)).replace("\\", "/")
            info["file_sha256"][rel] = h
    return info


def build_source_combinations(
    eligible_sources: Sequence[str],
    source_modes: Sequence[str],
) -> List[Tuple[str, Tuple[str, ...]]]:
    """
    Constrói a lista ordenada de configurações de fontes de treinamento sem duplicatas.

    Retorna tuplas (source_mode_label, (source_1, source_2, ...)):
      - 'all'          -> ("all_sources", tuple(eligible_sources))
      - 'single'       -> ("single_source", (src,)) para cada fonte elegível
      - 'combinations' -> ("pair_sources", (s1, s2)) e ("triple_sources", (s1, s2, s3))
      - 'power_set'    -> todos os subconjuntos de tamanho 1..K
    """
    sources = list(eligible_sources)
    if not sources:
        return []

    seen: set = set()
    ordered: List[Tuple[str, Tuple[str, ...]]] = []

    def _add(mode_label: str, combo: Tuple[str, ...]) -> None:
        if not combo or combo in seen:
            return
        seen.add(combo)
        ordered.append((mode_label, combo))

    modes = list(source_modes)
    if "power_set" in modes:
        modes = ["all", "single", "combinations"]

    if "all" in modes:
        _add("all_sources", tuple(sources))

    if "single" in modes:
        for s in sources:
            _add("single_source", (s,))

    if "combinations" in modes and len(sources) >= 2:
        for r in range(2, len(sources)):
            label = "pair_sources" if r == 2 else (
                "triple_sources" if r == 3 else f"subset_{r}_sources"
            )
            for combo in itertools.combinations(sources, r):
                _add(label, tuple(combo))

    return ordered


def format_source_combo(combo: Sequence[str]) -> str:
    """Formata a tupla de fontes em string canônica separada por '|'."""
    return "|".join(combo)


# =============================================================================
# 3. AUDITORIA DE MANIFESTO, CARREGAMENTO E EXTRAÇÃO DE ATRIBUTOS
# =============================================================================

def load_detrended_rms_manifest(variant_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Lê `data/processed_tl/detrended_rms/normalization_manifest.csv` se existir.
    Retorna um dicionário indexado pelo nome do dataset.
    """
    manifest_path = variant_root / "normalization_manifest.csv"
    records: Dict[str, Dict[str, Any]] = {}
    if not manifest_path.exists():
        return records

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("output_variant") == "detrended_rms":
                ds = row["dataset"]
                records[ds] = {
                    "reference_dataset": row.get("reference_dataset", ""),
                    "reference_variant": row.get("reference_variant", ""),
                    "reference_classes": row.get("reference_classes", ""),
                    "reference_rms": float(row["reference_rms"]) if row.get("reference_rms") else np.nan,
                    "reference_windows": int(row["reference_windows"]) if row.get("reference_windows") else 0,
                    "total_windows": int(row["total_windows"]) if row.get("total_windows") else 0,
                }
    return records


def audit_and_extract_dataset(
    dataset_name: str,
    role: str,
    variant_root: Path,
    manifest_records: Dict[str, Dict[str, Any]],
    require_manifest: bool = True,
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]], Dict[str, Any]]:
    """
    Verifica a disponibilidade, integridade do manifesto `detrended_rms`,
    semântica de classes e extrai os 141 atributos sem aplicar novo detrend
    ou normalização RMS aos sinais já processados.

    Retorna:
      (data_tuple_or_None, audit_row_dict)
    onde `data_tuple` = (X_features, y_labels_str, conds_str).
    """
    from src.features.extractors_v2 import extract_advanced_features
    from src.features.signalai_wrapper import extract_fusion_features

    ds_dir = variant_root / dataset_name
    fs = DATASETS_CONFIG.get(dataset_name, 12000)
    semantics = DATASET_CLASS_SEMANTICS.get(dataset_name, {})
    in_manifest = dataset_name in manifest_records
    manifest_info = manifest_records.get(dataset_name, {})

    audit_row: Dict[str, Any] = {
        "dataset": dataset_name,
        "role": role,
        "component": semantics.get("component", "unknown"),
        "fs_hz": fs,
        "normal_class_expected": semantics.get("normal_class", ""),
        "variant_dir": str(ds_dir),
        "dir_exists": ds_dir.exists(),
        "in_normalization_manifest": in_manifest,
        "manifest_reference_dataset": manifest_info.get("reference_dataset", ""),
        "manifest_reference_class": manifest_info.get("reference_classes", ""),
        "manifest_reference_rms": manifest_info.get("reference_rms", np.nan),
        "manifest_total_windows": manifest_info.get("total_windows", 0),
        "n_windows_loaded": 0,
        "n_conditions": 0,
        "n_classes": 0,
        "classes_found": "[]",
        "class_distribution": "{}",
        "eligible": False,
        "status_reason": "",
    }

    if not ds_dir.exists():
        audit_row["status_reason"] = (
            f"Diretório ausente em detrended_rms: {ds_dir}. "
            f"Execute normalize_dataset_tl.py --datasets {dataset_name} --variants detrended --use-processed-fallback."
        )
        return None, audit_row

    if require_manifest and not in_manifest:
        audit_row["status_reason"] = (
            f"Dataset '{dataset_name}' não consta em {variant_root / 'normalization_manifest.csv'}. "
            f"Execute normalize_dataset_tl.py e validate_dataset_tl.py antes do experimento."
        )
        return None, audit_row

    X_windows: List[np.ndarray] = []
    y_list: List[str] = []
    cond_list: List[str] = []

    cond_dirs = sorted([d for d in ds_dir.iterdir() if d.is_dir()])
    for cond_dir in cond_dirs:
        cond_name = cond_dir.name
        class_dirs = sorted([d for d in cond_dir.iterdir() if d.is_dir()])
        for class_dir in class_dirs:
            label = class_dir.name
            npy_files = sorted(class_dir.glob("*.npy"))
            for f in npy_files:
                sig = np.load(f, allow_pickle=False)
                sig_1d = np.asarray(sig, dtype=np.float64).reshape(-1)
                X_windows.append(sig_1d)
                y_list.append(label)
                cond_list.append(cond_name)

    if not X_windows:
        audit_row["status_reason"] = f"Nenhum arquivo .npy encontrado em {ds_dir}."
        return None, audit_row

    # Sinais em detrended_rms já estão detrended e normalizados por RMS_ref.
    # Não aplicamos detrend nem normalização RMS novamente.
    X_array = np.asarray(X_windows, dtype=np.float64)
    X_fusion = extract_fusion_features(
        X_array,
        fs,
        extract_advanced_features,
    )
    X = np.nan_to_num(
        np.asarray(X_fusion, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if X.ndim == 1:
        X = X.reshape(len(X_windows), -1)
    y = np.array(y_list)
    conds = np.array(cond_list)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    unique_conds = np.unique(conds)
    dist_dict = {str(k): int(v) for k, v in zip(unique_classes, class_counts)}

    audit_row["n_windows_loaded"] = int(len(X))
    audit_row["n_conditions"] = int(len(unique_conds))
    audit_row["n_classes"] = int(len(unique_classes))
    audit_row["classes_found"] = json.dumps([str(c) for c in unique_classes], ensure_ascii=False)
    audit_row["class_distribution"] = json.dumps(dist_dict, ensure_ascii=False)

    if len(unique_classes) < 2:
        audit_row["status_reason"] = (
            f"Número insuficiente de classes ({len(unique_classes)} < 2) no dataset {dataset_name}."
        )
        return None, audit_row

    if role == "target" and len(unique_conds) < 2:
        audit_row["status_reason"] = (
            f"Target '{dataset_name}' possui apenas {len(unique_conds)} condição operacional; "
            f"avaliação LOCO exige pelo menos 2 condições."
        )
        return None, audit_row

    audit_row["eligible"] = True
    audit_row["status_reason"] = "OK"
    return (X, y, conds), audit_row


# =============================================================================
# 4. BALANCEAMENTO SMOTE COM AUDITORIA POR FOLD E SEM VAZAMENTO
# =============================================================================

def balance_target_with_audit(
    X_train_scaled: np.ndarray,
    y_train_encoded: np.ndarray,
    class_names: Sequence[str],
    strategy: str = "smote",
    seed: int = 42,
    insufficient_policy: str = "fail_fold",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Aplica SMOTE exclusivamente aos dados de treinamento escalados de um fold LOCO,
    registrando estatísticas completas antes e depois do balanceamento.

    Parâmetros:
      - X_train_scaled      : matriz de atributos de treino após StandardScaler.fit_transform
      - y_train_encoded     : rótulos inteiros (0..C-1) do split de treino
      - class_names         : nomes originais das classes do dataset (LabelEncoder.classes_)
      - strategy            : 'smote' ou 'none'
      - seed                : semente aleatória passada ao SMOTE
      - insufficient_policy :
          * 'fail_fold'     : interrompe o fold quando alguma classe minoritária tem < 2 amostras
          * 'partial_smote' : aplica SMOTE apenas às classes com >= 2 amostras e registra
                              as classes que não puderam ser balanceadas

    Retorna:
      (X_balanced_or_None, y_balanced_or_None, audit_info)
    """
    unique_cls, counts_before = np.unique(y_train_encoded, return_counts=True)
    counts_before_map: Dict[str, int] = {
        str(class_names[int(c)]): int(cnt)
        for c, cnt in zip(unique_cls, counts_before)
    }
    n_before = int(len(y_train_encoded))

    audit_info: Dict[str, Any] = {
        "balance_status": "OK",
        "balance_reason": "OK",
        "n_train_before": n_before,
        "n_train_after": n_before,
        "counts_before": counts_before_map,
        "counts_after": dict(counts_before_map),
        "smote_params": {
            "strategy": strategy,
            "random_state": int(seed),
            "k_neighbors": None,
            "target_count": int(np.max(counts_before)) if len(counts_before) > 0 else 0,
            "sampling_strategy": None,
            "insufficient_policy": insufficient_policy,
        },
        "unbalanced_classes": {},
    }

    if len(unique_cls) < 2:
        audit_info["balance_status"] = "SKIPPED_INSUFFICIENT_CLASSES"
        audit_info["balance_reason"] = (
            f"Treino possui apenas {len(unique_cls)} classe(s); mínimo exigido = 2."
        )
        return None, None, audit_info

    if strategy == "none":
        audit_info["smote_params"]["sampling_strategy"] = "none"
        return X_train_scaled, y_train_encoded, audit_info

    if strategy != "smote":
        raise ValueError(f"Estratégia de balanceamento não suportada: {strategy}")

    max_count = int(np.max(counts_before))
    # Identifica classes que precisam de oversampling (count < max_count)
    # e verifica se alguma classe presente possui < 2 amostras (mínimo matemático do SMOTE)
    insufficient_classes: Dict[str, Dict[str, Any]] = {}
    eligible_minority_classes: Dict[int, int] = {}

    for c_idx, cnt in zip(unique_cls, counts_before):
        c_int = int(c_idx)
        c_label = str(class_names[c_int])
        cnt_int = int(cnt)
        if cnt_int < max_count:
            if cnt_int < 2:
                decision = (
                    "abort_fold"
                    if insufficient_policy == "fail_fold"
                    else "keep_unbalanced_in_partial_smote"
                )
                insufficient_classes[c_label] = {
                    "class_index": c_int,
                    "count_before": cnt_int,
                    "min_required_by_smote": 2,
                    "reason": "Amostras insuficientes para interpolação KNN no SMOTE (count < 2)",
                    "decision": decision,
                }
            else:
                eligible_minority_classes[c_int] = max_count

    # Também registra classes ausentes no fold de treino (count == 0)
    present_set = {int(c) for c in unique_cls}
    for c_int, c_label in enumerate(class_names):
        if c_int not in present_set:
            insufficient_classes[str(c_label)] = {
                "class_index": c_int,
                "count_before": 0,
                "min_required_by_smote": 2,
                "reason": "Classe ausente no conjunto de treinamento deste fold LOCO (count == 0)",
                "decision": "absent_in_train_fold",
            }

    audit_info["unbalanced_classes"] = insufficient_classes

    has_under_two = any(
        info["count_before"] > 0 and info["count_before"] < 2
        for info in insufficient_classes.values()
    )

    if has_under_two and insufficient_policy == "fail_fold":
        audit_info["balance_status"] = "FAILED_SMOTE_MIN_SAMPLES"
        audit_info["balance_reason"] = (
            f"Classe(s) com menos de 2 amostras no treino: "
            f"{json.dumps(insufficient_classes, ensure_ascii=False)}"
        )
        return None, None, audit_info

    if not eligible_minority_classes:
        # Todas as classes presentes já possuem max_count (já balanceado)
        # ou nenhuma classe elegível pôde ser balanceada em partial_smote
        audit_info["smote_params"]["k_neighbors"] = 0
        audit_info["smote_params"]["sampling_strategy"] = "already_balanced_or_no_eligible_minority"
        if has_under_two:
            audit_info["balance_status"] = "PARTIAL_NO_ELIGIBLE_CLASSES"
            audit_info["balance_reason"] = (
                "Nenhuma classe minoritária possuía >= 2 amostras para aplicar SMOTE parcial."
            )
        return X_train_scaled, y_train_encoded, audit_info

    from imblearn.over_sampling import SMOTE

    min_eligible_count = min(
        int(cnt)
        for c_idx, cnt in zip(unique_cls, counts_before)
        if int(c_idx) in eligible_minority_classes
    )
    k_neighbors = min(5, min_eligible_count - 1)

    if has_under_two and insufficient_policy == "partial_smote":
        sampling_strategy_arg: Any = dict(eligible_minority_classes)
        sampling_strategy_desc: Any = {
            str(class_names[k]): int(v) for k, v in eligible_minority_classes.items()
        }
    else:
        sampling_strategy_arg = "auto"
        sampling_strategy_desc = "auto"

    audit_info["smote_params"]["k_neighbors"] = int(k_neighbors)
    audit_info["smote_params"]["sampling_strategy"] = sampling_strategy_desc

    smote = SMOTE(
        sampling_strategy=sampling_strategy_arg,
        random_state=seed,
        k_neighbors=k_neighbors,
    )
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train_encoded)

    u_after, c_after = np.unique(y_res, return_counts=True)
    counts_after_map: Dict[str, int] = {
        str(class_names[int(c)]): int(cnt)
        for c, cnt in zip(u_after, c_after)
    }
    audit_info["n_train_after"] = int(len(y_res))
    audit_info["counts_after"] = counts_after_map
    if has_under_two and insufficient_policy == "partial_smote":
        audit_info["balance_status"] = "OK_PARTIAL_SMOTE"
        audit_info["balance_reason"] = (
            "SMOTE aplicado às classes com >= 2 amostras; classes com < 2 amostras mantidas sem oversampling."
        )

    return X_res.astype(np.float32), y_res, audit_info


# =============================================================================
# 5. CÁLCULO DE MÉTRICAS POR FOLD (COM TRATAMENTO RIGOROSO DE ROC-AUC)
# =============================================================================

def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    Calcula as métricas exigidas pelo protocolo:
      - Macro-F1 (métrica primária, restrita às classes relevantes do fold)
      - Accuracy
      - Balanced Accuracy
      - ROC-AUC (somente quando houver >= 2 classes presentes em y_true e probabilidades válidas;
                 nunca substitui ROC-AUC inválido por 0.0)
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    acc = float(accuracy_score(y_true, y_pred))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    relevant_classes = np.union1d(np.unique(y_true), np.unique(y_pred))
    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=relevant_classes,
            average="macro",
            zero_division=0,
        )
    )

    present_true = np.unique(y_true)
    roc_auc = np.nan
    roc_auc_valid = False
    roc_auc_note = ""

    if probs is None:
        roc_auc_note = "Probabilidades indisponíveis"
    elif len(present_true) < 2:
        roc_auc_note = (
            f"Apenas {len(present_true)} classe presente em y_test ({present_true.tolist()}); "
            f"ROC-AUC requer >= 2 classes no fold de teste."
        )
    elif not np.all(np.isfinite(probs)):
        roc_auc_note = "Probabilidades contêm valores não finitos (NaN/Inf)"
    else:
        try:
            if len(present_true) == 2:
                pos_cls = int(present_true[1])
                y_bin = (y_true == pos_cls).astype(int)
                roc_auc = float(roc_auc_score(y_bin, probs[:, pos_cls]))
                roc_auc_valid = True
                roc_auc_note = (
                    "OK_binary_present_classes"
                    if probs.shape[1] > 2
                    else "OK_binary"
                )
            else:
                sub_probs = probs[:, present_true].astype(np.float64)
                row_sums = sub_probs.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
                sub_probs = sub_probs / row_sums
                roc_auc = float(
                    roc_auc_score(
                        y_true,
                        sub_probs,
                        labels=present_true,
                        multi_class="ovr",
                        average="macro",
                    )
                )
                roc_auc_valid = True
                roc_auc_note = (
                    "OK_multiclass_ovr"
                    if len(present_true) == probs.shape[1]
                    else f"OK_multiclass_ovr_subset_{len(present_true)}_of_{probs.shape[1]}"
                )
        except ValueError as exc:
            roc_auc = np.nan
            roc_auc_valid = False
            roc_auc_note = f"ValueError: {exc}"

    return {
        "Macro F1": macro_f1,
        "ACC": acc,
        "Bal Acc": bal_acc,
        "ROC-AUC": roc_auc,
        "roc_auc_valid": roc_auc_valid,
        "roc_auc_note": roc_auc_note,
        "n_test_samples": int(len(y_true)),
        "n_test_classes_present": int(len(present_true)),
        "test_classes_present": [int(c) for c in present_true],
    }


# =============================================================================
# 6. TREINAMENTO E AVALIAÇÃO DOS MODELOS (MLP E TABNET)
# =============================================================================

def train_eval_mlp_fold(
    source_tensors: Dict[str, Tuple[np.ndarray, np.ndarray]],
    X_target_train_bal: np.ndarray,
    y_target_train_bal: np.ndarray,
    X_target_test_scaled: np.ndarray,
    num_classes_dict: Dict[str, int],
    target_name: str,
    epochs: int = 30,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Treina e avalia o modelo MLP (`HybridDLModel` com `MLPEncoder` compartilhado
    e cabeças específicas por domínio), preservando exatamente a arquitetura e o
    procedimento de `src/models/build_tabnet_resnet_balance.py`:
      - Etapa 1: Treinamento conjunto nas fontes + target balanceado (`epochs` épocas)
      - Etapa 2: Fine-tuning focado no target balanceado (`max(1, epochs // 2)` épocas, lr=5e-4)

    Retorna:
      (y_pred, probs) no fold de teste do target.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.build_tabnet_resnet_balance import HybridDLModel

    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = int(X_target_train_bal.shape[1])

    model = HybridDLModel(
        num_features=input_dim,
        dataset_classes_dict=num_classes_dict,
        encoder_type="mlp",
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    gen = torch.Generator()
    gen.manual_seed(seed)

    loaders: Dict[str, DataLoader] = {}
    for s_name, (X_s, y_s) in source_tensors.items():
        ds = TensorDataset(
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(y_s, dtype=torch.long),
        )
        loaders[s_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=gen,
        )

    target_bs = min(128, batch_size)
    target_ds = TensorDataset(
        torch.tensor(X_target_train_bal, dtype=torch.float32),
        torch.tensor(y_target_train_bal, dtype=torch.long),
    )
    loaders[target_name] = DataLoader(
        target_ds,
        batch_size=target_bs,
        shuffle=True,
        drop_last=False,
        generator=gen,
    )

    # Etapa 1: Pré-treinamento multi-domínio (fontes + target treino balanceado)
    model.train()
    for _ in range(epochs):
        for d_name, loader in loaders.items():
            for bx, by in loader:
                if bx.size(0) < 2:
                    continue
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits, _ = model(bx, dataset_name=d_name)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

    # Etapa 2: Fine-tuning exclusivo no target
    ft_epochs = max(1, epochs)
    optimizer_ft = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    target_loader = loaders[target_name]

    for _ in range(ft_epochs):
        for bx, by in target_loader:
            if bx.size(0) < 2:
                continue
            optimizer_ft.zero_grad()
            bx, by = bx.to(device), by.to(device)
            logits, _ = model(bx, dataset_name=target_name)
            loss = criterion(logits, by)
            loss.backward()
            optimizer_ft.step()

    # Avaliação no fold de teste isolado
    model.eval()
    X_test_tensor = torch.tensor(X_target_test_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits, _ = model(X_test_tensor, dataset_name=target_name)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    return preds, probs


def build_or_get_tabnet_pretrainer(
    cache: Dict[Tuple[str, int, int, int], Any],
    source_combo_key: str,
    X_pretrain: np.ndarray,
    pretrain_epochs: int,
    batch_size: int,
    seed: int,
) -> Any:
    """
    Executa (ou recupera do cache) o pré-treinamento não supervisionado do TabNet
    (`TabNetPretrainer`) sobre as fontes escaladas para uma combinação de fontes e seed.
    Como `X_pretrain` depende exclusivamente das fontes escaladas e da seed,
    reutilizar o `TabNetPretrainer` entre os folds LOCO de diferentes targets
    preserva o isolamento total do fold de teste e evita retrabalho redundante.
    """
    import copy

    cache_key = (source_combo_key, int(pretrain_epochs), int(batch_size), int(seed))
    if cache_key in cache:
        return copy.deepcopy(cache[cache_key])

    import torch
    from pytorch_tabnet.pretraining import TabNetPretrainer

    set_global_seed(seed)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    unsupervised_model = TabNetPretrainer(
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.9),
        mask_type="entmax",
        device_name=device_name,
        verbose=0,
        seed=seed,
    )

    unsupervised_model.fit(
        X_train=X_pretrain,
        eval_set=[X_pretrain[: min(2048, len(X_pretrain))]],
        max_epochs=pretrain_epochs,
        patience=10,
        batch_size=batch_size,
        virtual_batch_size=min(128, batch_size),
        num_workers=0,
        drop_last=False,
        pretraining_ratio=0.8,
    )

    cache[cache_key] = copy.deepcopy(unsupervised_model)
    return copy.deepcopy(unsupervised_model)


def train_eval_tabnet_fold(
    unsupervised_model: Any,
    X_target_train_bal: np.ndarray,
    y_target_train_bal: np.ndarray,
    X_target_test_scaled: np.ndarray,
    n_target_classes: int,
    finetune_epochs: int = 30,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Realiza o fine-tuning supervisionado do `TabNetClassifier` no conjunto de treino
    balanceado do target a partir dos pesos pré-treinados nas fontes (`from_unsupervised`),
    preservando os hiperparâmetros de `src/models/build_tabnet_resnet_balance.py`.

    Alinha o vetor de probabilidades para `n_target_classes` colunas caso alguma classe
    do target esteja ausente do fold de treino.
    """
    import copy
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier

    set_global_seed(seed)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    clf = TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-2, weight_decay=1e-5),
        scheduler_params={"step_size": 20, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",
        device_name=device_name,
        verbose=0,
        seed=seed,
    )

    clf.fit(
        X_train=X_target_train_bal,
        y_train=y_target_train_bal,
        eval_set=[(X_target_train_bal, y_target_train_bal)],
        eval_name=["train"],
        eval_metric=["balanced_accuracy"],
        max_epochs=finetune_epochs,
        patience=20,
        batch_size=min(256, len(X_target_train_bal), batch_size),
        virtual_batch_size=min(128, len(X_target_train_bal), batch_size),
        num_workers=0,
        drop_last=False,
        from_unsupervised=copy.deepcopy(unsupervised_model),
    )

    raw_probs = clf.predict_proba(X_target_test_scaled)
    preds = np.asarray(clf.predict(X_target_test_scaled), dtype=int)

    # Garante matriz de probabilidades com shape (N_test, n_target_classes)
    if raw_probs.shape[1] == n_target_classes:
        probs = raw_probs
    else:
        probs = np.zeros((len(X_target_test_scaled), n_target_classes), dtype=np.float64)
        for col_idx, cls_id in enumerate(clf.classes_):
            probs[:, int(cls_id)] = raw_probs[:, col_idx]

    return preds, probs


# =============================================================================
# 7. AGREGAÇÃO E TABELAS DE RESUMO CIENTÍFICO
# =============================================================================

def generate_summaries(df_results: pd.DataFrame, out_dir: Path) -> Dict[str, Path]:
    """
    Gera os arquivos CSV de resumo exigidos pelo protocolo:
      1. summary_by_target_source_model_seed.csv
      2. summary_by_target_source_model.csv
      3. summary_paired_mlp_vs_tabnet.csv (restrito estritamente aos mesmos folds válidos)
      4. summary_source_ranking_by_target.csv
    """
    paths: Dict[str, Path] = {}
    if df_results.empty:
        return paths

    metrics = ["Macro F1", "ACC", "Bal Acc", "ROC-AUC"]

    # 1. Resumo por (Target, Source_Mode, Source_Combo, Model, Seed)
    group_cols_seed = [
        "Target",
        "Source_Mode",
        "Source_Combo",
        "N_Sources",
        "Model",
        "Balance",
        "Seed",
    ]
    rows_seed: List[Dict[str, Any]] = []
    for keys, grp in df_results.groupby(group_cols_seed, dropna=False, sort=True):
        valid_grp = grp[grp["Fold_Status"] == "OK"]
        failed_grp = grp[grp["Fold_Status"] != "OK"]
        row: Dict[str, Any] = dict(zip(group_cols_seed, keys))
        row["n_total_folds"] = int(len(grp))
        row["n_valid_folds"] = int(len(valid_grp))
        row["n_failed_folds"] = int(len(failed_grp))
        row["n_roc_auc_valid_folds"] = int(
            valid_grp["ROC_AUC_Valid"].sum() if "ROC_AUC_Valid" in valid_grp else 0
        )
        for m in metrics:
            vals = pd.to_numeric(valid_grp[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows_seed.append(row)

    df_summary_seed = pd.DataFrame(rows_seed)
    p_seed = out_dir / "summary_by_target_source_model_seed.csv"
    df_summary_seed.to_csv(p_seed, index=False)
    paths["summary_by_target_source_model_seed"] = p_seed

    # 2. Resumo agregado sobre seeds e folds por (Target, Source_Mode, Source_Combo, Model)
    group_cols_model = [
        "Target",
        "Source_Mode",
        "Source_Combo",
        "N_Sources",
        "Model",
        "Balance",
    ]
    rows_model: List[Dict[str, Any]] = []
    for keys, grp in df_results.groupby(group_cols_model, dropna=False, sort=True):
        valid_grp = grp[grp["Fold_Status"] == "OK"]
        failed_grp = grp[grp["Fold_Status"] != "OK"]
        row = dict(zip(group_cols_model, keys))
        row["n_seeds"] = int(grp["Seed"].nunique())
        row["n_total_fold_evals"] = int(len(grp))
        row["n_valid_fold_evals"] = int(len(valid_grp))
        row["n_failed_fold_evals"] = int(len(failed_grp))
        row["n_roc_auc_valid_evals"] = int(
            valid_grp["ROC_AUC_Valid"].sum() if "ROC_AUC_Valid" in valid_grp else 0
        )
        for m in metrics:
            vals = pd.to_numeric(valid_grp[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows_model.append(row)

    df_summary_model = pd.DataFrame(rows_model)
    p_model = out_dir / "summary_by_target_source_model.csv"
    df_summary_model.to_csv(p_model, index=False)
    paths["summary_by_target_source_model"] = p_model

    # 3. Comparação pareada MLP vs TabNet restrita aos MESMOS folds válidos
    valid_only = df_results[df_results["Fold_Status"] == "OK"].copy()
    models_present = set(valid_only["Model"].unique())
    if {"MLP", "TabNet"}.issubset(models_present):
        key_cols = ["Target", "Source_Mode", "Source_Combo", "Balance", "Seed", "Fold"]
        mlp_df = valid_only[valid_only["Model"] == "MLP"].set_index(key_cols)
        tab_df = valid_only[valid_only["Model"] == "TabNet"].set_index(key_cols)
        common_idx = mlp_df.index.intersection(tab_df.index)

        paired_rows: List[Dict[str, Any]] = []
        if len(common_idx) > 0:
            mlp_common = mlp_df.loc[common_idx].reset_index()
            tab_common = tab_df.loc[common_idx].reset_index()
            merged = pd.merge(
                mlp_common,
                tab_common,
                on=key_cols,
                suffixes=("_MLP", "_TabNet"),
            )

            for (tgt, s_mode, s_combo, bal), grp in merged.groupby(
                ["Target", "Source_Mode", "Source_Combo", "Balance"], sort=True
            ):
                p_row: Dict[str, Any] = {
                    "Target": tgt,
                    "Source_Mode": s_mode,
                    "Source_Combo": s_combo,
                    "Balance": bal,
                    "n_common_valid_fold_evals": int(len(grp)),
                }
                for m in metrics:
                    m_mlp = pd.to_numeric(grp[f"{m}_MLP"], errors="coerce")
                    m_tab = pd.to_numeric(grp[f"{m}_TabNet"], errors="coerce")
                    both_valid = m_mlp.notna() & m_tab.notna()
                    m_mlp_v = m_mlp[both_valid]
                    m_tab_v = m_tab[both_valid]
                    p_row[f"{m}_MLP_mean"] = float(m_mlp_v.mean()) if len(m_mlp_v) > 0 else np.nan
                    p_row[f"{m}_MLP_std"] = float(m_mlp_v.std(ddof=1)) if len(m_mlp_v) > 1 else 0.0
                    p_row[f"{m}_TabNet_mean"] = float(m_tab_v.mean()) if len(m_tab_v) > 0 else np.nan
                    p_row[f"{m}_TabNet_std"] = float(m_tab_v.std(ddof=1)) if len(m_tab_v) > 1 else 0.0
                    diff = m_mlp_v - m_tab_v
                    p_row[f"{m}_Diff_MLP_minus_TabNet"] = (
                        float(diff.mean()) if len(diff) > 0 else np.nan
                    )
                p_row["Winner_Macro_F1"] = (
                    "MLP"
                    if p_row["Macro F1_MLP_mean"] > p_row["Macro F1_TabNet_mean"]
                    else ("TabNet" if p_row["Macro F1_TabNet_mean"] > p_row["Macro F1_MLP_mean"] else "Tie")
                )
                paired_rows.append(p_row)

        df_paired = pd.DataFrame(paired_rows)
        p_paired = out_dir / "summary_paired_mlp_vs_tabnet.csv"
        df_paired.to_csv(p_paired, index=False)
        paths["summary_paired_mlp_vs_tabnet"] = p_paired

    # 4. Ranking de fontes por Target e Modelo (ordenado por Macro F1_mean decrescente)
    if not df_summary_model.empty:
        df_ranking = df_summary_model.sort_values(
            by=["Target", "Model", "Macro F1_mean"],
            ascending=[True, True, False],
        ).copy()
        df_ranking["Rank_within_Target_Model"] = (
            df_ranking.groupby(["Target", "Model"]).cumcount() + 1
        )
        p_rank = out_dir / "summary_source_ranking_by_target.csv"
        df_ranking.to_csv(p_rank, index=False)
        paths["summary_source_ranking_by_target"] = p_rank

    return paths


# =============================================================================
# 8. PONTO DE ENTRADA PRINCIPAL
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experimento Cross-Dataset Transfer Learning com detrended_rms, SMOTE, MLP e TabNet."
        )
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        choices=list(DATASETS_CONFIG.keys()),
        help="Datasets de destino (targets) para avaliação LOCO.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        choices=list(DATASETS_CONFIG.keys()),
        help="Datasets de origem (sources) candidatos para treinamento.",
    )
    parser.add_argument(
        "--source-modes",
        nargs="+",
        default=["all", "single", "combinations"],
        choices=["all", "single", "combinations", "power_set"],
        help=(
            "Estratégias de combinação de fontes: "
            "'all' (todas as fontes elegíveis juntas), "
            "'single' (cada fonte individualmente), "
            "'combinations' (pares e trincas), "
            "'power_set' (todos os 2^K - 1 subconjuntos)."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["MLP", "TabNet"],
        choices=["MLP", "TabNet"],
        help="Modelos a avaliar (default: MLP TabNet).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
        help="Seeds de reprodutibilidade (default: 42 123 2026).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Número de épocas de treinamento/pré-treinamento (default: 30).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tamanho de lote (default: 64).",
    )
    parser.add_argument(
        "--balance-strategy",
        type=str,
        default="smote",
        choices=["smote", "none"],
        help="Estratégia de balanceamento no treino do target (default: smote).",
    )
    parser.add_argument(
        "--smote-insufficient-policy",
        type=str,
        default="fail_fold",
        choices=["fail_fold", "partial_smote"],
        help=(
            "Política explícita quando uma classe no treino possui < 2 amostras: "
            "'fail_fold' interrompe e registra o fold como inválido; "
            "'partial_smote' aplica SMOTE apenas às classes com >= 2 amostras e registra as demais."
        ),
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Limita o número de folds LOCO por target (útil para validação rápida).",
    )
    parser.add_argument(
        "--allow-missing-manifest",
        action="store_true",
        help="Não bloqueia um dataset caso normalization_manifest.csv não esteja presente (apenas para testes sintéticos).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Diretório raiz para a variante detrended_rms (default: data/processed_tl/detrended_rms).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Diretório base de saída (default: results/tl_cross_dataset_mlp_tabnet).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    from sklearn.preprocessing import LabelEncoder, StandardScaler

    variant_root = (
        Path(args.data_root)
        if args.data_root
        else PROJECT_ROOT / "data" / "processed_tl" / "detrended_rms"
    )
    base_out_root = (
        Path(args.output_root)
        if args.output_root
        else PROJECT_ROOT / "results" / "tl_cross_dataset_mlp_tabnet"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "experiment.log"
    logger = logging.getLogger(f"tl_cross_dataset_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("=" * 80)
    logger.info("INICIANDO EXPERIMENTO CROSS-DATASET TL: MLP vs. TabNet (detrended_rms + SMOTE)")
    logger.info("=" * 80)
    logger.info(f"Diretório de entrada (detrended_rms): {variant_root}")
    logger.info(f"Diretório de saída                 : {run_dir}")
    logger.info(f"Targets solicitados                : {args.targets}")
    logger.info(f"Sources solicitados                : {args.sources}")
    logger.info(f"Modos de fontes                    : {args.source_modes}")
    logger.info(f"Modelos                            : {args.models}")
    logger.info(f"Seeds                              : {args.seeds}")
    logger.info(f"Balanceamento                      : {args.balance_strategy} (policy={args.smote_insufficient_policy})")
    logger.info(f"Épocas / Batch size                : {args.epochs} / {args.batch_size}")
    if args.max_folds is not None:
        logger.info(f"Limite de folds LOCO (--max-folds) : {args.max_folds}")

    # -------------------------------------------------------------------------
    # 1. Auditoria de Manifesto, Carregamento e Extração de Atributos
    # -------------------------------------------------------------------------
    manifest_records = load_detrended_rms_manifest(variant_root)
    require_manifest = not args.allow_missing_manifest

    audit_rows: List[Dict[str, Any]] = []
    raw_datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    label_encoders: Dict[str, LabelEncoder] = {}
    num_classes_all: Dict[str, int] = {}

    # Evita carregar duas vezes caso algum dataset apareça em ambas as listas
    all_requested: List[Tuple[str, str]] = []
    for s in args.sources:
        all_requested.append((s, "source"))
    for t in args.targets:
        all_requested.append((t, "target"))

    eligible_sources: List[str] = []
    eligible_targets: List[str] = []

    for ds_name, role in all_requested:
        logger.info(f"Auditando e extraindo atributos de [{role.upper()}] {ds_name}...")
        data_tuple, audit_row = audit_and_extract_dataset(
            dataset_name=ds_name,
            role=role,
            variant_root=variant_root,
            manifest_records=manifest_records,
            require_manifest=require_manifest,
        )
        audit_rows.append(audit_row)

        if not audit_row["eligible"] or data_tuple is None:
            logger.warning(
                f"  -> Dataset {ds_name} ({role}) INELEGÍVEL: {audit_row['status_reason']}"
            )
            continue

        X_ds, y_ds, conds_ds = data_tuple
        raw_datasets[ds_name] = (X_ds, y_ds, conds_ds)
        le = LabelEncoder()
        le.fit(y_ds)
        label_encoders[ds_name] = le
        num_classes_all[ds_name] = int(len(le.classes_))

        logger.info(
            f"  -> OK: {len(X_ds)} janelas | {audit_row['n_conditions']} condições | "
            f"{audit_row['n_classes']} classes: {audit_row['class_distribution']}"
        )
        if role == "source" and ds_name not in eligible_sources:
            eligible_sources.append(ds_name)
        elif role == "target" and ds_name not in eligible_targets:
            eligible_targets.append(ds_name)

    df_audit = pd.DataFrame(audit_rows)
    audit_csv_path = run_dir / "dataset_audit.csv"
    df_audit.to_csv(audit_csv_path, index=False)
    logger.info(f"Relatório de auditoria de datasets salvo em: {audit_csv_path}")

    # Salva configuração completa antes de iniciar os loops
    code_id = get_code_identification()
    config_payload = {
        "timestamp": timestamp,
        "preprocessing_variant": "detrended_rms",
        "data_root": str(variant_root),
        "output_dir": str(run_dir),
        "requested_targets": args.targets,
        "requested_sources": args.sources,
        "eligible_targets": eligible_targets,
        "eligible_sources": eligible_sources,
        "source_modes": args.source_modes,
        "models": args.models,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "balance_strategy": args.balance_strategy,
        "smote_insufficient_policy": args.smote_insufficient_policy,
        "max_folds": args.max_folds,
        "label_encoding_policy": (
            "Codificação local por domínio (LabelEncoder próprio por dataset). "
            "Para o MLP, cada domínio possui cabeça linear independente (HybridDLModel.heads[domain]). "
            "Para o TabNet, as fontes são usadas em pré-treinamento não supervisionado (TabNetPretrainer) "
            "e o TabNetClassifier é ajustado exclusivamente no vocabulário de classes do target."
        ),
        "hyperparameters": {
            "MLP": {
                "architecture": "MLPEncoder(141->256->128->64) + Bottleneck(64->128->64) + DomainHead(64->C_d)",
                "optimizer_joint": "AdamW(lr=1e-3, weight_decay=1e-4)",
                "scheduler_joint": "StepLR(step_size=10, gamma=0.5)",
                "epochs_joint": args.epochs,
                "optimizer_finetune": "AdamW(lr=5e-4, weight_decay=1e-4)",
                "epochs_finetune": max(1, args.epochs // 2),
                "batch_size": args.batch_size,
            },
            "TabNet": {
                "pretrainer": "TabNetPretrainer(n_d=16, n_a=16, n_steps=4, lr=2e-2, mask_type='entmax', pretraining_ratio=0.8)",
                "classifier": "TabNetClassifier(n_d=16, n_a=16, n_steps=4, lr=2e-2, mask_type='sparsemax', StepLR(10, 0.9))",
                "epochs_pretrain": args.epochs,
                "epochs_finetune": args.epochs,
                "batch_size": args.batch_size,
                "virtual_batch_size": min(32, args.batch_size),
            },
        },
        "dataset_class_semantics": {
            k: v
            for k, v in DATASET_CLASS_SEMANTICS.items()
            if k in set(args.targets) | set(args.sources)
        },
        "code_identification": code_id,
    }
    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)

    if not eligible_sources:
        logger.error("Nenhuma fonte elegível disponível. Encerrando execução.")
        return 1
    if not eligible_targets:
        logger.error("Nenhum dataset de destino (target) elegível disponível. Encerrando execução.")
        return 1

    # -------------------------------------------------------------------------
    # 2. Preparação das Fontes (StandardScaler local por fonte)
    # -------------------------------------------------------------------------
    scaled_sources: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for s_name in eligible_sources:
        X_s, y_s, _ = raw_datasets[s_name]
        scaler_s = StandardScaler()
        X_s_scaled = scaler_s.fit_transform(X_s).astype(np.float32)
        y_s_enc = label_encoders[s_name].transform(y_s)
        scaled_sources[s_name] = (X_s_scaled, y_s_enc)

    # -------------------------------------------------------------------------
    # 3. Construção das Combinações de Fontes e Loop Experimental
    # -------------------------------------------------------------------------
    all_source_combos = build_source_combinations(eligible_sources, args.source_modes)
    logger.info(
        f"Total de combinações de fontes geradas: {len(all_source_combos)} -> "
        f"{[format_source_combo(c) for _, c in all_source_combos]}"
    )

    results_list: List[Dict[str, Any]] = []
    results_csv_path = run_dir / "results.csv"
    tabnet_pretrainer_cache: Dict[Tuple[str, int, int, int], Any] = {}

    for target_name in eligible_targets:
        X_t_all, y_t_all_str, conds_t_all = raw_datasets[target_name]
        le_target = label_encoders[target_name]
        y_t_all = le_target.transform(y_t_all_str)
        target_class_names = [str(c) for c in le_target.classes_]
        n_target_classes = len(target_class_names)

        unique_conds = sorted(np.unique(conds_t_all))
        if args.max_folds is not None:
            unique_conds = unique_conds[: args.max_folds]

        logger.info("=" * 80)
        logger.info(
            f"TARGET: {target_name} | Folds LOCO a executar: {len(unique_conds)} | "
            f"Classes ({n_target_classes}): {target_class_names}"
        )
        logger.info("=" * 80)

        for source_mode_label, raw_combo in all_source_combos:
            # Remove o próprio target caso ele apareça na combinação de fontes
            active_combo = tuple(s for s in raw_combo if s != target_name)
            if not active_combo:
                logger.warning(
                    f"Pulando combinação {raw_combo} para target {target_name} (vazia após excluir o próprio target)."
                )
                continue

            combo_str = format_source_combo(active_combo)
            source_subset = {s: scaled_sources[s] for s in active_combo}
            X_pretrain_stack = np.vstack([scaled_sources[s][0] for s in active_combo])

            num_classes_dict = {s: num_classes_all[s] for s in active_combo}
            num_classes_dict[target_name] = n_target_classes

            logger.info(
                f"--- Target: {target_name} | Modo: {source_mode_label} | Fontes ({len(active_combo)}): {combo_str} ---"
            )

            for seed in args.seeds:
                # Se TabNet estiver entre os modelos, prepara/recupera o TabNetPretrainer para (combo_str, seed)
                unsupervised_tabnet = None
                if "TabNet" in args.models:
                    unsupervised_tabnet = build_or_get_tabnet_pretrainer(
                        cache=tabnet_pretrainer_cache,
                        source_combo_key=combo_str,
                        X_pretrain=X_pretrain_stack,
                        pretrain_epochs=args.epochs,
                        batch_size=args.batch_size,
                        seed=seed,
                    )

                for test_cond in unique_conds:
                    test_mask = conds_t_all == test_cond
                    train_mask = ~test_mask

                    X_target_train = X_t_all[train_mask]
                    y_target_train = y_t_all[train_mask]
                    X_target_test = X_t_all[test_mask]
                    y_target_test = y_t_all[test_mask]

                    # Isolamento estrito do fold de teste:
                    # StandardScaler ajustado APENAS em X_target_train
                    scaler_t = StandardScaler()
                    if len(X_target_train) > 0:
                        X_target_train_scaled = scaler_t.fit_transform(X_target_train).astype(np.float32)
                        X_target_test_scaled = (
                            scaler_t.transform(X_target_test).astype(np.float32)
                            if len(X_target_test) > 0
                            else np.empty((0, X_t_all.shape[1]), dtype=np.float32)
                        )
                    else:
                        X_target_train_scaled = np.empty((0, X_t_all.shape[1]), dtype=np.float32)
                        X_target_test_scaled = np.empty((0, X_t_all.shape[1]), dtype=np.float32)

                    # Aplica SMOTE exclusivamente ao split de treinamento escalado do fold
                    X_train_bal, y_train_bal, smote_audit = balance_target_with_audit(
                        X_train_scaled=X_target_train_scaled,
                        y_train_encoded=y_target_train,
                        class_names=target_class_names,
                        strategy=args.balance_strategy,
                        seed=seed,
                        insufficient_policy=args.smote_insufficient_policy,
                    )

                    # Se o balanceamento interrompeu o fold (ex.: classe com < 2 amostras),
                    # registra a falha explicitamente para ambos os modelos sem forçar treinamento inválido
                    if X_train_bal is None or y_train_bal is None:
                        logger.warning(
                            f"  [Seed {seed} | Fold {test_cond}] FOLD INTERROMPIDO: "
                            f"{smote_audit['balance_status']} - {smote_audit['balance_reason']}"
                        )
                        for m_name in args.models:
                            results_list.append(
                                {
                                    "Target": target_name,
                                    "Source_Mode": source_mode_label,
                                    "Source_Combo": combo_str,
                                    "N_Sources": len(active_combo),
                                    "Model": m_name,
                                    "Variant": "detrended_rms",
                                    "Balance": args.balance_strategy,
                                    "Seed": seed,
                                    "Fold": str(test_cond),
                                    "Fold_Status": smote_audit["balance_status"],
                                    "Status_Reason": smote_audit["balance_reason"],
                                    "N_Train_Before": smote_audit["n_train_before"],
                                    "N_Train_After": smote_audit["n_train_after"],
                                    "Class_Counts_Before": json.dumps(smote_audit["counts_before"], ensure_ascii=False),
                                    "Class_Counts_After": json.dumps(smote_audit["counts_after"], ensure_ascii=False),
                                    "SMOTE_Params": json.dumps(smote_audit["smote_params"], ensure_ascii=False),
                                    "Unbalanced_Classes": json.dumps(smote_audit["unbalanced_classes"], ensure_ascii=False),
                                    "N_Test_Samples": int(len(y_target_test)),
                                    "N_Test_Classes_Present": int(len(np.unique(y_target_test))),
                                    "Macro F1": np.nan,
                                    "ACC": np.nan,
                                    "Bal Acc": np.nan,
                                    "ROC-AUC": np.nan,
                                    "ROC_AUC_Valid": False,
                                    "ROC_AUC_Note": "Fold interrompido antes do treinamento",
                                }
                            )
                        pd.DataFrame(results_list).to_csv(results_csv_path, index=False)
                        continue

                    # Avalia cada modelo sobre exatamente a mesma partição balanceada (X_train_bal, y_train_bal)
                    for m_name in args.models:
                        try:
                            if m_name == "MLP":
                                preds, probs = train_eval_mlp_fold(
                                    source_tensors=source_subset,
                                    X_target_train_bal=X_train_bal,
                                    y_target_train_bal=y_train_bal,
                                    X_target_test_scaled=X_target_test_scaled,
                                    num_classes_dict=num_classes_dict,
                                    target_name=target_name,
                                    epochs=args.epochs,
                                    batch_size=args.batch_size,
                                    seed=seed,
                                )
                            elif m_name == "TabNet":
                                preds, probs = train_eval_tabnet_fold(
                                    unsupervised_model=unsupervised_tabnet,
                                    X_target_train_bal=X_train_bal,
                                    y_target_train_bal=y_train_bal,
                                    X_target_test_scaled=X_target_test_scaled,
                                    n_target_classes=n_target_classes,
                                    finetune_epochs=args.epochs,
                                    batch_size=args.batch_size,
                                    seed=seed,
                                )
                            else:
                                raise ValueError(f"Modelo desconhecido: {m_name}")

                            metrics_res = compute_fold_metrics(
                                y_true=y_target_test,
                                y_pred=preds,
                                probs=probs,
                            )

                            auc_str = (
                                f"{metrics_res['ROC-AUC']:.4f}"
                                if metrics_res["roc_auc_valid"]
                                else f"NaN ({metrics_res['roc_auc_note']})"
                            )
                            logger.info(
                                f"  [{target_name} | {combo_str} | {m_name} | Seed {seed} | Fold {test_cond}] "
                                f"Train {smote_audit['n_train_before']}->{smote_audit['n_train_after']} | "
                                f"F1={metrics_res['Macro F1']:.4f} | ACC={metrics_res['ACC']:.4f} | "
                                f"BalAcc={metrics_res['Bal Acc']:.4f} | AUC={auc_str}"
                            )

                            results_list.append(
                                {
                                    "Target": target_name,
                                    "Source_Mode": source_mode_label,
                                    "Source_Combo": combo_str,
                                    "N_Sources": len(active_combo),
                                    "Model": m_name,
                                    "Variant": "detrended_rms",
                                    "Balance": args.balance_strategy,
                                    "Seed": seed,
                                    "Fold": str(test_cond),
                                    "Fold_Status": "OK",
                                    "Status_Reason": smote_audit["balance_reason"],
                                    "N_Train_Before": smote_audit["n_train_before"],
                                    "N_Train_After": smote_audit["n_train_after"],
                                    "Class_Counts_Before": json.dumps(smote_audit["counts_before"], ensure_ascii=False),
                                    "Class_Counts_After": json.dumps(smote_audit["counts_after"], ensure_ascii=False),
                                    "SMOTE_Params": json.dumps(smote_audit["smote_params"], ensure_ascii=False),
                                    "Unbalanced_Classes": json.dumps(smote_audit["unbalanced_classes"], ensure_ascii=False),
                                    "N_Test_Samples": metrics_res["n_test_samples"],
                                    "N_Test_Classes_Present": metrics_res["n_test_classes_present"],
                                    "Macro F1": metrics_res["Macro F1"],
                                    "ACC": metrics_res["ACC"],
                                    "Bal Acc": metrics_res["Bal Acc"],
                                    "ROC-AUC": metrics_res["ROC-AUC"],
                                    "ROC_AUC_Valid": metrics_res["roc_auc_valid"],
                                    "ROC_AUC_Note": metrics_res["roc_auc_note"],
                                }
                            )
                        except Exception as exc:
                            logger.error(
                                f"  [ERRO] Falha em {target_name} | {combo_str} | {m_name} | Seed {seed} | Fold {test_cond}: {exc}"
                            )
                            results_list.append(
                                {
                                    "Target": target_name,
                                    "Source_Mode": source_mode_label,
                                    "Source_Combo": combo_str,
                                    "N_Sources": len(active_combo),
                                    "Model": m_name,
                                    "Variant": "detrended_rms",
                                    "Balance": args.balance_strategy,
                                    "Seed": seed,
                                    "Fold": str(test_cond),
                                    "Fold_Status": "ERROR_MODEL_EXECUTION",
                                    "Status_Reason": str(exc),
                                    "N_Train_Before": smote_audit["n_train_before"],
                                    "N_Train_After": smote_audit["n_train_after"],
                                    "Class_Counts_Before": json.dumps(smote_audit["counts_before"], ensure_ascii=False),
                                    "Class_Counts_After": json.dumps(smote_audit["counts_after"], ensure_ascii=False),
                                    "SMOTE_Params": json.dumps(smote_audit["smote_params"], ensure_ascii=False),
                                    "Unbalanced_Classes": json.dumps(smote_audit["unbalanced_classes"], ensure_ascii=False),
                                    "N_Test_Samples": int(len(y_target_test)),
                                    "N_Test_Classes_Present": int(len(np.unique(y_target_test))),
                                    "Macro F1": np.nan,
                                    "ACC": np.nan,
                                    "Bal Acc": np.nan,
                                    "ROC-AUC": np.nan,
                                    "ROC_AUC_Valid": False,
                                    "ROC_AUC_Note": f"Erro de execução: {exc}",
                                }
                            )

                    # Salvamento incremental a cada fold
                    pd.DataFrame(results_list).to_csv(results_csv_path, index=False)

    # -------------------------------------------------------------------------
    # 4. Geração dos Resumos Finais
    # -------------------------------------------------------------------------
    df_final = pd.DataFrame(results_list)
    df_final.to_csv(results_csv_path, index=False)
    summary_paths = generate_summaries(df_final, run_dir)

    logger.info("=" * 80)
    logger.info("EXPERIMENTO CONCLUÍDO COM SUCESSO")
    logger.info(f"Resultados detalhados por fold : {results_csv_path}")
    for name, p in summary_paths.items():
        logger.info(f"Resumo [{name}] : {p}")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

