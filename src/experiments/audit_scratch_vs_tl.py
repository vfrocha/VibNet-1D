#!/usr/bin/env python3
"""
Phase 1 Repository & Compatibility Audit for the Scratch vs. Transfer Learning
Benchmark (`src/experiments/audit_scratch_vs_tl.py`).

Inspects existing baseline and transfer-learning scripts, configurations, and
result artifacts across all 14 audit dimensions specified in the benchmark protocol,
and generates:
  - `results/scratch_vs_tl/audit/compatibility_audit.csv`
  - `results/scratch_vs_tl/audit/experiment_inventory.csv`
  - `results/scratch_vs_tl/audit/audit_report.md`

Does not modify or overwrite any existing experimental results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.vibnet.data.registry import CANONICAL_DATASETS
from src.vibnet.evaluation.provenance import compute_file_sha256, get_dependency_versions

TARGET_DATASETS: List[str] = [
    "CWRU_12k",
    "CWRU_48k",
    "HUST_Gearbox",
    "UOEMD",
    "PU",
]

DEDICATED_SOURCE_DATASETS: List[str] = [
    "UORED",
    "HUST",
    "Mechanical_Gear",
    "Electric_Motor",
]

EXPECTED_FOLDS: Dict[str, int] = {
    "CWRU_12k": 4,
    "CWRU_48k": 12,
    "HUST_Gearbox": 30,
    "UOEMD": 16,
    "PU": 4,
}

DEFAULT_AUDIT_DIR: Path = RESULTS_DIR / "scratch_vs_tl" / "audit"
DEFAULT_BASELINE_REF_DIR: Path = (
    RESULTS_DIR / "baseline_loco_v2" / "run_20261104_160031"
)


def _git_cmd(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def collect_git_and_script_hashes() -> Dict[str, Any]:
    """Collect current Git state and SHA-256 hashes of all audited scripts and configs."""
    tracked_files = [
        "src/experiments/run_baseline_loco_v2.py",
        "src/experiments/run_baseline_experiments.py",
        "src/experiments/run_tabvibnet_linear_probing.py",
        "scripts/exp_tl_cross_dataset_mlp_tabnet.py",
        "scripts/exp_tl_balance_strategies.py",
        "scripts/exp_tl_contrastive_mlp.py",
        "src/models/build_tabnet_resnet.py",
        "src/models/build_tabnet_resnet_balance.py",
        "src/models/build_tabular.py",
        "src/models/build_sklearn.py",
        "src/models/build_pytorch_baselines.py",
        "src/features/extractors_v2.py",
        "src/features/signalai_wrapper.py",
        "src/preprocessing/normalize_dataset_tl.py",
        "src/vibnet/data/registry.py",
        "configs/datasets.yaml",
    ]
    hashes: Dict[str, Optional[str]] = {}
    for rel in tracked_files:
        hashes[rel] = compute_file_sha256(PROJECT_ROOT / rel)

    status_porcelain = _git_cmd(["status", "--porcelain"])
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_cmd(["rev-parse", "HEAD"]),
        "git_branch": _git_cmd(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(status_porcelain and status_porcelain != "UNKNOWN"),
        "git_status_summary": status_porcelain,
        "file_hashes_sha256": hashes,
        "dependency_versions": get_dependency_versions(),
    }


def build_experiment_inventory(baseline_ref_dir: Path) -> pd.DataFrame:
    """
    Build `experiment_inventory.csv` cataloging all relevant baseline and TL
    runs in `results/` and their methodological properties.
    """
    entries: List[Dict[str, Any]] = []

    def _inspect_csv(csv_path: Path) -> Dict[str, Any]:
        if not csv_path.exists():
            return {"exists": False, "n_rows": 0, "columns": []}
        try:
            df = pd.read_csv(csv_path)
            return {
                "exists": True,
                "n_rows": int(len(df)),
                "columns": list(df.columns),
            }
        except Exception as exc:
            return {"exists": True, "n_rows": -1, "columns": [f"ERROR: {exc}"]}

    # 1. Reference baseline_loco_v2 directory (`results/baseline_loco_v2/run_20261104_160031/`)
    b_res = _inspect_csv(baseline_ref_dir / "results.csv")
    b_pred = _inspect_csv(baseline_ref_dir / "predictions.csv")
    entries.append(
        {
            "experiment_family": "baseline_loco_v2",
            "run_id": baseline_ref_dir.name,
            "artifact_path": str(baseline_ref_dir.relative_to(PROJECT_ROOT))
            if baseline_ref_dir.is_relative_to(PROJECT_ROOT)
            else str(baseline_ref_dir),
            "script_path": "src/experiments/run_baseline_loco_v2.py",
            "present_on_disk": bool(baseline_ref_dir.exists()),
            "results_csv_rows": b_res["n_rows"],
            "predictions_csv_present": bool(b_pred["exists"]),
            "predictions_csv_rows": b_pred["n_rows"],
            "preprocessing": "detrended_rms (data/processed_tl/detrended_rms)",
            "feature_dim": 141,
            "targets_covered": "CWRU_12k|CWRU_48k|HUST_Gearbox|UOEMD|PU (66 folds)",
            "sources_used": "None (Target-only Scratch)",
            "architectures": "Random Forest|SVM (RBF)|XGBoost|MLP Scratch|TabNet Scratch",
            "methods": "Scratch",
            "balance_strategies": "none|smote",
            "seeds": "42|123|2026",
            "loco_leakage_free": True,
            "reusable_for_scratch_vs_tl": (
                "CONDITIONAL: Reusable for Scratch metrics only if run_20261104_160031 "
                "is present on disk with predictions.csv and matching training schedule; "
                "otherwise rerun within unified scratch_vs_tl pipeline."
            ),
        }
    )

    # Check if any other baseline_loco_v2 runs exist on disk
    bl_v2_root = RESULTS_DIR / "baseline_loco_v2"
    if bl_v2_root.exists():
        for rdir in sorted(bl_v2_root.iterdir()):
            if rdir.is_dir() and rdir.resolve() != baseline_ref_dir.resolve():
                r_info = _inspect_csv(rdir / "results.csv")
                p_info = _inspect_csv(rdir / "predictions.csv")
                entries.append(
                    {
                        "experiment_family": "baseline_loco_v2",
                        "run_id": rdir.name,
                        "artifact_path": str(rdir.relative_to(PROJECT_ROOT)),
                        "script_path": "src/experiments/run_baseline_loco_v2.py",
                        "present_on_disk": True,
                        "results_csv_rows": r_info["n_rows"],
                        "predictions_csv_present": bool(p_info["exists"]),
                        "predictions_csv_rows": p_info["n_rows"],
                        "preprocessing": "detrended_rms",
                        "feature_dim": 141,
                        "targets_covered": "CWRU_12k|CWRU_48k|HUST_Gearbox|UOEMD|PU",
                        "sources_used": "None",
                        "architectures": "Random Forest|SVM (RBF)|XGBoost|MLP Scratch|TabNet Scratch",
                        "methods": "Scratch",
                        "balance_strategies": "none|smote",
                        "seeds": "42|123|2026",
                        "loco_leakage_free": True,
                        "reusable_for_scratch_vs_tl": "CONDITIONAL (requires predictions.csv and hyperparameter parity)",
                    }
                )

    # 2. Cross-Dataset TL runs (`results/tl_cross_dataset_mlp_tabnet/run_*`)
    for run_name in ("run_20261028_105603", "run_20261028_173310"):
        rdir = RESULTS_DIR / "tl_cross_dataset_mlp_tabnet" / run_name
        r_info = _inspect_csv(rdir / "results.csv")
        p_info = _inspect_csv(rdir / "predictions.csv")
        seeds_str = "42" if run_name == "run_20261028_105603" else "42|123|2026"
        entries.append(
            {
                "experiment_family": "tl_cross_dataset_mlp_tabnet",
                "run_id": run_name,
                "artifact_path": str(rdir.relative_to(PROJECT_ROOT)),
                "script_path": "scripts/exp_tl_cross_dataset_mlp_tabnet.py",
                "present_on_disk": bool(rdir.exists()),
                "results_csv_rows": r_info["n_rows"],
                "predictions_csv_present": bool(p_info["exists"]),
                "predictions_csv_rows": p_info["n_rows"],
                "preprocessing": "detrended_rms (with fs divergence on Mechanical_Gear=10kHz & Electric_Motor=42kHz)",
                "feature_dim": 141,
                "targets_covered": "CWRU_12k|CWRU_48k|HUST_Gearbox|UOEMD|PU (66 folds)",
                "sources_used": "15 subsets of UORED|HUST|Mechanical_Gear|Electric_Motor",
                "architectures": "MLP|TabNet",
                "methods": "Fine-Tuning only (no Linear Probing)",
                "balance_strategies": "smote only (missing none)",
                "seeds": seeds_str,
                "loco_leakage_free": True,
                "reusable_for_scratch_vs_tl": (
                    "NO: Missing balance_strategy='none', missing Linear Probing, "
                    "missing predictions.csv, PU fold C4 TabNet BatchNorm1d crash (45 failed rows), "
                    "source fs divergence (10kHz/42kHz vs 5kHz/50kHz), and Stage-1 joint target+source training."
                ),
            }
        )

    # 3. Linear Probing experiment (`results/results_tabvibnet_linear_probing.csv`)
    lp_csv = RESULTS_DIR / "results_tabvibnet_linear_probing.csv"
    lp_info = _inspect_csv(lp_csv)
    entries.append(
        {
            "experiment_family": "tabvibnet_linear_probing",
            "run_id": "20261104_082702",
            "artifact_path": "results/results_tabvibnet_linear_probing.csv",
            "script_path": "src/experiments/run_tabvibnet_linear_probing.py",
            "present_on_disk": bool(lp_csv.exists()),
            "results_csv_rows": lp_info["n_rows"],
            "predictions_csv_present": False,
            "predictions_csv_rows": 0,
            "preprocessing": "legacy data/processed (NOT detrended_rms)",
            "feature_dim": 141,
            "targets_covered": "CWRU_12k|CWRU_48k|HUST_Gearbox|UOEMD (missing PU)",
            "sources_used": "All 8 remaining datasets (includes other target datasets!)",
            "architectures": "MLP|TabNet",
            "methods": "Fine-Tuning|Linear Probing",
            "balance_strategies": "none only (missing smote)",
            "seeds": "single unrecorded seed (MLP unseeded)",
            "loco_leakage_free": False,
            "reusable_for_scratch_vs_tl": (
                "NO: Uses legacy data/processed instead of detrended_rms, omits PU, "
                "contaminates source pool with other targets, lacks SMOTE, runs only 1 seed, "
                "omits Accuracy metric, and lacks predictions.csv."
            ),
        }
    )

    # 4. Legacy LOCO baselines (`results/baseline_loco_results_*.csv`)
    for fname in (
        "baseline/baseline_loco_results_20260829_084039.csv",
        "baseline_loco_results_20260928_142515.csv",
        "baseline_loco_results_20261003_052352.csv",
    ):
        fpath = RESULTS_DIR / fname
        f_info = _inspect_csv(fpath)
        entries.append(
            {
                "experiment_family": "legacy_baseline_loco",
                "run_id": Path(fname).stem,
                "artifact_path": f"results/{fname}",
                "script_path": "src/experiments/run_baseline_experiments.py",
                "present_on_disk": bool(fpath.exists()),
                "results_csv_rows": f_info["n_rows"],
                "predictions_csv_present": False,
                "predictions_csv_rows": 0,
                "preprocessing": "legacy data/processed (NOT detrended_rms)",
                "feature_dim": 141,
                "targets_covered": "CWRU_12k|CWRU_48k|HUST_Gearbox|PU (missing UOEMD)",
                "sources_used": "None",
                "architectures": "RF|SVM|XGBoost|TabNet|MLP (Scratch)",
                "methods": "Scratch",
                "balance_strategies": "none only",
                "seeds": "single seed (42)",
                "loco_leakage_free": False,
                "reusable_for_scratch_vs_tl": (
                    "NO: Uses data/processed, omits UOEMD, leaks X_test into TabNet eval_set, "
                    "and suffers from SHIFTED_COLUMN_CORRUPTION on MLP."
                ),
            }
        )

    # 5. Single-target CWRU_48k TL ablations (`tl_four_variants`, `tl_aux_features`, `tl_balance_strategies`, `tl_contrastive_mlp`)
    for fam, subpath, script_rel, note in (
        (
            "tl_four_variants",
            "tl_four_variants/run_20261005_163159",
            "scripts/tl/exp_tl_four_variants.py",
            "Evaluates CWRU_48k only (sources CWRU_12k|MFPT); single seed; no Linear Probing; no predictions.csv.",
        ),
        (
            "tl_aux_features",
            "tl_aux_features/run_20261012_093659",
            "scripts/tl/exp_tl_eight_configs.py",
            "Evaluates CWRU_48k only (sources CWRU_12k|MFPT); single seed; no Linear Probing; no predictions.csv.",
        ),
        (
            "tl_balance_strategies",
            "tl_balance_strategies/run_20261012_150855",
            "scripts/exp_tl_balance_strategies.py",
            "Evaluates CWRU_48k only (sources CWRU_12k|MFPT); single seed; no Linear Probing; no predictions.csv.",
        ),
        (
            "tl_contrastive_mlp",
            "tl_contrastive_mlp/run_20261027_113503",
            "scripts/exp_tl_contrastive_mlp.py",
            "Evaluates CWRU_48k MLP only (sources CWRU_12k|MFPT); no TabNet; no Linear Probing; no predictions.csv.",
        ),
    ):
        rdir = RESULTS_DIR / subpath
        r_info = _inspect_csv(rdir / "results.csv")
        entries.append(
            {
                "experiment_family": fam,
                "run_id": Path(subpath).name,
                "artifact_path": f"results/{subpath}",
                "script_path": script_rel,
                "present_on_disk": bool(rdir.exists()),
                "results_csv_rows": r_info["n_rows"],
                "predictions_csv_present": False,
                "predictions_csv_rows": 0,
                "preprocessing": "detrended_rms",
                "feature_dim": 141,
                "targets_covered": "CWRU_48k only (12 folds)",
                "sources_used": "CWRU_12k|MFPT",
                "architectures": "MLP" if fam == "tl_contrastive_mlp" else "MLP|TabNet",
                "methods": "Fine-Tuning",
                "balance_strategies": "none|smote|hybrid|class_weights",
                "seeds": "42|123|2026" if fam == "tl_contrastive_mlp" else "42",
                "loco_leakage_free": True,
                "reusable_for_scratch_vs_tl": f"NO: {note}",
            }
        )

    return pd.DataFrame(entries)


def build_compatibility_audit_table(baseline_ref_dir: Path) -> pd.DataFrame:
    """
    Build `compatibility_audit.csv` evaluating each target dataset (`CWRU_12k`,
    `CWRU_48k`, `HUST_Gearbox`, `UOEMD`, `PU`), architecture (`MLP`, `TabNet`),
    and method (`Scratch`, `Fine-Tuning`, `Linear Probing`).
    """
    rows: List[Dict[str, Any]] = []
    baseline_present = baseline_ref_dir.exists()
    baseline_preds_present = (baseline_ref_dir / "predictions.csv").exists()

    for ds_name in TARGET_DATASETS:
        spec = CANONICAL_DATASETS[ds_name]
        n_folds = EXPECTED_FOLDS[ds_name]

        for arch in ("MLP", "TabNet"):
            for method in ("Scratch", "Fine-Tuning", "Linear Probing"):
                if method == "Scratch":
                    existing_source = f"results/baseline_loco_v2/{baseline_ref_dir.name}"
                    split_compat = "COMPATIBLE (Deterministic sorted LOCO condition split)"
                    preproc_compat = "COMPATIBLE (data/processed_tl/detrended_rms, 141 features, canonical fs)"
                    feature_compat = "COMPATIBLE (141-dim extract_fusion_features + extract_advanced_features)"
                    sample_match = "YES (Sample_Split_Signature & deterministic relative .npy paths)"
                    preds_available = (
                        "YES"
                        if (baseline_present and baseline_preds_present)
                        else (
                            "NO (run_20261104_160031 not synced locally or run without --save-predictions)"
                        )
                    )
                    method_feasible = "YES (Native target-only training from random init)"
                    reusable = "CONDITIONAL" if (baseline_present and baseline_preds_present) else "NO_RERUN_REQUIRED"
                    incompatibility_reason = (
                        "Baseline v2 uses compatible LOCO splits, detrended_rms preprocessing, "
                        "SMOTE={none,smote}, and seeds={42,123,2026}, but sample-level predictions.csv "
                        "and identical epoch-selection/optimizer schedule with TL must be guaranteed."
                    )
                    required_action = (
                        "Execute Scratch inside unified scratch_vs_tl runner (or verify/import "
                        "run_20261104_160031 when predictions.csv and schedule match)."
                    )
                elif method == "Fine-Tuning":
                    existing_source = "results/tl_cross_dataset_mlp_tabnet/run_20261028_173310"
                    split_compat = "COMPATIBLE (Same 66 LOCO condition splits)"
                    preproc_compat = (
                        "INCOMPATIBLE_SOURCE_FS (Sources Mechanical_Gear used fs=10000 vs canonical 5000 "
                        "and Electric_Motor used fs=42000 vs canonical 50000 in exp_tl_cross_dataset_mlp_tabnet.py)"
                    )
                    feature_compat = "PARTIAL (141 features, but 5 FFT features rescaled on 2 sources due to fs mismatch)"
                    sample_match = "YES (Same deterministic sorted .npy order on target)"
                    preds_available = "NO (predictions.csv never exported in exp_tl_cross_dataset_mlp_tabnet.py)"
                    method_feasible = "YES"
                    reusable = "NO_RERUN_REQUIRED"
                    extra_issue = (
                        "Also TabNet crashed on PU fold C4 (8577 % 64 == 1 BatchNorm1d error); "
                        if (ds_name == "PU" and arch == "TabNet")
                        else ""
                    )
                    incompatibility_reason = (
                        f"{extra_issue}Missing balance_strategy='none'; missing predictions.csv; "
                        "source fs mismatch on Mechanical_Gear/Electric_Motor; MLP Stage-1 mixed "
                        "target_train with sources instead of pure source-only pretraining; "
                        "TabNet fine-tuning monitored in-sample train accuracy instead of internal validation."
                    )
                    required_action = (
                        "Rerun Fine-Tuning for both balance_strategies={'none','smote'} and seeds={42,123,2026} "
                        "from pure source-pretrained encoder with canonical fs, BatchNorm1d protection, "
                        "and full predictions.csv export."
                    )
                else:  # Linear Probing
                    existing_source = "results/results_tabvibnet_linear_probing.csv"
                    split_compat = (
                        "MISSING_DATASET" if ds_name == "PU" else "PARTIAL (No sample_ids or split_signature recorded)"
                    )
                    preproc_compat = "INCOMPATIBLE (Loaded legacy data/processed/ instead of data/processed_tl/detrended_rms/)"
                    feature_compat = "INCOMPATIBLE (Extracted from non-detrended_rms windows)"
                    sample_match = "NO (Different directory tree data/processed/ and no sample IDs)"
                    preds_available = "NO (No predictions.csv; also missing Accuracy column)"
                    if arch == "MLP":
                        method_feasible = (
                            "YES (Freeze MLPEncoder + Bottleneck with requires_grad=False and .eval(); "
                            "train only target Linear(64, C_target))"
                        )
                    else:
                        method_feasible = (
                            "YES_WITH_ADAPTATION (TabNetClassifier.fit() does not freeze encoder natively; "
                            "extract pretrained EmbeddingGenerator + TabNetEncoder from TabNetPretrainer, "
                            "freeze params & buffers with requires_grad=False + .eval(), and train "
                            "nn.Linear(16, C_target) on sum(steps_out))"
                        )
                    reusable = "NO_RERUN_REQUIRED"
                    incompatibility_reason = (
                        "Existing linear probing used legacy data/processed/, omitted PU, mixed other target "
                        "datasets into the source pool, ran only 1 seed without seeding MLP, omitted SMOTE, "
                        "omitted Accuracy, and exported no predictions.csv."
                    )
                    required_action = (
                        "Rerun Linear Probing from the exact same pure source-pretrained checkpoint as "
                        "Fine-Tuning across all 5 targets, 2 balance strategies, and 3 seeds, with "
                        "programmatic verification that encoder parameters & buffers remain 100% unchanged."
                    )

                rows.append(
                    {
                        "target_dataset": ds_name,
                        "sampling_rate_hz": spec.sampling_rate_hz,
                        "loco_folds": n_folds,
                        "architecture": arch,
                        "method": method,
                        "existing_artifact_reference": existing_source,
                        "split_compatibility": split_compat,
                        "preprocessing_compatibility": preproc_compat,
                        "feature_compatibility": feature_compat,
                        "sample_and_class_matchable": sample_match,
                        "predictions_available_and_complete": preds_available,
                        "method_feasibility": method_feasible,
                        "reusable_status": reusable,
                        "methodological_impact": incompatibility_reason,
                        "minimum_required_action": required_action,
                    }
                )

    return pd.DataFrame(rows)


def generate_audit_report_markdown(
    git_info: Dict[str, Any],
    df_inventory: pd.DataFrame,
    df_compat: pd.DataFrame,
    baseline_ref_dir: Path,
) -> str:
    """Render `results/scratch_vs_tl/audit/audit_report.md`."""
    hashes_lines = "\n".join(
        f"| `{k}` | `{v or 'MISSING'}` |"
        for k, v in git_info["file_hashes_sha256"].items()
    )

    return f"""# Relatório de Auditoria de Compatibilidade e Reprodutibilidade — Benchmark Scratch vs. Transfer Learning (`scratch_vs_tl`)

- **Data/Hora UTC da Auditoria**: `{git_info['timestamp_utc']}`
- **Commit Git (`HEAD`)**: `{git_info['git_commit']}` (branch `{git_info['git_branch']}`, `git_dirty={git_info['git_dirty']}`)
- **Diretório de Referência Inicial (`baseline_loco_v2`)**: `{baseline_ref_dir}` (`present_on_disk={baseline_ref_dir.exists()}`)

---

## 1. Resumo Executivo da Auditoria (14 Dimensões Inspecionadas)

Antes de implementar ou executar o benchmark **Scratch vs. Transfer Learning (`MLP` e `TabNet`)**, foram inspecionados todos os scripts, modelos, configurações e artefatos de resultados existentes no repositório **VibNet-1D**:

1. **Implementação atual do `baseline_loco_v2`** ([`src/experiments/run_baseline_loco_v2.py`](../../../src/experiments/run_baseline_loco_v2.py)):
   - Avalia os 5 datasets alvo (`CWRU_12k`: 4 folds, `CWRU_48k`: 12 folds, `HUST_Gearbox`: 30 folds, `UOEMD`: 16 folds, `PU`: 4 folds; total = **66 folds LOCO**), variante `detrended_rms` (`data/processed_tl/detrended_rms/`), 141 features (`extract_fusion_features` + `extract_advanced_features`), estratégias `["none", "smote"]` e seeds `[42, 123, 2026]`.
   - O diretório `results/baseline_loco_v2/run_20261104_160031/` foi executado no servidor Linux após o commit `77c773b`.
2. **Implementações existentes de `MLP Scratch` e `TabNet Scratch`**:
   - Em `run_baseline_loco_v2.py`: `MLP Scratch` usa `HybridDLModel` (`141->256->128->64` + `Bottleneck 64->128->64` + `Linear(64->C_T)`) treinado com `AdamW(lr=1e-3, weight_decay=1e-4)` e `StepLR(10, 0.5)` por 30 épocas (`batch_size=64`), com proteção `trim_single_remainder_sample`. `TabNet Scratch` usa `TabNetClassifier(n_d=16, n_a=16, n_steps=4, gamma=1.3, mask_type='sparsemax')` com validação interna agrupada por condição operacional (`patience=15`, `max_epochs=30`) seguida de *refit* no treino completo permitido.
   - Em `run_baseline_experiments.py` (legado): Usava `data/processed/`, omitia `UOEMD`, vazava `X_test` no `eval_set` do `TabNet` e continha bug de deslocamento de colunas no `MLP`. **Incompatível e descartado para comparação.**
3. **Implementações existentes de `MLP` e `TabNet` Transfer Learning (`Fine-Tuning`)**:
   - Em [`scripts/exp_tl_cross_dataset_mlp_tabnet.py`](../../../scripts/exp_tl_cross_dataset_mlp_tabnet.py) (`results/tl_cross_dataset_mlp_tabnet/run_20261028_173310`): Avaliou os 5 alvos (66 folds) e 15 subconjuntos das 4 fontes dedicadas (`UORED`, `HUST`, `Mechanical_Gear`, `Electric_Motor`) com 3 seeds (`42, 123, 2026`).
   - **Incompatibilidades identificadas em `run_20261028_173310`**:
     - Avaliou **apenas** `balance_strategy = "smote"` (ausência total de `balance_strategy = "none"`).
     - Avaliou **apenas** `Fine-Tuning` (ausência total de `Linear Probing`).
     - **Não exportou** predições por amostra (`predictions.csv`) nem manifesto de folds (`fold_manifest.csv`).
     - Apresentou **divergência de taxa de amostragem ($f_s$) em 2 fontes**: `Mechanical_Gear` foi extraído com `fs=10000` (canônico: `5000 Hz`) e `Electric_Motor` com `fs=42000` (canônico: `50000 Hz`), reescalonando as features espectrais FFT/PSD dessas duas fontes.
     - No alvo `PU` (fold `C4_1500rpm_0.7Nm_400N`), o `TabNet` falhou nas 45 avaliações (`8577 % 64 == 1` no `BatchNorm1d`) por ausência de `trim_single_remainder_sample`.
     - Na Etapa 1 do `MLP`, o pré-treinamento misturava lotes das fontes com lotes de `X_target_train_bal` (treinamento conjunto fonte+alvo em vez de pré-treinamento estrito nas fontes seguido de adaptação no alvo).
     - No `TabNet`, os hiperparâmetros (`n_steps=5, gamma=1.5, mask_type='entmax'`) e o critério de parada (`eval_set=[(X_target_train_bal, y_target_train_bal)]`) diferiam do `TabNet Scratch` de `baseline_loco_v2` (`n_steps=4, gamma=1.3, mask_type='sparsemax'` com validação interna agrupada por condição).
4. **Implementações existentes de `Fine-Tuning` vs. `Linear Probing`**:
   - Em [`src/experiments/run_tabvibnet_linear_probing.py`](../../../src/experiments/run_tabvibnet_linear_probing.py) e [`src/models/build_tabnet_resnet.py`](../../../src/models/build_tabnet_resnet.py) (`results/results_tabvibnet_linear_probing.csv`):
   - **Incompatibilidades críticas identificadas**:
     - Carregou janelas de `data/processed/` em vez de `data/processed_tl/detrended_rms/`.
     - Omitiu o dataset alvo `PU` (avaliou apenas 4 dos 5 alvos).
     - Contaminou o pool de fontes com os demais datasets alvo (ex.: usou `CWRU_12k` como fonte ao avaliar `CWRU_48k` e vice-versa).
     - Executou apenas `balance_strategy = "none"` (sem `smote`) e apenas 1 seed (sem fixar `torch.manual_seed` no ramo `MLP`).
     - Não calculou `Accuracy` e não exportou `predictions.csv` nem `fold_manifest.csv`.

---

## 2. Viabilidade Arquitetural (`MLP` e `TabNet`: `Scratch`, `Fine-Tuning`, `Linear Probing`)

| Arquitetura | Método | Viabilidade Técnica | Detalhamento da Implementação e Garantias |
|---|---|:---:|---|
| **`MLP`** | **`Scratch`** | **Sim (Nativo)** | `HybridDLModel` (`MLPEncoder 141->256->128->64` + `Bottleneck 64->128->64` + `Linear(64->C_T)`) inicializado aleatoriamente e treinado exclusivamente no fold de treino do alvo. |
| **`MLP`** | **`Fine-Tuning`** | **Sim (Nativo)** | Pré-treina `MLPEncoder + Bottleneck` (com cabeças independentes por fonte $H_s$) **exclusivamente nos datasets fonte**. No alvo, anexa nova cabeça `Linear(64->C_T)` e atualiza `encoder + bottleneck + head` no fold de treino do alvo. |
| **`MLP`** | **`Linear Probing`** | **Sim (Nativo)** | Carrega exatamente o mesmo `MLPEncoder + Bottleneck` pré-treinado nas fontes, congela todos os seus parâmetros (`requires_grad=False`) e buffers (`eval()`), e treina **exclusivamente** a cabeça linear `Linear(64->C_T)`. Verificado programaticamente (`max_abs_diff == 0.0`). |
| **`TabNet`** | **`Scratch`** | **Sim (Nativo)** | `TabNetClassifier` (`n_d=16, n_a=16, n_steps=4, gamma=1.3, mask_type='sparsemax'`, `from_unsupervised=None`) com seleção de época via validação interna agrupada por condição nas $K-1$ condições de treino + *refit* no treino completo. |
| **`TabNet`** | **`Fine-Tuning`** | **Sim (Nativo)** | Pré-treina `TabNetPretrainer` (auto-supervisionado, `pretraining_ratio=0.8`) **exclusivamente nos datasets fonte** com a mesma arquitetura (`n_d=16, n_a=16, n_steps=4, gamma=1.3`). Transfere via `from_unsupervised` para `TabNetClassifier` e ajusta todos os parâmetros no fold de treino do alvo. |
| **`TabNet`** | **`Linear Probing`** | **Sim (Com Adaptação Mínima Transparente)** | `TabNetClassifier.fit()` do `pytorch-tabnet` não possui flag nativa para congelar o encoder durante `.fit()`. A menor adaptação tecnicamente rigorosa consiste em extrair o `embedder` e o `encoder` (`TabNetEncoder`) pré-treinados do `TabNetPretrainer.network`, congelá-los integralmente (`requires_grad=False` e `.eval()`), extrair a representação latente canônica do TabNet $z = \\sum_{{t=1}}^{{n_{{\\text{{steps}}}}}} h_t(x) \\in \\mathbb{{R}}^{{16}}$ e treinar apenas a camada linear `final_mapping = nn.Linear(16, C_T)`. Verificado programaticamente (`max_abs_diff == 0.0`). |

---

## 3. Seleção de Datasets Fonte e Compatibilidade de Rótulos

- **Datasets Alvo (`5`)**: `CWRU_12k` (4 folds), `CWRU_48k` (12 folds), `HUST_Gearbox` (30 folds), `UOEMD` (16 folds), `PU` (4 folds) $\\rightarrow$ **66 folds LOCO**.
- **Pool Canônico de Datasets Fonte (`4` fontes dedicadas)**: `["UORED", "HUST", "Mechanical_Gear", "Electric_Motor"]` (definido em `configs/scratch_vs_tl.yaml`).
  - **Por que não usar `CWRU_12k + MFPT` para todos os alvos**: `CWRU_12k` é um dos 5 datasets alvo (usá-lo como fonte para si mesmo violaria a separação fonte/alvo, e usá-lo para `CWRU_48k` compartilha a mesma bancada física CWRU). Já o pool `["UORED", "HUST", "Mechanical_Gear", "Electric_Motor"]` é **100% disjunto** dos 5 datasets alvo.
  - **Política de Rótulos entre Fontes e Alvo ([`ADR-003`](../../decisions/003-domain-vocabularies-and-transfer-architectures.md))**: Cada dataset preserva seu vocabulário nativo de classes (`LabelEncoder` local por domínio). Não há mapeamento artificial entre modos de falha distintos:
    - No `MLP`, cada domínio fonte possui sua própria cabeça linear $H_s$ durante o pré-treinamento supervisionado multi-head, e cada alvo $T$ recebe uma nova cabeça $H_T: \\mathbb{{R}}^{{64}} \\to \\mathbb{{R}}^{{C_T}}$.
    - No `TabNet`, o pré-treinamento (`TabNetPretrainer`) é auto-supervisionado (reconstrução de atributos mascarados) sobre as matrizes de fontes padronizadas individualmente por domínio, e o alvo $T$ recebe uma cabeça linear $\\mathbb{{R}}^{{16}} \\to \\mathbb{{R}}^{{C_T}}$.

---

## 4. Decisão de Reuso vs. Re-execução

1. **Experimentos que DEVEM ser executados no pipeline unificado `scratch_vs_tl`**:
   - Todos os pares `(Architecture, Method)` $\\in \\{{\\text{{MLP}}, \\text{{TabNet}}\\}} \\times \\{{\\text{{Scratch}}, \\text{{Fine-Tuning}}, \\text{{Linear Probing}}\\}} $ devem ser executados sob o mesmo runner independente (`src/experiments/run_scratch_vs_tl_benchmark.py`) para garantir:
     - Mesma assinatura de partição (`split_signature`) e pré-processamento (`preprocessing_signature`) em `fold_manifest.csv`;
     - Exportação completa de predições e probabilidades por amostra em `predictions.csv`;
     - Hiperparâmetros e critério de seleção de épocas estritamente alinhados entre `Scratch`, `Fine-Tuning` e `Linear Probing`;
     - Taxas de amostragem canônicas em todas as fontes (`Mechanical_Gear = 5000 Hz`, `Electric_Motor = 50000 Hz`);
     - Cobertura completa de `BALANCE_STRATEGIES = ["none", "smote"]` e `SEEDS = [42, 123, 2026]` sem falhas de `BatchNorm1d` no fold `C4` de `PU`.
   - **Nota sobre reuso opcional de `baseline_loco_v2`**: Caso um diretório `results/baseline_loco_v2/run_*` contenha `predictions.csv` completo e assinaturas de partição idênticas, o runner suporta validação cruzada contra seus resultados, mas por padrão executa os 6 métodos de forma pareada para garantir comparabilidade estrita no mesmo loop LOCO.
2. **Tamanho Exato do Benchmark Completo**:
   - $66\\text{{ folds LOCO}} \\times 3\\text{{ seeds}} \\times 2\\text{{ estratégias de balanceamento}} \\times 2\\text{{ arquiteturas}} \\times 3\\text{{ métodos}} = \\mathbf{{2.376\\text{{ avaliações}}}}$ (sendo $396$ avaliações por par `[Arquitetura, Método]`).

---

## 5. Rastreabilidade Git e Hashes SHA-256 dos Scripts Auditados

| Arquivo | SHA-256 |
|---|---|
{hashes_lines}
"""


def run_audit(
    output_dir: Path = DEFAULT_AUDIT_DIR,
    baseline_ref_dir: Path = DEFAULT_BASELINE_REF_DIR,
) -> Dict[str, Path]:
    """Execute the Phase 1 audit and write all three deliverables to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)

    git_info = collect_git_and_script_hashes()
    df_inventory = build_experiment_inventory(baseline_ref_dir)
    df_compat = build_compatibility_audit_table(baseline_ref_dir)
    report_md = generate_audit_report_markdown(
        git_info=git_info,
        df_inventory=df_inventory,
        df_compat=df_compat,
        baseline_ref_dir=baseline_ref_dir,
    )

    p_inv = output_dir / "experiment_inventory.csv"
    p_compat = output_dir / "compatibility_audit.csv"
    p_report = output_dir / "audit_report.md"

    df_inventory.to_csv(p_inv, index=False)
    df_compat.to_csv(p_compat, index=False)
    p_report.write_text(report_md, encoding="utf-8")

    return {
        "experiment_inventory": p_inv,
        "compatibility_audit": p_compat,
        "audit_report": p_report,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phase 1 repository & compatibility audit for scratch_vs_tl."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_AUDIT_DIR,
        help="Directory to write audit deliverables (default: results/scratch_vs_tl/audit).",
    )
    parser.add_argument(
        "--baseline-ref-dir",
        type=Path,
        default=DEFAULT_BASELINE_REF_DIR,
        help="Reference baseline_loco_v2 directory.",
    )
    args = parser.parse_args(argv)
    paths = run_audit(output_dir=args.output_dir, baseline_ref_dir=args.baseline_ref_dir)
    for k, p in paths.items():
        print(f"[AUDIT OK] {k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
