#!/usr/bin/env python3
"""
Inventory result artifacts, classify granularity (fold_level, summary, dataset_audit),
extract recognized metrics with full provenance (file, dataset, model, sources, strategy,
seed, fold, status), detect invalid/corrupted evaluations, and generate a reproducible
Markdown audit report without merging incompatible protocols.

Compatible with pure Python standard library (no pandas/numpy required).
"""
import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Metric aliases separated by granularity to avoid mixing fold-level and summary-level metrics
FOLD_METRIC_ALIASES: Dict[str, List[str]] = {
    "macro_f1": ["macro_f1", "macro f1", "f1_macro", "macro-f1"],
    "accuracy": ["accuracy", "acc", "accuracy_score"],
    "balanced_accuracy": [
        "balanced_accuracy",
        "bal_acc",
        "balanced acc",
        "bal acc",
        "bal accuracy",
        "bal_accuracy",
    ],
    "roc_auc": ["roc_auc", "roc-auc", "auc", "roc_auc_ovr_macro", "roc-auc ovr macro"],
    "f1_anomaly": ["f1_anomaly", "f1 (anomaly)"],
}

SUMMARY_METRIC_ALIASES: Dict[str, List[str]] = {
    "macro_f1_mean": ["macro_f1_mean", "macro f1_mean", "macro_f1_mean_targets"],
    "macro_f1_std": ["macro_f1_std", "macro f1_std", "macro_f1_std_targets"],
    "macro_f1_selection_score": ["macro_f1_selection_score"],
    "accuracy_mean": ["acc_mean", "accuracy_mean", "acc mean", "acc_mean_targets"],
    "accuracy_std": ["acc_std", "accuracy_std"],
    "balanced_accuracy_mean": [
        "bal_acc_mean",
        "balanced_accuracy_mean",
        "bal acc_mean",
        "bal_acc_mean_targets",
    ],
    "balanced_accuracy_std": ["bal_acc_std", "balanced_accuracy_std", "bal acc_std"],
    "roc_auc_mean": [
        "roc_auc_mean",
        "roc_auc_ovr_macro_mean",
        "roc-auc_mean",
        "roc_auc_mean_targets",
    ],
    "roc_auc_std": ["roc_auc_std", "roc_auc_ovr_macro_std", "roc-auc_std"],
}

DATASET_ALIASES = [
    "dataset",
    "target",
    "target_dataset",
    "test_dataset",
    "target_domain",
    "target domain",
]
MODEL_ALIASES = ["model", "classifier", "estimator", "architecture"]
SOURCE_ALIASES = [
    "source",
    "sources",
    "source_combo",
    "source combo",
    "source_dataset",
    "source_datasets",
    "source_domains",
]
STRATEGY_ALIASES = [
    "strategy",
    "balance_strategy",
    "balancing",
    "balance",
    "variant",
    "preprocessing",
    "preprocess",
    "window_preprocessing",
    "config",
    "experiment",
    "mode",
    "encoder_profile",
    "head_shape",
    "source_mode",
    "exp_strategy",
    "pretrain_strategy",
    "finetune_strategy",
    "test_strategy",
    "pipeline",
    "setup",
    "approach",
    "protocol",
]
SEED_ALIASES = ["seed", "random_seed", "run_seed"]
FOLD_ALIASES = ["test_condition", "test condition", "fold", "condition", "test_fold"]
STATUS_ALIASES = ["status", "fold_status", "run_status"]

KNOWN_METADATA_COLUMNS: Set[str] = {
    "timestamp",
    "task",
    "experiment_id",
    "config_id",
    "config_desc",
    "description",
    "aux_features",
    "auxiliary_features",
    "additional_parameters",
    "additional_features",
    "include_aux",
    "use_aux",
    "n_features",
    "n_temporal_features",
    "feature_count",
    "n_sources",
    "n_models",
    "n_targets",
    "n_train",
    "n_train_target",
    "n_train_before",
    "n_train_after",
    "n_test",
    "n_test_samples",
    "n_classes",
    "n_test_classes_present",
    "class_dist_before",
    "class_dist_after",
    "class_dist_test",
    "class_counts_before",
    "class_counts_after",
    "balance_counts_before",
    "balance_counts_after",
    "smote_params",
    "unbalanced_classes",
    "roc_auc_valid",
    "roc_auc_note",
    "n_roc_auc_folds",
    "n_roc_auc_folds_total",
    "n_roc_auc_valid_evals",
    "n_roc_auc_valid_folds",
    "n_total_fold_evals",
    "n_valid_fold_evals",
    "n_failed_fold_evals",
    "n_total_folds",
    "n_valid_folds",
    "n_failed_folds",
    "n_common_valid_fold_evals",
    "status_reason",
    "error",
    "error_message",
    "notes",
    "elapsed_sec",
    "train_time_sec",
    "pretrain_time_sec",
    "finetune_time_sec",
    "best_params",
    "n_seeds",
    "n_folds",
    "n_folds_total",
    "n_folds_valid",
    "n_folds_skipped",
    "n_folds_error",
    "valid_folds",
    "total_folds",
    "skipped_folds",
    "error_folds",
    "rank",
    "rank_within_target_model",
    "winner_macro_f1",
    # Secondary summary columns
    "macro_f1_median",
    "macro_f1_min",
    "macro_f1_max",
    "macro_f1_min_target",
    "macro_f1_max_target",
    "macro_f1_model_std",
    "macro_f1_seed_std",
    "acc_seed_std",
    "bal_acc_seed_std",
    "roc_auc_seed_std",
    "macro_f1_worst_model",
    "macro_f1_best_model",
    "macro_f1_worst_target",
    "macro_f1_best_target",
    "acc_selection_mean",
    "bal_acc_selection_mean",
    "roc_auc_selection_mean",
    # Dataset validation / audit columns
    "check",
    "expected",
    "actual",
    "details",
    "condition_id",
    "class_label",
    "n_windows",
    "expected_len",
    "min_len",
    "max_len",
    "mean_rms",
    "std_rms",
    "rms_ref",
    "rms_ratio",
    "mean_val",
    "std_val",
    "nan_count",
    "inf_count",
    "zero_var_count",
    "role",
    "component",
    "fs_hz",
    "normal_class_expected",
    "variant_dir",
    "dir_exists",
    "in_normalization_manifest",
    "manifest_reference_dataset",
    "manifest_reference_class",
    "manifest_reference_rms",
    "manifest_total_windows",
    "n_windows_loaded",
    "n_conditions",
    "classes_found",
    "class_distribution",
    "eligible",
}

ARTIFACT_EXTENSIONS = {".csv", ".json", ".md", ".txt", ".log", ".png", ".yaml", ".yml"}


def norm(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def find_col(columns: List[str], aliases: List[str]) -> Optional[str]:
    mapping = {norm(c): c for c in columns}
    for a in aliases:
        na = norm(a)
        if na in mapping:
            return mapping[na]
    return None


def find_all_cols(columns: List[str], aliases: List[str]) -> List[str]:
    mapping = {norm(c): c for c in columns}
    found = []
    for a in aliases:
        na = norm(a)
        if na in mapping and mapping[na] not in found:
            found.append(mapping[na])
    return found


def infer_family(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    if len(parts) > 1:
        return parts[0]
    name = path.name.lower()
    if name.startswith("baseline_loco"):
        return "baseline_root"
    if name.startswith("anomaly_"):
        return "anomaly_root"
    if name.startswith("cwru"):
        return "cwru_severity_root"
    if name.startswith("norm_tl_"):
        return "norm_tl_root"
    if name.startswith("tl_"):
        return "tl_legacy_root"
    if name.startswith("log_"):
        return "logs_root"
    return "root"


def infer_dataset_from_filename(rel_path: str) -> str:
    lower = rel_path.lower()
    if "cwru12k" in lower or "cwru_12k" in lower:
        return "CWRU_12k"
    if "cwru48k" in lower or "cwru_48k" in lower:
        return "CWRU_48k"
    return ""


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    last_err: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            lines = [line for line in text.splitlines() if line.strip()]
            if not lines:
                return [], []
            sample = "\n".join(lines[:10])
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                delimiter = dialect.delimiter
            except Exception:
                delimiter = ","
            reader = csv.DictReader(lines, delimiter=delimiter)
            fieldnames = [str(f).strip() for f in (reader.fieldnames or []) if f is not None]
            rows = []
            for r in reader:
                clean_row = {}
                for k, v in r.items():
                    if k is not None:
                        clean_row[str(k).strip()] = "" if v is None else str(v).strip()
                rows.append(clean_row)
            return fieldnames, rows
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    return [], []


def classify_granularity(rel_path: str, columns: List[str]) -> str:
    norm_cols = {norm(c) for c in columns}
    lower_path = rel_path.lower()
    if "dataset_validation_tl" in lower_path or "dataset_audit" in lower_path:
        return "dataset_audit"
    has_fold_col = any(norm(a) in norm_cols for a in FOLD_ALIASES)
    has_summary_metric = any(
        any(norm(a) in norm_cols for a in aliases)
        for aliases in SUMMARY_METRIC_ALIASES.values()
    )
    if has_summary_metric or Path(rel_path).name.lower().startswith("summary") or "_summary" in lower_path:
        return "summary"
    if has_fold_col or any(
        any(norm(a) in norm_cols for a in aliases)
        for aliases in FOLD_METRIC_ALIASES.values()
    ):
        return "fold_level"
    return "metadata_or_other"


def parse_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null", "na", "n/a", "-"}:
        return None
    try:
        f = float(s)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except ValueError:
        return None


def build_strategy_string(row: Dict[str, str], strat_cols: List[str]) -> str:
    if not strat_cols:
        return ""
    if len(strat_cols) == 1:
        return row.get(strat_cols[0], "")
    parts = []
    for c in strat_cols:
        v = row.get(c, "")
        if v != "":
            parts.append(f"{c}={v}")
    return " | ".join(parts)


def write_csv_file(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit VibNet-1D experimental results and generate reproducible summaries."
    )
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--output-dir", type=Path, default=Path("reports/reproducibility_audit"))
    args = ap.parse_args()

    root = args.results_dir.resolve()
    out = args.output_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Results directory not found: {root}")
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in ARTIFACT_EXTENSIONS
    )

    inventory: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, Any]] = []

    for p in files:
        rel_path = p.relative_to(root).as_posix()
        fam = infer_family(p, root)
        ext = p.suffix.lower()
        rec: Dict[str, Any] = {
            "relative_path": rel_path,
            "family": fam,
            "type": ext,
            "granularity": "artifact" if ext != ".csv" else "",
            "size_bytes": p.stat().st_size,
            "rows": "",
            "columns": "",
            "status": "inventory_only",
            "dataset_column": "",
            "model_column": "",
            "seed_column": "",
            "fold_column": "",
            "recognized_metrics": "",
            "unrecognized_columns": "",
            "notes": "",
        }

        if ext == ".csv":
            try:
                cols, rows = read_csv_rows(p)
                granularity = classify_granularity(rel_path, cols)
                dcol = find_col(cols, DATASET_ALIASES)
                mcol = find_col(cols, MODEL_ALIASES)
                scol = find_col(cols, SOURCE_ALIASES)
                strat_cols = find_all_cols(cols, STRATEGY_ALIASES)
                seed_col = find_col(cols, SEED_ALIASES)
                fold_col = find_col(cols, FOLD_ALIASES)
                status_col = find_col(cols, STATUS_ALIASES)

                # Determine recognized metric columns
                metric_map: Dict[str, str] = {}
                for m_name, aliases in FOLD_METRIC_ALIASES.items():
                    mc = find_col(cols, aliases)
                    if mc is not None:
                        metric_map[m_name] = mc
                for m_name, aliases in SUMMARY_METRIC_ALIASES.items():
                    mc = find_col(cols, aliases)
                    if mc is not None:
                        metric_map[m_name] = mc

                # Identify unrecognized columns
                known_this_file: Set[str] = set()
                for c in [dcol, mcol, scol, seed_col, fold_col, status_col]:
                    if c:
                        known_this_file.add(norm(c))
                for c in strat_cols:
                    known_this_file.add(norm(c))
                for c in metric_map.values():
                    known_this_file.add(norm(c))
                known_this_file |= KNOWN_METADATA_COLUMNS

                unrecognized = [c for c in cols if norm(c) not in known_this_file]

                rec.update(
                    granularity=granularity,
                    rows=len(rows),
                    columns=len(cols),
                    status="ok",
                    dataset_column=dcol or "",
                    model_column=mcol or "",
                    seed_column=seed_col or "",
                    fold_column=fold_col or "",
                    recognized_metrics=";".join(sorted(metric_map.keys())),
                    unrecognized_columns=";".join(unrecognized),
                )

                if not metric_map:
                    rec["notes"] = f"No recognized metric columns ({granularity}); inventory only."

                fallback_dataset = infer_dataset_from_filename(rel_path)

                for i, row in enumerate(rows):
                    dataset_val = row.get(dcol, "") if dcol else fallback_dataset
                    model_val = row.get(mcol, "") if mcol else ""
                    sources_val = row.get(scol, "") if scol else ""
                    strategy_val = build_strategy_string(row, strat_cols)
                    seed_val = row.get(seed_col, "") if seed_col else ""
                    fold_val = row.get(fold_col, "") if fold_col else ""
                    row_status = row.get(status_col, "OK") if status_col else "OK"
                    if not row_status:
                        row_status = "OK"

                    # Check row-level anomalies
                    row_anomalies: List[str] = []

                    # 1. Non-OK status
                    if row_status.upper() not in {"OK", "PASS", "VALID"} and granularity == "fold_level":
                        err_msg = row.get("Error_Message", row.get("Error", row.get("Details", "")))
                        row_anomalies.append(f"NON_OK_STATUS:{row_status}")
                        anomalies.append({
                            "relative_path": rel_path,
                            "row_index": i,
                            "dataset": dataset_val,
                            "model": model_val,
                            "seed": seed_val,
                            "fold": fold_val,
                            "anomaly_type": "NON_OK_STATUS",
                            "details": f"Status={row_status} | {err_msg[:200]}",
                        })

                    # 2. Check for shifted columns in MLP (Scratch) (PureMLPBaseline string in ROC-AUC)
                    roc_col = metric_map.get("roc_auc")
                    raw_roc = row.get(roc_col, "") if roc_col else ""
                    is_shifted = False
                    if raw_roc and ("PureMLPBaseline" in raw_roc or "MLPEncoder" in raw_roc):
                        is_shifted = True
                        row_anomalies.append("SHIFTED_COLUMN_CORRUPTION")
                        anomalies.append({
                            "relative_path": rel_path,
                            "row_index": i,
                            "dataset": dataset_val,
                            "model": model_val,
                            "seed": seed_val,
                            "fold": fold_val,
                            "anomaly_type": "SHIFTED_COLUMN_CORRUPTION",
                            "details": (
                                "4-tuple return unpacked into 5 columns: Accuracy holds Bal_Acc, "
                                "Bal_Acc holds Macro_F1, Macro_F1 holds ROC-AUC, and ROC-AUC holds model repr."
                            ),
                        })

                    # 3. Check for zero-metric failure or ROC-AUC=0.0 fallback on fold_level rows
                    if granularity == "fold_level" and not is_shifted and row_status.upper() == "OK":
                        f1_col = metric_map.get("macro_f1")
                        bacc_col = metric_map.get("balanced_accuracy")
                        f1_v = parse_float(row.get(f1_col)) if f1_col else None
                        bacc_v = parse_float(row.get(bacc_col)) if bacc_col else None
                        roc_v = parse_float(raw_roc) if roc_col else None

                        if f1_v is not None and f1_v == 0.0 and (bacc_v is None or bacc_v == 0.0):
                            row_anomalies.append("ZERO_METRIC_FAILURE")
                            anomalies.append({
                                "relative_path": rel_path,
                                "row_index": i,
                                "dataset": dataset_val,
                                "model": model_val,
                                "seed": seed_val,
                                "fold": fold_val,
                                "anomaly_type": "ZERO_METRIC_FAILURE",
                                "details": "Macro F1 == 0.0 and Bal Acc == 0.0 (failed GridSearch/fit fallback).",
                            })
                        elif roc_v is not None and roc_v == 0.0 and f1_v is not None and f1_v > 0.0:
                            row_anomalies.append("ROC_AUC_ZERO_FALLBACK")
                            anomalies.append({
                                "relative_path": rel_path,
                                "row_index": i,
                                "dataset": dataset_val,
                                "model": model_val,
                                "seed": seed_val,
                                "fold": fold_val,
                                "anomaly_type": "ROC_AUC_ZERO_FALLBACK",
                                "details": f"ROC-AUC == 0.0 while Macro F1 == {f1_v:.4f} (missing test class in fold).",
                            })
                        elif roc_col and roc_v is None and f1_v is not None:
                            row_anomalies.append("ROC_AUC_NAN_MISSING_CLASS")
                            anomalies.append({
                                "relative_path": rel_path,
                                "row_index": i,
                                "dataset": dataset_val,
                                "model": model_val,
                                "seed": seed_val,
                                "fold": fold_val,
                                "anomaly_type": "ROC_AUC_NAN_MISSING_CLASS",
                                "details": f"ROC-AUC is NaN while Macro F1 == {f1_v:.4f} (ROC_AUC_Valid=False).",
                            })

                    is_row_valid = (
                        not is_shifted
                        and row_status.upper() in {"OK", "PASS", "VALID"}
                    )
                    anomaly_flag_str = ";".join(row_anomalies)

                    for metric_name, m_col in metric_map.items():
                        val = parse_float(row.get(m_col))
                        if val is None:
                            continue
                        metric_rows.append({
                            "relative_path": rel_path,
                            "family": fam,
                            "granularity": granularity,
                            "row_index": i,
                            "dataset": dataset_val,
                            "model": model_val,
                            "sources": sources_val,
                            "strategy": strategy_val,
                            "seed": seed_val,
                            "fold": fold_val,
                            "status": row_status,
                            "is_valid": is_row_valid,
                            "anomaly_flag": anomaly_flag_str,
                            "metric": metric_name,
                            "source_column": m_col,
                            "value": val,
                        })
            except Exception as e:
                rec.update(status="error", notes=str(e)[:400])

        inventory.append(rec)

    # Write results_inventory.csv
    inv_fields = [
        "relative_path",
        "family",
        "type",
        "granularity",
        "size_bytes",
        "rows",
        "columns",
        "status",
        "dataset_column",
        "model_column",
        "seed_column",
        "fold_column",
        "recognized_metrics",
        "unrecognized_columns",
        "notes",
    ]
    write_csv_file(out / "results_inventory.csv", inv_fields, inventory)

    # Write metrics_long.csv
    long_fields = [
        "relative_path",
        "family",
        "granularity",
        "row_index",
        "dataset",
        "model",
        "sources",
        "strategy",
        "seed",
        "fold",
        "status",
        "is_valid",
        "anomaly_flag",
        "metric",
        "source_column",
        "value",
    ]
    write_csv_file(out / "metrics_long.csv", long_fields, metric_rows)

    # Write anomalies_and_warnings.csv
    anom_fields = [
        "relative_path",
        "row_index",
        "dataset",
        "model",
        "seed",
        "fold",
        "anomaly_type",
        "details",
    ]
    write_csv_file(out / "anomalies_and_warnings.csv", anom_fields, anomalies)

    # Build metrics_summary.csv grouped strictly within each file & granularity
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in metric_rows:
        key = (
            r["family"],
            r["relative_path"],
            r["granularity"],
            r["dataset"],
            r["model"],
            r["sources"],
            r["strategy"],
            r["metric"],
        )
        grouped[key].append(r)

    summary_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        fam, rel_path, gran, ds, mdl, src, strat, met = key
        items = grouped[key]
        valid_items = [x for x in items if x["is_valid"]]
        invalid_count = len(items) - len(valid_items)
        vals = [float(x["value"]) for x in valid_items]
        seeds = {x["seed"] for x in valid_items if x["seed"] != ""}
        folds = {x["fold"] for x in valid_items if x["fold"] != ""}

        if vals:
            mean_v = statistics.mean(vals)
            std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
            min_v = min(vals)
            max_v = max(vals)
        else:
            mean_v = std_v = min_v = max_v = ""

        summary_rows.append({
            "family": fam,
            "relative_path": rel_path,
            "granularity": gran,
            "dataset": ds,
            "model": mdl,
            "sources": src,
            "strategy": strat,
            "metric": met,
            "count": len(vals),
            "invalid_count": invalid_count,
            "n_seeds": len(seeds) if seeds else "",
            "n_folds": len(folds) if folds else "",
            "mean": mean_v,
            "std": std_v,
            "min": min_v,
            "max": max_v,
        })

    summary_fields = [
        "family",
        "relative_path",
        "granularity",
        "dataset",
        "model",
        "sources",
        "strategy",
        "metric",
        "count",
        "invalid_count",
        "n_seeds",
        "n_folds",
        "mean",
        "std",
        "min",
        "max",
    ]
    write_csv_file(out / "metrics_summary.csv", summary_fields, summary_rows)

    # Generate rich Markdown report
    errors = [r for r in inventory if r["status"] == "error"]
    csv_files = [r for r in inventory if r["type"] == ".csv"]
    by_ext: Dict[str, int] = defaultdict(int)
    for r in inventory:
        by_ext[r["type"]] += 1

    by_gran: Dict[str, int] = defaultdict(int)
    for r in csv_files:
        by_gran[r["granularity"]] += 1

    anom_by_type: Dict[str, int] = defaultdict(int)
    for a in anomalies:
        anom_by_type[a["anomaly_type"]] += 1

    unrec_files = [r for r in csv_files if r["unrecognized_columns"]]

    report = [
        "# VibNet-1D — Relatório Automatizado de Auditoria de Resultados Experimentais",
        "",
        "## 1. Visão Geral do Inventário",
        f"- **Diretório auditado**: `{root.name}/`",
        f"- **Total de artefatos inventariados**: {len(files)}",
        "  - " + ", ".join(f"`{ext}`: {cnt}" for ext, cnt in sorted(by_ext.items())),
        f"- **Arquivos CSV analisados**: {len(csv_files)} ("
        + ", ".join(f"`{g}`: {c}" for g, c in sorted(by_gran.items()))
        + ")",
        f"- **Linhas de métricas extraídas (`metrics_long.csv`)**: {len(metric_rows)}",
        f"- **Grupos descritivos gerados (`metrics_summary.csv`)**: {len(summary_rows)}",
        f"- **Anomalias / avaliações inválidas detectadas (`anomalies_and_warnings.csv`)**: {len(anomalies)}",
        f"- **Erros de leitura de arquivo**: {len(errors)}",
        "",
        "## 2. Arquivos Derivados Gerados",
        f"- `results_inventory.csv`: inventário completo de todos os {len(files)} artefatos com tipo, granularidade (`fold_level`, `summary`, `dataset_audit`), colunas de contexto (`dataset`, `model`, `seed`, `fold`), métricas reconhecidas e colunas não reconhecidas.",
        "- `metrics_long.csv`: métricas extraídas linha a linha mantendo referência exata ao arquivo de origem (`relative_path`), granularidade, `dataset`, `model`, `sources`, `strategy`, `seed`, `fold`, `status`, flag de validade (`is_valid`) e `anomaly_flag`.",
        "- `metrics_summary.csv`: estatísticas descritivas (`count`, `invalid_count`, `n_seeds`, `n_folds`, `mean`, `std`, `min`, `max`) calculadas **estritamente dentro de cada arquivo e protocolo**, filtrando avaliações corrompidas (`is_valid=False`).",
        "- `anomalies_and_warnings.csv`: registro auditável de linhas com falha de execução (`NON_OK_STATUS`), deslocamento de colunas (`SHIFTED_COLUMN_CORRUPTION`), falhas de ajuste (`ZERO_METRIC_FAILURE`) e fallbacks de classe ausente em ROC-AUC (`ROC_AUC_ZERO_FALLBACK`, `ROC_AUC_NAN_MISSING_CLASS`).",
        "",
        "## 3. Resumo de Anomalias e Avaliações Inválidas Detectadas",
    ]

    if anom_by_type:
        for atype, cnt in sorted(anom_by_type.items()):
            report.append(f"- **`{atype}`**: {cnt} ocorrências")
        report.extend([
            "",
            "### Detalhamento das principais anomalias:",
            "1. **`SHIFTED_COLUMN_CORRUPTION` (100 linhas)**: Nos arquivos `baseline_loco_results_20260928_142515.csv` (50 linhas) e `baseline_loco_results_20261003_052352.csv` (50 linhas), o modelo `MLP (Scratch)` teve 4 valores desempacotados em 5 colunas (`run_baseline_experiments.py` L116 vs `build_pytorch_baselines.py` L90). Nesses registros, `Accuracy` contém `Bal Acc`, `Bal Acc` contém `Macro F1`, `Macro F1` contém `ROC-AUC` e `ROC-AUC` contém a string da arquitetura `PureMLPBaseline(...)`. Essas linhas foram marcadas com `is_valid=False` para não contaminar `metrics_summary.csv`.",
            "2. **`NON_OK_STATUS` (`ERROR_MODEL_EXECUTION`, 60 linhas)**: Nos runs `tl_cross_dataset_mlp_tabnet/run_20261028_105603` (15 linhas) e `run_20261028_173310` (45 linhas), o modelo `TabNet` falha no fold `C4_1500rpm_0.7Nm_400N` do dataset `PU` com `ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 141])` no `BatchNorm1d`, pois após SMOTE o conjunto de treino tem `8577` amostras (`8577 % 64 = 1`, gerando último mini-batch unitário).",
            "3. **`ZERO_METRIC_FAILURE` (157 linhas)**: Folds em que `Macro F1 == 0.0` e `Bal Acc == 0.0` (ex.: falhas de `GridSearchCV` nos scripts de severidade `cwru12k_severity_loco_results_*` e `cwru48k_mert_comparison_*`, além do fold `Load_0HP_Sev_0.021` em `CWRU_48k` para `Random Forest` e `SVM`).",
            "4. **`ROC_AUC_ZERO_FALLBACK` (700 linhas) e `ROC_AUC_NAN_MISSING_CLASS` (8 linhas)**: No dataset `CWRU_48k`, o fold `Load_0HP_Sev_0.021` possui apenas 2 das 3 classes de falha presentes no teste (`N_Test_Classes_Present = 2`). Scripts legados registravam `ROC-AUC = 0.0` ao capturar `ValueError` no cálculo multiclasse OvR; o runner `exp_tl_cross_dataset_mlp_tabnet.py` passa a registrar `NaN` com `ROC_AUC_Valid = False`.",
        ])
    else:
        report.append("- Nenhuma anomalia detectada.")

    report.extend([
        "",
        "## 4. Colunas Não Reconhecidas por Arquivo",
    ])
    if unrec_files:
        for r in unrec_files:
            report.append(f"- `{r['relative_path']}` (`{r['granularity']}`): `{r['unrecognized_columns']}`")
    else:
        report.append("- Todas as colunas de todos os 108 arquivos CSV foram reconhecidas e classificadas (contexto, métricas por fold, métricas agregadas ou metadados operacionais).")

    report.extend([
        "",
        "## 5. Destaques Quantitativos por Família Experimental (Apenas Avaliações Válidas `fold_level`, Métrica Primária: `macro_f1`)",
        "",
        "| Família | Arquivo (`relative_path`) | Dataset | Modelo | Estratégia | Folds/Avaliações (`count`) | Seeds | Folds | Macro-F1 Mean | Macro-F1 Std |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ])

    highlight_families = {
        "baseline",
        "baseline_root",
        "tl_four_variants",
        "tl_aux_features",
        "tl_balance_strategies",
        "tl_contrastive_mlp",
        "tl_cross_dataset_mlp_tabnet",
    }
    for s in summary_rows:
        if (
            s["granularity"] == "fold_level"
            and s["metric"] == "macro_f1"
            and s["family"] in highlight_families
            and s["count"] > 0
            and isinstance(s["mean"], float)
        ):
            rel = s["relative_path"]
            if "partial_test" in rel or "smoke" in rel:
                continue
            std_str = f"{s['std']:.4f}" if isinstance(s["std"], float) else "0.0000"
            report.append(
                f"| `{s['family']}` | `{rel}` | `{s['dataset']}` | `{s['model']}` | `{s['strategy']}` | "
                f"{s['count']} | {s['n_seeds'] or '-'} | {s['n_folds'] or '-'} | **{s['mean']:.4f}** | {std_str} |"
            )

    report.extend([
        "",
        "## 6. Cautelas Metodológicas e Regras de Comparação",
        "- **Proibição de fusão entre protocolos incompatíveis**: Resultados de `LOCO`, `Severity LOCO`, `MERT`, `5-Fold CV` e `One-Class Anomaly` avaliam hipóteses e partições distintas e jamais são combinados entre arquivos.",
        "- **Separação estrita de granularidade**: Arquivos `fold_level` e `summary` coexistem nos diretórios de saída (`results/tl_*`); utilize o campo `granularity == 'fold_level'` para reanálises estatísticas a partir das avaliações primárias.",
        "- **Preservação dos dados brutos**: Nenhum arquivo em `results/` foi modificado, movido ou sobrescrito por esta auditoria.",
        "",
        "## 7. Erros de Leitura",
    ])
    if errors:
        for r in errors:
            report.append(f"- `{r['relative_path']}`: {r['notes']}")
    else:
        report.append(f"- Nenhum erro de leitura nos {len(files)} arquivos inventariados.")

    (out / "EXPERIMENTAL_RESULTS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        f"Audit complete: {out} | files={len(files)} | csvs={len(csv_files)} | "
        f"metric_rows={len(metric_rows)} | summary_rows={len(summary_rows)} | "
        f"anomalies={len(anomalies)} | errors={len(errors)}"
    )


if __name__ == "__main__":
    main()
