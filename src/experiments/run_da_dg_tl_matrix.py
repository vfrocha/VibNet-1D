#!/usr/bin/env python3
"""
Entrypoint for the VibNet-1D Domain Adaptation (DA), Domain Generalization (DG),
and Transfer Learning (TL) experimental matrix (`src/experiments/run_da_dg_tl_matrix.py`).

Supports:
  - `--dry-run` (safe inspection & audit mode) to validate configurations, variable
    classification head dimensions (`S1`: 2, `S2`: 5, `S3`: 2, `S4`: 5), DA head
    expansion by canonical class name, recording-level partition manifests, RMS
    unit conversion (`1 g = 9.80665 m/s^2`) & geometric mean aggregation, 126/15
    feature scale-invariance catalog, CWRU_12k/48k external calibration compatibility,
    and hierarchical source balancing without running multi-hour training jobs or
    overwriting existing results.
  - `--export-partition-manifests <DIR>`: deterministically generates and saves the
    recording-level partition manifests (`JSON` and `CSV`) for all source configurations
    (`S1..S4`) and DA target datasets (`CWRU_12k`, `CWRU_48k`, `PU`) across all
    canonical seeds (`42`, `123`, `2026`).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR
from src.vibnet.data.registry import load_yaml_config
from src.vibnet.evaluation.da_dg_tl_protocol import (
    CANONICAL_SEEDS,
    DA_DG_TARGETS,
    DATASET_PHYSICAL_UNITS,
    HARMONIZED_CLASSES,
    SOURCE_CONFIGS,
    TL_TARGETS,
    WindowRecord,
    allocate_non_overwriting_run_dir,
    assess_source_rms_comparability,
    audit_cwru_12k_48k_calibration_compatibility,
    canonicalize_class_sequence,
    compute_dg_global_rms_reference_ms2,
    create_da_recording_split,
    create_source_recording_split,
    diagnose_source_config_class_coverage,
    get_141_feature_scale_invariance_catalog,
    get_scale_invariant_126_indices,
    get_scale_sensitive_15_indices,
    get_source_config_classes,
    load_dataset_window_records,
)
from src.vibnet.evaluation.da_dg_tl_runner import (
    MatrixExecutionContext,
    atomic_write_json,
    execute_da_job,
    execute_dg_job,
    execute_tl_job,
    is_job_completed,
)
from src.vibnet.evaluation.provenance import collect_run_provenance, write_run_metadata

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "da_dg_tl_matrix.yaml"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed_tl" / "detrended_rms"


def export_all_partition_manifests(
    data_root: Path,
    manifest_out_dir: Path,
    seeds: Sequence[int] = CANONICAL_SEEDS,
) -> Dict[str, Any]:
    """
    Generate and save reproducible recording-level partition manifests (`CSV` and `JSON`)
    for all source configurations (`S1..S4`) and all DA targets (`CWRU_12k`, `CWRU_48k`, `PU`)
    across all requested seeds (`42`, `123`, `2026`) (Section 8 & Deliverable 9).
    """
    manifest_out_dir.mkdir(parents=True, exist_ok=True)

    # Load records once per dataset from data/processed_tl/detrended_rms
    all_needed_datasets = sorted(
        {ds for src_list in SOURCE_CONFIGS.values() for ds in src_list}
        | set(DA_DG_TARGETS)
    )
    records_by_dataset: Dict[str, List[WindowRecord]] = {}
    for ds_name in all_needed_datasets:
        recs, _ = load_dataset_window_records(
            dataset_name=ds_name,
            data_root=data_root,
            load_waveforms=False,
        )
        records_by_dataset[ds_name] = recs

    all_recording_rows: List[Dict[str, Any]] = []
    summary_manifests: Dict[str, Any] = {
        "data_root": str(data_root),
        "seeds": [int(s) for s in seeds],
        "da_target_splits": {},
        "dg_source_splits": {},
    }

    # 1. DA target recording splits (20% adaptation / 80% test; CWRU_12k Normal exception: 2 adapt / 6 test)
    for tgt in DA_DG_TARGETS:
        summary_manifests["da_target_splits"][tgt] = {}
        tgt_records = records_by_dataset[tgt]
        for seed in seeds:
            _, man = create_da_recording_split(
                records=tgt_records,
                seed=int(seed),
                adapt_fraction=0.20,
                adapt_val_fraction=0.0,
            )
            summary_manifests["da_target_splits"][tgt][str(seed)] = {
                "split_signature": man.split_signature,
                "recording_counts_by_class": man.recording_counts_by_class,
                "window_counts_by_class": man.window_counts_by_class,
                "recording_proportions": man.recording_proportions,
                "window_proportions": man.window_proportions,
                "partition_recordings": {
                    k: list(v) for k, v in man.partition_recordings.items()
                },
            }
            for entry in man.recording_entries:
                row = entry.to_dict()
                row["protocol"] = "DA"
                row["config_or_target"] = tgt
                row["split_signature"] = man.split_signature
                all_recording_rows.append(row)

    # 2. DG / Source recording splits (80% source_train / 20% source_val)
    for sc_id, src_tuple in SOURCE_CONFIGS.items():
        summary_manifests["dg_source_splits"][sc_id] = {}
        sc_records: List[WindowRecord] = []
        for s_name in src_tuple:
            sc_records.extend(records_by_dataset[s_name])
        for seed in seeds:
            _, man = create_source_recording_split(
                records=sc_records,
                seed=int(seed),
                val_fraction=0.20,
                source_config_id=sc_id,
            )
            summary_manifests["dg_source_splits"][sc_id][str(seed)] = {
                "sources": list(src_tuple),
                "head_classes": list(get_source_config_classes(sc_id)),
                "head_output_dim": len(get_source_config_classes(sc_id)),
                "split_signature": man.split_signature,
                "recording_counts_by_class": man.recording_counts_by_class,
                "window_counts_by_class": man.window_counts_by_class,
                "recording_proportions": man.recording_proportions,
                "window_proportions": man.window_proportions,
                "partition_recordings": {
                    k: list(v) for k, v in man.partition_recordings.items()
                },
            }
            for entry in man.recording_entries:
                row = entry.to_dict()
                row["protocol"] = "DG_SOURCE"
                row["config_or_target"] = sc_id
                row["split_signature"] = man.split_signature
                all_recording_rows.append(row)

    json_path = manifest_out_dir / "da_dg_recording_partition_manifests.json"
    csv_path = manifest_out_dir / "da_dg_recording_partition_manifests.csv"

    json_path.write_text(
        json.dumps(summary_manifests, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fieldnames = [
        "protocol",
        "config_or_target",
        "dataset",
        "recording_key",
        "recording_prefix",
        "raw_class",
        "harmonized_class",
        "condition",
        "partition",
        "seed",
        "window_count",
        "split_signature",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_recording_rows)

    return {
        "json_manifest_path": str(json_path),
        "csv_manifest_path": str(csv_path),
        "total_recording_manifest_rows": len(all_recording_rows),
        "summary": summary_manifests,
    }


def build_execution_plan(
    config: Dict[str, Any],
    protocols: Sequence[str],
    source_configs: Sequence[str],
    targets: Optional[Sequence[str]],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    """Build a deterministic audit & execution plan for the requested matrix slice."""
    coverage_by_src = {
        sc: diagnose_source_config_class_coverage(sc) for sc in source_configs
    }
    # Audit both the pre-decimation MFPT reference reported in Section 5 (0.874 g)
    # and the harmonized 48.828 kHz on-disk MFPT reference (0.767595 g)
    audit_source_rms_reported = {
        "UORED": 5.85,
        "HUST": 0.181,
        "MFPT": 0.874,
    }
    observed_source_rms_ondisk = {
        "UORED": 5.850288142457682,
        "HUST": 0.18114055825864453,
        "MFPT": 0.7675950299980164,
    }
    observed_source_units = {
        "UORED": "m/s^2",
        "HUST": "m/s^2",
        "MFPT": "g",
    }
    rms_comparability_reported = assess_source_rms_comparability(
        source_rms_refs=audit_source_rms_reported,
        source_units=observed_source_units,
    )
    rms_comparability_ondisk = assess_source_rms_comparability(
        source_rms_refs=observed_source_rms_ondisk,
        source_units=observed_source_units,
    )

    dg_global_rms_by_config_reported = {
        sc: compute_dg_global_rms_reference_ms2(
            source_rms_refs=audit_source_rms_reported,
            source_units=observed_source_units,
            active_sources=SOURCE_CONFIGS[sc],
        )
        for sc in source_configs
    }
    dg_global_rms_by_config_ondisk = {
        sc: compute_dg_global_rms_reference_ms2(
            source_rms_refs=observed_source_rms_ondisk,
            source_units=observed_source_units,
            active_sources=SOURCE_CONFIGS[sc],
        )
        for sc in source_configs
    }

    target_present_classes_map: Dict[str, Tuple[str, ...]] = {
        "CWRU_12k": ("Normal", "Ball", "Inner", "Outer"),
        "CWRU_48k": ("Ball", "Inner", "Outer"),
        "PU": ("Normal", "Outer", "Inner & Outer"),
    }

    jobs: List[Dict[str, Any]] = []
    for proto in protocols:
        proto_targets = (
            list(targets)
            if targets
            else list(
                config.get("targets", {}).get(
                    proto, DA_DG_TARGETS if proto in ("DA", "DG") else TL_TARGETS
                )
            )
        )
        for sc in source_configs:
            src_classes = get_source_config_classes(sc)
            for tgt in proto_targets:
                if proto == "DG":
                    head_classes: Any = list(src_classes)
                    head_dim: Any = len(src_classes)
                    tgt_unit = DATASET_PHYSICAL_UNITS.get(tgt, "unknown")
                    if tgt_unit == "V":
                        exec_status = "BLOCKED_141_DG_INCOMPATIBLE_UNIT_VOLTS_VS_MS2"
                        unit_note = (
                            "PU signals are in Volts (V) without calibrated sensor sensitivity "
                            "(V -> m/s^2); 141-feature DG is blocked until unit conversion is "
                            "resolved or evaluated under 126-feature scale-invariant ablation."
                        )
                    else:
                        exec_status = "READY_PHYSICAL_UNIT_COMPATIBLE_MS2"
                        unit_note = (
                            f"Target unit '{tgt_unit}' convertible to m/s^2 (1 g = 9.80665 m/s^2)."
                        )
                elif proto == "DA":
                    tgt_classes = target_present_classes_map.get(tgt, HARMONIZED_CLASSES)
                    expanded = canonicalize_class_sequence(list(src_classes) + list(tgt_classes))
                    head_classes = list(expanded)
                    head_dim = len(expanded)
                    exec_status = "READY_INTERNAL_OR_RELATED_ADAPT_NORMAL_RMS"
                    unit_note = (
                        "DA uses target's own adaptation Normal RMS_ref (or CWRU_12k adaptation "
                        "Normal for CWRU_48k); self-consistent within target domain."
                    )
                else:
                    head_classes = "native_target_vocabulary"
                    head_dim = "native_target_num_classes"
                    exec_status = "READY_TARGET_LOCO_TRAIN_NORMAL_RMS"
                    unit_note = "TL uses target LOCO training conditions Normal RMS_ref."

                for seed in seeds:
                    jobs.append(
                        {
                            "protocol": proto,
                            "source_config": sc,
                            "sources": list(SOURCE_CONFIGS[sc]),
                            "target": tgt,
                            "target_physical_unit": DATASET_PHYSICAL_UNITS.get(tgt, "unknown"),
                            "seed": int(seed),
                            "initial_source_head_classes": list(src_classes),
                            "initial_source_head_dim": len(src_classes),
                            "evaluation_head_classes": head_classes,
                            "evaluation_head_dim": head_dim,
                            "execution_status_141_features": exec_status,
                            "physical_unit_note": unit_note,
                        }
                    )

    return {
        "experiment_id": config.get("experiment_id", "da_dg_tl_matrix"),
        "protocols": list(protocols),
        "source_configs": list(source_configs),
        "seeds": [int(s) for s in seeds],
        "total_planned_runs": len(jobs),
        "source_variable_head_coverage": coverage_by_src,
        "dg_global_rms_ms2_by_config_audit_values": dg_global_rms_by_config_reported,
        "dg_global_rms_ms2_by_config_harmonized_ondisk": dg_global_rms_by_config_ondisk,
        "dg_source_rms_comparability_audit_values": rms_comparability_reported,
        "dg_source_rms_comparability_harmonized_ondisk": rms_comparability_ondisk,
        "cwru_12k_48k_calibration_audit": audit_cwru_12k_48k_calibration_compatibility(),
        "feature_scale_invariance_summary": {
            "total_features": 141,
            "scale_invariant_count": len(get_scale_invariant_126_indices()),
            "scale_sensitive_count": len(get_scale_sensitive_15_indices()),
            "scale_sensitive_indices": list(get_scale_sensitive_15_indices()),
        },
        "jobs_preview": jobs[:15],
        "jobs": jobs,
    }


def _make_job_id(protocol: str, source_config: str, target: str, seed: int) -> str:
    return f"{protocol}__{source_config}__{target}__seed{int(seed)}"


def consolidate_matrix_summary(run_dir: Path) -> Dict[str, Any]:
    """
    Rebuild `matrix_summary.json` and `matrix_summary.csv` atomically from all
    `jobs/*/COMPLETED.json` files inside `run_dir`. Never modifies individual job dirs.
    """
    jobs_root = run_dir / "jobs"
    rows: List[Dict[str, Any]] = []
    completed_count = 0
    blocked_count = 0

    if jobs_root.is_dir():
        for job_dir in sorted(jobs_root.iterdir()):
            sentinel = job_dir / "COMPLETED.json"
            if not sentinel.is_file():
                continue
            try:
                data = json.loads(sentinel.read_text(encoding="utf-8"))
            except Exception:
                continue
            status = str(data.get("status", "UNKNOWN"))
            if status == "COMPLETED":
                completed_count += 1
                models_f1 = data.get("models_macro_f1", {})
                for model_name, macro_f1 in sorted(models_f1.items()):
                    rows.append(
                        {
                            "job_id": job_dir.name,
                            "status": status,
                            "protocol": data.get("protocol", ""),
                            "source_config": data.get("source_config", ""),
                            "target": data.get("target", ""),
                            "seed": data.get("seed", ""),
                            "model": model_name,
                            "macro_f1": macro_f1,
                            "elapsed_seconds": data.get("elapsed_seconds", ""),
                            "timestamp_utc": data.get("timestamp_utc", ""),
                        }
                    )
            elif status == "BLOCKED_INCOMPATIBLE_UNIT":
                blocked_count += 1
                rows.append(
                    {
                        "job_id": job_dir.name,
                        "status": status,
                        "protocol": data.get("protocol", ""),
                        "source_config": data.get("source_config", ""),
                        "target": data.get("target", ""),
                        "seed": data.get("seed", ""),
                        "model": "BLOCKED",
                        "macro_f1": "",
                        "elapsed_seconds": 0.0,
                        "timestamp_utc": data.get("timestamp_utc", ""),
                    }
                )

    summary = {
        "run_dir": str(run_dir),
        "completed_jobs": completed_count,
        "blocked_jobs": blocked_count,
        "total_summary_rows": len(rows),
        "rows": rows,
    }
    atomic_write_json(run_dir / "matrix_summary.json", summary)

    csv_path = run_dir / "matrix_summary.csv"
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    fieldnames = [
        "job_id",
        "status",
        "protocol",
        "source_config",
        "target",
        "seed",
        "model",
        "macro_f1",
        "elapsed_seconds",
        "timestamp_utc",
    ]
    with tmp_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_csv.replace(csv_path)
    return summary


def execute_matrix_plan(
    config: Dict[str, Any],
    plan: Dict[str, Any],
    run_dir: Path,
    include_classical: bool = True,
    max_loco_folds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute all jobs in `plan["jobs"]` with safe atomic resume:
      - Jobs whose `jobs/<job_id>/COMPLETED.json` already exists are skipped without overwriting.
      - DG jobs targeting `PU` (Volts) under 141 features are recorded as `BLOCKED_INCOMPATIBLE_UNIT`.
      - After every job, consolidates `matrix_summary.csv` and `matrix_summary.json`.
    """
    ctx = MatrixExecutionContext(run_dir=run_dir, config=config, project_root=PROJECT_ROOT)
    jobs_root = run_dir / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict[str, Any]] = list(plan.get("jobs", []))
    executed_now = 0
    skipped_resume = 0
    blocked_now = 0

    for idx, job in enumerate(jobs, start=1):
        proto = str(job["protocol"])
        sc = str(job["source_config"])
        tgt = str(job["target"])
        seed = int(job["seed"])
        job_id = _make_job_id(proto, sc, tgt, seed)
        job_dir = jobs_root / job_id

        if is_job_completed(job_dir):
            skipped_resume += 1
            print(f"[{idx}/{len(jobs)}] [SKIP-RESUME] {job_id} already completed.")
            continue

        print("=" * 78)
        print(f"[{idx}/{len(jobs)}] [RUN] {job_id}")
        print("=" * 78)
        if proto == "DG":
            res = execute_dg_job(
                ctx=ctx,
                source_config_id=sc,
                target_dataset=tgt,
                seed=seed,
                job_dir=job_dir,
                include_classical=include_classical,
            )
        elif proto == "DA":
            res = execute_da_job(
                ctx=ctx,
                source_config_id=sc,
                target_dataset=tgt,
                seed=seed,
                job_dir=job_dir,
                include_classical=include_classical,
            )
        elif proto == "TL":
            res = execute_tl_job(
                ctx=ctx,
                source_config_id=sc,
                target_dataset=tgt,
                seed=seed,
                job_dir=job_dir,
                include_classical=include_classical,
                max_loco_folds=max_loco_folds,
            )
        else:
            raise ValueError(f"Unsupported protocol '{proto}'.")

        if res.get("status") == "BLOCKED_INCOMPATIBLE_UNIT":
            blocked_now += 1
            print(f"[{idx}/{len(jobs)}] [BLOCKED] {job_id}: {res.get('reason')}")
        else:
            executed_now += 1
            print(f"[{idx}/{len(jobs)}] [DONE] {job_id} ({res.get('elapsed_seconds')}s)")

        consolidate_matrix_summary(run_dir)

    summary = consolidate_matrix_summary(run_dir)
    return {
        "run_dir": str(run_dir),
        "total_planned_jobs": len(jobs),
        "executed_now": executed_now,
        "skipped_resume": skipped_resume,
        "blocked_now": blocked_now,
        "summary": summary,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Orquestrador reprodutível para a matriz experimental DA, DG e TL (VibNet-1D)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Caminho para o arquivo YAML de configuração (default: configs/da_dg_tl_matrix.yaml).",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["DA", "DG", "TL"],
        choices=["DA", "DG", "TL"],
        help="Protocolos experimentais a planejar/executar.",
    )
    parser.add_argument(
        "--source-configs",
        nargs="+",
        default=["S1", "S2", "S3", "S4"],
        choices=["S1", "S2", "S3", "S4"],
        help="Configurações de origem (S1=UORED, S2=HUST, S3=MFPT, S4=UORED+HUST+MFPT).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Subconjunto opcional de datasets de destino.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(CANONICAL_SEEDS),
        help="Seeds de reprodutibilidade (default: 42 123 2026).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Raiz dos datasets normalizados para exportação de manifestos (default: data/processed_tl/detrended_rms).",
    )
    parser.add_argument(
        "--export-partition-manifests",
        type=Path,
        default=None,
        help="Exporta os manifestos reproduzíveis de partições por gravação (CSV e JSON) para o diretório informado.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_DIR / "da_dg_tl_matrix",
        help="Diretório raiz para salvar artefatos sem sobrescrever execuções anteriores.",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Diretório de uma execução anterior para retomada segura (pula jobs com COMPLETED.json sem sobrescrever).",
    )
    parser.add_argument(
        "--max-loco-folds",
        type=int,
        default=None,
        help="Limite opcional de folds LOCO no TL (default: None = todos os folds).",
    )
    parser.add_argument(
        "--skip-classical-baselines",
        action="store_true",
        help="Se definido, pula baselines clássicos (RF/SVM). Por padrão RF e SVM são executados.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida a configuração, diagnósticos de cobertura/RMS e plano sem iniciar treinamentos longos.",
    )
    args = parser.parse_args(argv)

    cfg = load_yaml_config(args.config)
    plan = build_execution_plan(
        config=cfg,
        protocols=args.protocols,
        source_configs=args.source_configs,
        targets=args.targets,
        seeds=args.seeds,
    )

    if args.export_partition_manifests is not None:
        manifest_info = export_all_partition_manifests(
            data_root=args.data_root,
            manifest_out_dir=args.export_partition_manifests,
            seeds=args.seeds,
        )
        plan["exported_partition_manifests"] = {
            "json_manifest_path": manifest_info["json_manifest_path"],
            "csv_manifest_path": manifest_info["csv_manifest_path"],
            "total_recording_manifest_rows": manifest_info["total_recording_manifest_rows"],
        }

    if args.dry_run:
        plan_print = {k: v for k, v in plan.items() if k != "jobs"}
        print(json.dumps(plan_print, indent=2, ensure_ascii=False))
        return 0

    if args.resume_dir is not None:
        run_dir = args.resume_dir.expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
        print(f"[RESUME] Retomando execução segura em: {run_dir}")
    else:
        run_dir = allocate_non_overwriting_run_dir(args.output_root, experiment_prefix="matrix_run")
        provenance = collect_run_provenance(
            project_root=PROJECT_ROOT,
            script_paths=[
                __file__,
                "src/vibnet/evaluation/da_dg_tl_protocol.py",
                "src/vibnet/evaluation/da_dg_tl_runner.py",
                "configs/da_dg_tl_matrix.yaml",
            ],
            seeds=args.seeds,
            extra_config={k: v for k, v in plan.items() if k != "jobs"},
        )
        write_run_metadata(run_dir, provenance)
        atomic_write_json(run_dir / "execution_plan.json", plan)
        print(f"[NEW RUN] Diretório de execução criado: {run_dir}")

    exec_report = execute_matrix_plan(
        config=cfg,
        plan=plan,
        run_dir=run_dir,
        include_classical=not bool(args.skip_classical_baselines),
        max_loco_folds=args.max_loco_folds,
    )
    print("=" * 78)
    print(
        f"[MATRIX SUMMARY] run_dir={exec_report['run_dir']} | "
        f"executed_now={exec_report['executed_now']} | "
        f"skipped_resume={exec_report['skipped_resume']} | "
        f"blocked_now={exec_report['blocked_now']}"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
