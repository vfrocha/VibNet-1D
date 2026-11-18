#!/usr/bin/env python3
"""
Entrypoint for the VibNet-1D Domain Adaptation (DA), Domain Generalization (DG),
and Transfer Learning (TL) experimental matrix (`src/experiments/run_da_dg_tl_matrix.py`).

Supports `--dry-run` (default safe inspection & audit mode) to validate configurations,
source-to-5-class coverage, recording-level partition manifests, RMS comparability,
and hierarchical source balancing without running multi-hour training jobs or
overwriting existing results.
"""

from __future__ import annotations

import argparse
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
    HARMONIZED_CLASSES,
    SOURCE_CONFIGS,
    TL_TARGETS,
    allocate_non_overwriting_run_dir,
    assess_source_rms_comparability,
    diagnose_source_config_class_coverage,
)
from src.vibnet.evaluation.provenance import collect_run_provenance, write_run_metadata

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "da_dg_tl_matrix.yaml"


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
    observed_source_rms = {
        "UORED": 5.850288,
        "HUST": 0.181141,
        "MFPT": 0.873542,
    }
    observed_source_units = {
        "UORED": "m/s^2",
        "HUST": "m/s^2",
        "MFPT": "g",
    }
    rms_comparability = assess_source_rms_comparability(
        source_rms_refs=observed_source_rms,
        source_units=observed_source_units,
    )

    jobs: List[Dict[str, Any]] = []
    for proto in protocols:
        proto_targets = (
            list(targets)
            if targets
            else list(config.get("targets", {}).get(proto, DA_DG_TARGETS if proto in ("DA", "DG") else TL_TARGETS))
        )
        for sc in source_configs:
            for tgt in proto_targets:
                for seed in seeds:
                    jobs.append(
                        {
                            "protocol": proto,
                            "source_config": sc,
                            "sources": list(SOURCE_CONFIGS[sc]),
                            "target": tgt,
                            "seed": int(seed),
                            "head_classes": list(HARMONIZED_CLASSES)
                            if proto in ("DA", "DG")
                            else "native_target_vocabulary",
                        }
                    )

    return {
        "experiment_id": config.get("experiment_id", "da_dg_tl_matrix"),
        "protocols": list(protocols),
        "source_configs": list(source_configs),
        "seeds": [int(s) for s in seeds],
        "total_planned_runs": len(jobs),
        "source_5class_coverage": coverage_by_src,
        "dg_source_rms_comparability_audit": rms_comparability,
        "jobs_preview": jobs[:15],
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
        "--output-root",
        type=Path,
        default=RESULTS_DIR / "da_dg_tl_matrix",
        help="Diretório raiz para salvar artefatos sem sobrescrever execuções anteriores.",
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

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    run_dir = allocate_non_overwriting_run_dir(args.output_root, experiment_prefix="run")
    provenance = collect_run_provenance(
        project_root=PROJECT_ROOT,
        script_paths=[
            __file__,
            "src/vibnet/evaluation/da_dg_tl_protocol.py",
            "configs/da_dg_tl_matrix.yaml",
        ],
        seeds=args.seeds,
        extra_config=plan,
    )
    write_run_metadata(run_dir, provenance)
    (run_dir / "execution_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[OK] Plano e metadados de proveniência registrados em: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

