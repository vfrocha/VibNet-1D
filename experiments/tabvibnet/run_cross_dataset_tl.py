"""
Modular entrypoint for Cross-Dataset Multi-Source Transfer Learning (`MLP` vs `TabNet`)
(`experiments/tabvibnet/run_cross_dataset_tl.py`).

Loads `configs/tabvibnet/tl_cross_dataset_mlp_tabnet.yaml`, validates dataset metadata
via `src.vibnet.data.registry`, records execution provenance (`run_metadata.json`) via
`src.vibnet.evaluation.provenance`, and delegates full execution to
`scripts/exp_tl_cross_dataset_mlp_tabnet.py`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vibnet.data.registry import get_dataset_spec, load_yaml_config
from src.vibnet.evaluation.protocols import build_source_combinations
from src.vibnet.evaluation.provenance import collect_run_provenance, write_run_metadata


def build_execution_plan(config_path: Path) -> Dict[str, Any]:
    """Load YAML config and compute the exact target/source/fold execution plan."""
    cfg = load_yaml_config(config_path)
    targets: List[str] = list(
        cfg.get("targets", ["CWRU_12k", "CWRU_48k", "HUST_Gearbox", "PU", "UOEMD"])
    )
    sources: List[str] = list(
        cfg.get("sources", ["UORED", "HUST", "Mechanical_Gear", "Electric_Motor"])
    )
    source_modes: List[str] = list(
        cfg.get("source_modes", ["single", "pairs", "triplets", "all"])
    )
    seeds: List[int] = [int(s) for s in cfg.get("seeds", [42, 123, 2026])]
    models: List[str] = list(cfg.get("models", ["MLP", "TabNet"]))

    target_specs = {t: get_dataset_spec(t) for t in targets}
    source_specs = {s: get_dataset_spec(s) for s in sources}
    combos = build_source_combinations(sources, modes=source_modes)

    total_loco_folds = sum(spec.num_conditions for spec in target_specs.values())
    expected_rows = total_loco_folds * len(combos) * len(seeds) * len(models)

    return {
        "config_path": config_path.as_posix(),
        "config": cfg,
        "targets": targets,
        "sources": sources,
        "source_modes": source_modes,
        "num_source_combinations": len(combos),
        "source_combinations": ["|".join(c) for c in combos],
        "seeds": seeds,
        "models": models,
        "total_loco_folds_per_combo": total_loco_folds,
        "expected_total_rows": expected_rows,
        "sampling_rates_hz": {
            name: spec.sampling_rate_hz
            for name, spec in {**target_specs, **source_specs}.items()
        },
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TabVibNet Cross-Dataset Multi-Source Transfer Learning Runner"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tabvibnet" / "tl_cross_dataset_mlp_tabnet.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print execution plan without starting training.",
    )
    parser.add_argument(
        "--write-metadata-to",
        type=Path,
        default=None,
        help="Optional directory to write run_metadata.json.",
    )
    args, passthrough = parser.parse_known_args(argv)

    plan = build_execution_plan(args.config)
    tracked_scripts = [
        Path(__file__),
        PROJECT_ROOT / "scripts" / "exp_tl_cross_dataset_mlp_tabnet.py",
        PROJECT_ROOT / "src" / "models" / "build_tabnet_resnet_balance.py",
        PROJECT_ROOT / "src" / "features" / "extractors_v2.py",
        PROJECT_ROOT / "src" / "features" / "signalai_wrapper.py",
    ]
    provenance = collect_run_provenance(
        project_root=PROJECT_ROOT,
        script_paths=tracked_scripts,
        seeds=plan["seeds"],
        extra_config=plan,
        fold_accounting={
            "expected_folds": plan["expected_total_rows"],
            "valid_folds": 0,
            "failed_folds": 0,
            "error_breakdown": {},
        },
    )

    if args.write_metadata_to is not None:
        write_run_metadata(args.write_metadata_to, provenance)

    if args.dry_run:
        print(json.dumps({"plan": plan, "provenance": provenance}, indent=2))
        return 0

    from scripts.exp_tl_cross_dataset_mlp_tabnet import main as legacy_main

    run_ts = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    default_out = PROJECT_ROOT / "results" / "tl_cross_dataset_mlp_tabnet" / run_ts
    write_run_metadata(default_out, provenance)
    sys.argv = [sys.argv[0], *passthrough]
    legacy_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
