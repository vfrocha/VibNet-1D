"""
Modular entrypoint for Supervised Contrastive Multi-Head MLP Transfer Learning
(`experiments/tabvibnet/run_contrastive_tl.py`).

Loads `configs/tabvibnet/tl_contrastive_mlp.yaml`, records execution provenance
(`run_metadata.json`), and delegates execution to `scripts/exp_tl_contrastive_mlp.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vibnet.data.registry import get_dataset_spec, load_yaml_config
from src.vibnet.evaluation.provenance import collect_run_provenance, write_run_metadata


def build_execution_plan(config_path: Path) -> Dict[str, Any]:
    """Load YAML config and summarize the Contrastive MLP ablation plan."""
    cfg = load_yaml_config(config_path)
    target = str(cfg.get("target_dataset", "CWRU_48k"))
    sources = list(cfg.get("source_datasets", ["CWRU_12k", "MFPT"]))
    seeds = [int(s) for s in cfg.get("seeds", [42, 123, 2026])]
    target_spec = get_dataset_spec(target)
    source_specs = {s: get_dataset_spec(s) for s in sources}

    return {
        "config_path": config_path.as_posix(),
        "config": cfg,
        "target_dataset": target,
        "source_datasets": sources,
        "seeds": seeds,
        "num_loco_folds": target_spec.num_conditions,
        "sampling_rates_hz": {
            target: target_spec.sampling_rate_hz,
            **{s: spec.sampling_rate_hz for s, spec in source_specs.items()},
        },
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TabVibNet Supervised Contrastive MLP Transfer Learning Runner"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tabvibnet" / "tl_contrastive_mlp.yaml",
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
        PROJECT_ROOT / "scripts" / "exp_tl_contrastive_mlp.py",
        PROJECT_ROOT / "src" / "models" / "build_mlp_contrastive.py",
    ]
    provenance = collect_run_provenance(
        project_root=PROJECT_ROOT,
        script_paths=tracked_scripts,
        seeds=plan["seeds"],
        extra_config=plan,
    )

    if args.write_metadata_to is not None:
        write_run_metadata(args.write_metadata_to, provenance)

    if args.dry_run:
        print(json.dumps({"plan": plan, "provenance": provenance}, indent=2))
        return 0

    from scripts.exp_tl_contrastive_mlp import main as legacy_main

    sys.argv = [sys.argv[0], *passthrough]
    legacy_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
