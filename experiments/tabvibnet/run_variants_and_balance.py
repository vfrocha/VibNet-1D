"""
Modular entrypoint for Signal Preprocessing Variants, Auxiliary Energy Features,
and Target-Only Balance Strategies (`experiments/tabvibnet/run_variants_and_balance.py`).

Loads `configs/tabvibnet/tl_variants_and_balance.yaml`, records execution provenance
(`run_metadata.json`), and delegates to:
- `scripts/exp_tl_four_variants.py` (`--stage variants`)
- `scripts/exp_tl_aux_features.py` (`--stage aux_features`)
- `scripts/exp_tl_balance_strategies.py` (`--stage balance`)
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


def build_execution_plan(config_path: Path, stage: str) -> Dict[str, Any]:
    """Load YAML config and return the execution plan for the selected stage."""
    cfg = load_yaml_config(config_path)
    target = str(cfg.get("target_dataset", "CWRU_48k"))
    sources = list(cfg.get("source_datasets", ["CWRU_12k", "MFPT"]))
    seeds = [int(s) for s in cfg.get("seeds", [42])]
    target_spec = get_dataset_spec(target)

    return {
        "stage": stage,
        "config_path": config_path.as_posix(),
        "config": cfg,
        "target_dataset": target,
        "source_datasets": sources,
        "seeds": seeds,
        "num_loco_folds": target_spec.num_conditions,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TabVibNet Preprocessing Variants & Target Balance Runner"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tabvibnet" / "tl_variants_and_balance.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--stage",
        choices=("variants", "aux_features", "balance"),
        default="variants",
        help="Experimental stage to run.",
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

    plan = build_execution_plan(args.config, args.stage)
    tracked_scripts = [
        Path(__file__),
        PROJECT_ROOT / "scripts" / "exp_tl_four_variants.py",
        PROJECT_ROOT / "scripts" / "exp_tl_aux_features.py",
        PROJECT_ROOT / "scripts" / "exp_tl_balance_strategies.py",
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

    sys.argv = [sys.argv[0], *passthrough]
    if args.stage == "variants":
        from scripts.exp_tl_four_variants import main as legacy_main
    elif args.stage == "aux_features":
        from scripts.exp_tl_aux_features import main as legacy_main
    else:
        from scripts.exp_tl_balance_strategies import main as legacy_main

    legacy_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
